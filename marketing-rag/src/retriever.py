"""
retriever.py
Hybrid retrieval: BM25 (sparse) + Qdrant dense + Reciprocal Rank Fusion + Gemini reranker.

Pipeline:
  1. Dense retrieval: Qdrant top-20 by cosine similarity
  2. Sparse retrieval: BM25 over in-memory corpus (all chunk texts)
  3. Score fusion: Reciprocal Rank Fusion (RRF) to merge ranked lists
  4. Reranker: Gemini Flash pointwise scorer — take top-5

The BM25 index is built once on startup and kept in memory.
The Qdrant client uses the persistent on-disk storage written by ingest.py.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import math

logger = logging.getLogger(__name__)

COLLECTION_NAME  = "marketing_docs"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"   # local, no API, 384-dim
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIM        = 384
QDRANT_PATH = Path(__file__).parent.parent / "data" / "qdrant_storage"

# Load embedding model once at module level
logger.info("Loading embedding model...")
_embed_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info(f"Embedding model ready (dim={_embed_model.get_embedding_dimension()})")

logger.info("Loading cross-encoder reranker...")
_reranker = CrossEncoder(RERANKER_MODEL)
logger.info("Reranker ready")


# ── BM25 index (built once, cached in module scope) ────────────────────────────

_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: List[Dict[str, Any]] = []  # parallel list to index


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    import re
    return re.findall(r'\w+', text.lower())


def build_bm25_index(chunks: List[Dict[str, Any]]) -> None:
    """Build the BM25 index from a list of chunk dicts. Called once at startup."""
    global _bm25_index, _bm25_corpus
    _bm25_corpus = chunks
    tokenized = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 index built: {len(chunks)} chunks")


def _bm25_search(query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
    """Run BM25 search, return list of (chunk, score) sorted descending."""
    if _bm25_index is None:
        raise RuntimeError("BM25 index not built. Call build_bm25_index() first.")
    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(_bm25_corpus[i], float(score)) for i, score in ranked if score > 0]


# ── Qdrant dense retrieval ─────────────────────────────────────────────────────

def _dense_search(
    qdrant: QdrantClient,
    query: str,
    top_k: int = 20,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Embed query locally (BGE prefix: 'Represent this sentence for searching relevant passages:')
    then run cosine similarity search in Qdrant.
    BGE models expect this specific prefix for queries (not for documents).
    """
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    vector = _embed_model.encode(
        prefixed_query, normalize_embeddings=True
    ).tolist()

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [(hit.payload, float(hit.score)) for hit in hits]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_fuse(
    dense_results: List[Tuple[Dict[str, Any], float]],
    sparse_results: List[Tuple[Dict[str, Any], float]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.
    Chunks are identified by (doc_id, chunk_index). Returns merged list sorted by RRF score.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    def key(chunk: Dict[str, Any]) -> str:
        return f"{chunk['doc_id']}_{chunk['chunk_index']}"

    for rank, (chunk, _) in enumerate(dense_results):
        k_ = key(chunk)
        scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[k_] = chunk

    for rank, (chunk, _) in enumerate(sparse_results):
        k_ = key(chunk)
        scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[k_] = chunk

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[k] for k, _ in merged]


# ── Local reranker ────────────────────────────────────────────────────────────

def _rerank_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank chunks using local ms-marco-MiniLM-L-6-v2 cross-encoder.
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        # MS MARCO logits are roughly -10 to +10. Use sigmoid to get a 0-1 confidence score.
        c["relevance_score"] = 1.0 / (1.0 + math.exp(-float(score)))

    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    return candidates


# ── Public API ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Stateful retriever. Holds the Qdrant client and BM25 index.
    Instantiate once at server startup.
    """

    def __init__(self):
        if not QDRANT_PATH.exists():
            raise RuntimeError(
                f"Qdrant storage not found at {QDRANT_PATH}. "
                "Run scripts/ingest.py first."
            )
        self.qdrant = QdrantClient(path=str(QDRANT_PATH))
        self._load_bm25_corpus()
        logger.info("Retriever initialized")

    def _load_bm25_corpus(self):
        """Load all chunk texts from Qdrant payload to build BM25 index."""
        all_chunks = []
        offset = None
        while True:
            result = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                all_chunks.append(point.payload)
            if next_offset is None:
                break
            offset = next_offset

        build_bm25_index(all_chunks)
        logger.info(f"BM25 corpus loaded: {len(all_chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        top_n_rerank: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline:
          dense → sparse → RRF → rerank → top_n_rerank chunks
        """
        dense_hits = _dense_search(self.qdrant, query, top_k=top_k_dense)
        sparse_hits = _bm25_search(query, top_k=top_k_sparse)
        fused = _rrf_fuse(dense_hits, sparse_hits)
        top_fused = fused[:15]
        reranked = _rerank_candidates(query, top_fused)

        # 4. Return top 5
        top_k = min(5, len(reranked))
        return reranked[:top_k]

    def retrieve_for_eval(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[str]:
        """
        Retrieval eval helper: returns ordered list of doc_ids after local reranking.
        """
        dense_hits = _dense_search(self.qdrant, query, top_k=20)
        sparse_hits = _bm25_search(query, top_k=20)
        fused = _rrf_fuse(dense_hits, sparse_hits)
        top_fused = fused[:15]
        reranked = _rerank_candidates(query, top_fused)
        return [c["doc_id"] for c in reranked[:top_k]]
