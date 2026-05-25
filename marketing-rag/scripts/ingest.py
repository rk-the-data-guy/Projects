"""
ingest.py
Chunk → embed → upsert all corpus documents into Qdrant.

Embedding: BAAI/bge-small-en-v1.5 (sentence-transformers, runs locally on CPU).
  - No API calls, no rate limits, no cost.
  - 384-dim vectors, top-tier quality for its size (MTEB leaderboard).
  - 520 docs embed in ~30 seconds on CPU.

Qdrant state is persisted to data/qdrant_storage.
Script is resumable: re-running skips already-indexed points.

Run:
  python scripts/ingest.py          # resume mode (default)
  python scripts/ingest.py --no-resume   # fresh start
"""

import json
import sys
import time
import tiktoken
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
CORPUS_DIR   = Path(__file__).parent.parent / "data" / "corpus"
QDRANT_PATH  = Path(__file__).parent.parent / "data" / "qdrant_storage"
COLLECTION_NAME   = "marketing_docs"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"  # 384-dim, no API, runs on CPU
EMBED_DIM         = 384
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
BATCH_SIZE = 64  # local model: large batches are fine and fast

enc = tiktoken.get_encoding("cl100k_base")

# Load model once at module level (cached to ~/.cache/huggingface after first download)
print("⏳ Loading embedding model (downloads once, ~33MB)...")
_embed_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Model ready — dim={_embed_model.get_embedding_dimension()}")


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Sentence-aware chunker: splits on sentence/paragraph boundaries,
    groups into chunks within the token budget, with overlap carry-over.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_tokens: List[int] = []
    current_sentences: List[str] = []

    for sentence in sentences:
        s_tokens = enc.encode(sentence)
        if len(current_tokens) + len(s_tokens) > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Overlap: carry the last `overlap` tokens worth of sentences
            overlap_sentences: List[str] = []
            overlap_count = 0
            for sent in reversed(current_sentences):
                t = enc.encode(sent)
                if overlap_count + len(t) <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_count += len(t)
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = enc.encode(" ".join(current_sentences))

        current_sentences.append(sentence)
        current_tokens.extend(s_tokens)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using the local BGE model.
    BGE models expect a query prefix for queries, but for documents
    (during ingestion) no prefix is needed.
    Returns list of float lists ready for Qdrant.
    """
    # encode() returns numpy array — convert to plain Python lists
    vectors = _embed_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(resume: bool = True):
    print(f"\n📂 Loading corpus from {CORPUS_DIR}")
    doc_files = sorted(CORPUS_DIR.glob("*.json"))
    if not doc_files:
        print("❌ No documents found. Run scripts/synthesize_corpus.py first.")
        sys.exit(1)
    print(f"   Found {len(doc_files)} documents")

    # Set up Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(QDRANT_PATH))

    # Resume support
    already_upserted: set = set()
    if resume and qdrant.collection_exists(COLLECTION_NAME):
        print(f"🔄 Resume mode: scanning existing collection...")
        offset = None
        while True:
            pts, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=500, offset=offset,
                with_payload=False, with_vectors=False,
            )
            for p in pts:
                already_upserted.add(p.id)
            if next_offset is None:
                break
            offset = next_offset
        print(f"   {len(already_upserted)} chunks already indexed — skipping them")
    else:
        if qdrant.collection_exists(COLLECTION_NAME):
            qdrant.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Deleted existing collection")

    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"✅ Created collection '{COLLECTION_NAME}' (dim={EMBED_DIM})")

    # Build all chunks
    all_chunks: List[Dict[str, Any]] = []
    for doc_file in doc_files:
        with open(doc_file) as f:
            doc = json.load(f)
        body = doc.get("body", "")
        if not body:
            continue
        chunks = chunk_text(body)
        for chunk_idx, chunk_text_str in enumerate(chunks):
            point_id = abs(hash(f"{doc['doc_id']}_{chunk_idx}")) % (2**31)
            all_chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "date": doc.get("date", ""),
                "campaign": doc.get("campaign", ""),
                "brand": doc.get("brand", ""),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "text": chunk_text_str,
                "point_id": point_id,
            })

    total_chunks = len(all_chunks)
    pending = [c for c in all_chunks if c["point_id"] not in already_upserted]
    print(f"📄 {len(doc_files)} docs → {total_chunks} chunks | {len(pending)} to embed\n")

    if not pending:
        print("✅ All chunks already indexed.")
        return

    # Embed all pending at once (local model is fast)
    t0 = time.time()
    print(f"⚡ Embedding {len(pending)} chunks locally (BGE-small, CPU)...")
    texts = [c["text"] for c in pending]
    all_vectors = embed_batch(texts)
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s ({len(pending)/elapsed:.0f} chunks/sec)\n")

    # Upsert in batches
    upserted = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch_chunks = pending[i : i + BATCH_SIZE]
        batch_vectors = all_vectors[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=chunk["point_id"],
                vector=vector,
                payload={k: v for k, v in chunk.items() if k != "point_id"},
            )
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        upserted += len(points)

    total_time = time.time() - t0
    print(f"✅ Ingestion complete: {upserted} chunks in Qdrant")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Storage: {QDRANT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true", help="Delete and re-index from scratch")
    args = parser.parse_args()
    main(resume=not args.no_resume)
