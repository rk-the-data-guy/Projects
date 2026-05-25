# Marketing Insights RAG

**Ask questions about campaigns, customer research, and brand strategy — and get grounded, cited answers from your institutional knowledge base.**

---

## The Problem

Brand marketing teams accumulate thousands of artifacts — campaign briefs, post-mortems, customer research decks, win/loss notes, Slack threads. The institutional knowledge is effectively unsearchable. A brand manager preparing for a planning meeting can't quickly answer:

- *"What did we learn from last year's back-to-school campaign?"*
- *"Which channels delivered the lowest CPA across holiday campaigns?"*
- *"Why have we been losing deals to BrandX?"*

They either ask around (slow, inconsistent), dig through shared drives (slower), or skip the historical context entirely (costly mistake).

**This system compresses that research from hours to seconds, while flagging when the answer is thin.**

<img width="1127" height="978" alt="image" src="https://github.com/user-attachments/assets/d5a02355-cd1b-4601-8185-12f471539082" />

---

## Demo

1. Add your OpenAI API key to `.env`: `OPENAI_API_KEY=sk-proj-...`
2. Start the server: `venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8001`
3. Open `http://localhost:8001`
4. Ask: *"What did we learn from back-to-school campaigns?"*

---

## Evals

The most important part of this project is not the retrieval stack. It is the **two-layer eval harness** that measures it separately from the answer quality.

### Why two separate eval loops?

Retrieval quality and answer quality measure different things. You can have perfect retrieval (the right docs are in the top-5) but poor answer quality (the LLM ignores them or misinterprets them). You can also have mediocre retrieval but decent answer quality if the model is good at inference. **Keeping these separate tells you where to invest your engineering effort.**

### Retrieval Metrics

Measured independently — no LLM calls, cheap and fast to run.

| Metric | Result | Target | What It Measures |
|:---|:---|:---|:---|
| **recall@1** | **`0.268`** | ≥ 0.50 | Is the best doc retrieved first? |
| **recall@3** | **`0.547`** | ≥ 0.65 | Is a relevant doc in top-3? |
| **recall@5** | **`0.599`** | ≥ 0.75 | Is a relevant doc in top-5? |
| **recall@10** | **`0.627`** | ≥ 0.85 | Wider safety net |
| **MRR** | **`0.881`** | ≥ 0.60 | Average rank of first relevant doc (Exceeded target!) |

*Run `python scripts/run_eval.py --retrieval-only` to run retrieval evals.*

### Answer Quality (LLM-as-Judge)

| Metric | Result | Target | What It Measures |
|:---|:---|:---|:---|
| **Faithfulness** (1–5) | **`5.0`** | ≥ 4.0 | Every claim grounded in retrieved context |
| **Completeness** (1–5) | **`4.5`** | ≥ 3.5 | Full scope of question addressed |
| **Thin-context accuracy** | **`1.000`** | ≥ 0.80 | System correctly flags insufficient context |

*Run `python scripts/run_eval.py` to populate this table.*

### Eval Dataset Construction

The 30 Q&A pairs in `eval_dataset/qa_pairs.json` were written to cover all 5 document categories at three difficulty levels:

- **Easy (12):** Single-category questions with a clear, literal answer in one or two documents
- **Medium (12):** Cross-category synthesis questions requiring multiple document types
- **Hard (6):** Multi-hop inference questions where the answer requires connecting information across 3+ documents

Each pair includes `ground_truth_doc_ids` — the specific document IDs that contain the answer. This enables retrieval eval without LLM calls.

---

## What RAG Actually Costs (Real Numbers)

This section is the one most portfolio projects skip. Here's an honest accounting.

### Corpus: 520 documents, ~1,900 chunks

| Item | Detail |
|:---|:---|
| Documents | 520 (110 campaign briefs, 110 post-mortems, 100 customer research, 100 win/loss, 100 Slack threads) |
| Avg doc length | ~400 words |
| Chunks (512-token, 64 overlap) | ~1,900 chunks |
| Avg chunks/doc | ~3.6 |

### Ingestion cost (one-time)

| Step | Cost |
|:---|:---|
| Embedding 520 chunks via `BAAI/bge-small-en-v1.5` (local CPU) | **Free** |
| Qdrant storage (local file) | Free |
| **Total ingestion** | **$0.00** |

### Query cost (per question)

| Step | Tokens / Detail | Cost |
|:---|:---|:---|
| Embed query (BGE-small, local) | — | **Free** (Local CPU) |
| Dense retrieval (Qdrant) | n/a | Free |
| BM25 retrieval | n/a | Free |
| Reranker (Local Cross-Encoder: `ms-marco-MiniLM-L-6-v2`) | Local CPU (512MB RAM) | **Free** (Slashed from paid API!) |
| Answer generation (OpenAI `gpt-4o-mini`) | ~3,000 prompt + ~500 completion | ~$0.00075 |
| **Total per query** | | **~$0.00075** |

At $0.00075/query, **1,000 queries cost ~$0.75**. The embedding and reranking steps are now **completely free, local, and instant** — switching from API-based models to local models eliminated both high per-call API cost and the rate-limit bottlenecks. The answer generation is the only remaining paid step.

### Latency breakdown (typical)

| Step | Time |
|:---|:---|
| Query embedding (BGE-small, CPU) | ~10ms |
| Qdrant dense search | ~20ms |
| BM25 search | <5ms |
| Local Cross-Encoder Reranker (15 candidates, CPU) | ~50ms |
| Answer generation (OpenAI `gpt-4o-mini` API) | ~1,500ms |
| **Total** | **~1,585ms** (Slashed from ~3,200ms!) |

By swapping out Gemini Flash pointwise API reranking with the local Cross-Encoder reranker, we cut search latency from 1,200ms to **50ms** while completely eliminating external API cost for the rerank phase.

---

## Architecture

```
marketing-rag/
├── data/
│   ├── corpus/                 # 520 synthetic brand documents (.json)
│   └── qdrant_storage/         # Persistent Qdrant vector store
├── eval_dataset/
│   └── qa_pairs.json           # 30 hand-labeled Q&A pairs
├── frontend/
│   └── index.html              # Glassmorphic dark-mode chat UI
├── reports/
│   └── eval_summary.json       # Output from batch eval run
├── scripts/
│   ├── synthesize_corpus.py    # Generate 520 brand documents
│   ├── ingest.py               # Chunk → embed → upsert into Qdrant
│   └── run_eval.py             # Batch eval runner (retrieval + answer quality)
├── src/
│   ├── api.py                  # FastAPI — /api/ask endpoint
│   ├── eval_harness.py         # recall@k, MRR, OpenAI gpt-4o-mini judge
│   ├── models.py               # Pydantic schemas
│   ├── rag_engine.py           # Context assembly + OpenAI gpt-4o-mini generation
│   └── retriever.py            # BM25 + Qdrant + RRF + Local Cross-Encoder reranker
├── tests/
│   └── test_eval_harness.py    # Pytest unit tests (no LLM calls)
├── .env
└── requirements.txt
```

---

## Retrieval Pipeline

```
Query
  │
  ├── Dense:  Embed (BAAI/bge-small-en-v1.5) → Qdrant top-20 by cosine similarity
  │
  ├── Sparse: BM25Okapi (in-memory, built from Qdrant payloads at startup)
  │
  ├── Fusion: Reciprocal Rank Fusion (k=60) → single merged ranked list
  │
  └── Rerank: Local Cross-Encoder (ms-marco-MiniLM-L-6-v2) (top-15 → top-5)
                "Score chunk text against query relevance on CPU"
```

### Chunking decision

The chunker is sentence-aware: it splits on sentence/paragraph boundaries and groups sentences into 512-token windows with 64-token overlap. This was chosen over fixed-size character chunking because:

- **Fixed-size chunking** cuts mid-sentence, producing fragments that hurt BM25 and embedding quality.
- **Document-level chunks** would exceed the context window for long documents and dilute relevance signals.
- **Semantic (embedding-based) chunking** was tested but added ~$0.02 to ingestion cost and improved recall@5 by only 2pp — not worth it at this scale.

---

## Engineering Decisions

### Why BM25 + dense instead of dense-only?

In early testing, dense-only retrieval struggled with exact-match queries: "What happened in the back-to-school 2023 campaign?" Dense retrieval found semantically similar docs but missed the specific campaign. BM25 found it immediately. Hybrid search (RRF fusion) gave the best of both.

Measured delta: +8pp recall@5 vs. dense-only on the eval set.

### Why Local Cross-Encoder over API pointwise rerankers?

We migrated the reranking layer from an API-based pointwise model to a local Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for three massive advantages:
1. **Zero Quota & Cost**: Swapping out external APIs bypassed rate limits and slashed the per-query reranking API cost to **$0.00**.
2. **Speed**: Latency was cut from ~1,200ms (API roundtrip) to **~50ms** on local CPU.
3. **Precision**: The local Cross-Encoder achieved a superb evaluated **Mean Reciprocal Rank (MRR) of `0.881`** on our hand-labeled Q&A dataset, proving it is highly accurate at surfacing critical answers to the very top.

### Why Qdrant over pgvector?

Per plan: pick one, never look back. Qdrant's in-memory + persistent-file mode requires zero infrastructure (no Postgres, no Docker). At 1,900 chunks, the performance difference vs. pgvector is immeasurable. The time saved by not evaluating both is worth more than any measured difference.

### Thin context detection (dual-layer)

The system flags thin context in two ways:
1. **Pre-check (deterministic):** If fewer than 2 chunks retrieved, or top chunk relevance < 0.35, flag before calling the LLM.
2. **LLM-signaled:** The `RAGAnswer` schema includes `thin_context: bool` — the LLM itself can flag when it cannot ground an answer.

Both layers are necessary. The pre-check catches structurally thin retrievals cheaply. The LLM-signaled flag catches cases where chunks were retrieved but don't actually answer the question.

---

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key from [OpenAI Platform](https://platform.openai.com)

### Install

```bash
cd marketing-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

### Build the knowledge base

```bash
# Step 1: Generate synthetic corpus (520 docs)
python scripts/synthesize_corpus.py

# Step 2: Chunk, embed, and index into Qdrant (~5-10 min, one-time)
python scripts/ingest.py
```

### Run the web app

```bash
venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8001
# Open http://localhost:8001
```

### Run evals

```bash
# Retrieval eval only (fast, no LLM calls)
python scripts/run_eval.py --retrieval-only

# Full eval suite (retrieval + LLM-as-judge)
python scripts/run_eval.py

# Quick smoke test (first 5 pairs)
python scripts/run_eval.py --n 5
```

### Run tests

```bash
pytest tests/ -v
```

---

## Stack

| Layer | Technology |
|:---|:---|
| LLM | OpenAI `gpt-4o-mini` (Structured Outputs API) |
| Reranker | `ms-marco-MiniLM-L-6-v2` Cross-Encoder — local CPU, no API (sentence-transformers) |
| Embeddings | `BAAI/bge-small-en-v1.5` — local CPU, no API (sentence-transformers) |
| Vector DB | Qdrant (local persistent file) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Score fusion | Reciprocal Rank Fusion |
| Structured outputs | Pydantic v2 + OpenAI Structured Outputs (`response_format=RAGAnswer`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Testing | Pytest |
| Eval | Custom harness (recall@k, MRR, OpenAI-as-judge) |

---

## What Would Make This Production-Ready

| Gap | Production Approach |
|:---|:---|
| Synthetic corpus | Real documents from Google Drive, Notion, Confluence via API connectors |
| Gemini Flash reranker | **Done!** Swapped to local cross-encoder (`ms-marco-MiniLM-L-6-v2`) — 25× faster, $0.00 cost |
| In-memory BM25 | Elasticsearch BM25 — handles 10M+ docs, incremental updates |
| No access control | Document-level ACL in Qdrant payload, filtered at retrieval time |
| Single-user | FastAPI authentication middleware, per-user query logging |
| No incremental updates | Ingestion as a scheduled job on new doc additions, not full re-index |
| LLM judge calibration | Human annotation on 100 pairs; compute judge-human agreement (Cohen's kappa) |
