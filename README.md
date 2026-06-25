# CiteQuest-RAG

**Academic Search + Citation-grounded RAG + Lightweight Research Agent**

A local-first academic paper retrieval and question-answering system. Search papers by keyword or semantic meaning, ask questions grounded in real paper evidence with verifiable citations, and generate structured research summaries.

> ⚠️ **Status: Early Development (v0.1)** — Search MVP phase. See [PLAN.md](docs/PLAN.md) for roadmap.

## What Makes This Different from a Chatbot Wrapper

- Every answer includes **citations** that map to specific chunks in retrieved papers — you can verify where each claim comes from.
- **Hybrid retrieval** (lexical BM25 + dense vector) rather than naive keyword search.
- Built-in **retrieval evaluation** (Recall@k, MRR, nDCG) and **RAG evaluation** (citation precision, faithfulness).
- Not just a prompt over a vector DB — ingest pipeline, metadata store, multi-mode retrieval, and citation verification are all explicit and inspectable.

## Architecture

```text
User Query
    ↓
FastAPI Router
    ↓
Lexical Retriever ─────┐
                       ├── Hybrid Merger → Context Builder → LLM Answer
Vector Retriever  ─────┘                                      ↓
                                                        Citation Verifier
                                                               ↓
                                                        JSON Response
```

## Tech Stack

| Layer | Choice |
|---|---|
| API Framework | FastAPI + Pydantic v2 |
| Metadata Store | SQLite |
| Lexical Search | SQLite FTS5 + BM25 |
| Vector Search | FAISS (local) |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| LLM Provider | OpenAI-compatible API |
| Container | Docker Compose (v1.0) |

## Quickstart (coming soon)

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Prepare demo data and indexes
python scripts/sample_corpus.py --size 10000 --output data/raw/openalex_sample.jsonl
python scripts/build_metadata_db.py --input data/raw/openalex_sample.jsonl
python scripts/build_fts.py
python scripts/build_faiss.py

# Run API
uvicorn app.main:app --reload

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"retrieval augmented generation evaluation","top_k":5,"mode":"hybrid"}'

# Ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How is RAG usually evaluated?","top_k":8}'
```

## Documentation

- [SPEC.md](docs/SPEC.md) — Full architecture, scope, and design decisions
- [PLAN.md](docs/PLAN.md) — Week-by-week development roadmap and checklists
- [API_CONTRACT.md](docs/API_CONTRACT.md) — API request/response specifications
- [DATA_PLAN.md](docs/DATA_PLAN.md) — Data sourcing and processing strategy
- [EVAL_PLAN.md](docs/EVAL_PLAN.md) — Evaluation design and metrics

## Roadmap

| Version | Goal |
|---|---|
| v0.1 | SQLite FTS5 lexical search + FastAPI `/search` |
| v0.2 | FAISS vector search + hybrid retrieval + retrieval eval |
| v0.3 | Citation-grounded RAG + `/ask` + citation verification |
| v0.4 | Research agent workflows (compare, related work) |
| v1.0 | Docker, README, evaluation report, GitHub release |

## License

MIT
