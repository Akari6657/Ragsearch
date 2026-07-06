# CiteQuest-RAG

Academic Search + Citation-grounded RAG + AI Overview

Local-first academic paper search and question answering with SQLite FTS5,
FAISS vector retrieval, and citation-aware RAG.

## Current Status

- Lexical search is ready with SQLite FTS5 over local chunks.
- Vector and hybrid search require a built FAISS index (`data/indexes/faiss/index.faiss` + `id_map.json`).
- `/search` returns `503 INDEX_NOT_READY` for `mode=vector` or `mode=hybrid` until FAISS is built.
- Runtime index paths can be overridden with `CITEQUEST_DB_PATH` and `CITEQUEST_FAISS_DIR`.
- `/ask` and AI Overview are implemented; without `LLM_API_KEY`, the app uses a mock LLM provider for local development.
- Local generated data and indexes are intentionally not committed.

---

## Quickstart

```bash
# 1. Setup
cd Citequest
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# 2. Configure LLM (optional — works without for search-only)
cp .env.example .env
# Edit .env with your DeepSeek API key

# 3. Create a small demo corpus from an existing local peS2o full-text corpus
python scripts/sample_corpus.py \
  --input data/raw/peS2o_cs_fulltext_50000.jsonl \
  --output data/raw/demo_peS2o_1000.jsonl \
  --size 1000 \
  --seed 42

# 4. Build demo indexes
python scripts/build_metadata_db.py \
  --input data/raw/demo_peS2o_1000.jsonl \
  --db data/indexes/demo/metadata.sqlite
python scripts/build_fts.py --db data/indexes/demo/metadata.sqlite
python scripts/build_faiss.py \
  --db data/indexes/demo/metadata.sqlite \
  --output-dir data/indexes/demo/faiss

# 5. Run against the demo indexes
CITEQUEST_DB_PATH=data/indexes/demo/metadata.sqlite \
CITEQUEST_FAISS_DIR=data/indexes/demo/faiss \
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 6. Open http://127.0.0.1:8000
```

If you do not already have a local peS2o corpus, use
`scripts/download_fulltext.py` or `scripts/download_arxiv.py` to create one
first. The larger 50k-paper peS2o corpus is treated as a benchmark dataset, not
the default quickstart path.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Search frontend |
| `/health` | GET | Service status + index availability |
| `/search` | POST | Paper search + optional AI Overview |
| `/ask` | POST | RAG question answering |
| `/docs` | GET | OpenAPI docs (Swagger) |

### POST /search

```json
{
  "query": "attention mechanism in transformers",
  "top_k": 10,
  "mode": "hybrid",
  "alpha": 0.5,
  "include_overview": true
}
```

Modes: `lexical` (BM25), `vector` (FAISS), `hybrid` (both, alpha-weighted).

When `include_overview: true`, the router decides whether to generate an AI Overview:
- Questions ("what is...", "为什么...") → triggers RAG
- Keywords ("GAN image generation") → skips RAG
- Ambiguous queries → LLM decides

### POST /ask

```json
{
  "question": "What are multimodal large language models?",
  "top_k": 8,
  "retrieval_mode": "hybrid",
  "alpha": 0.3
}
```

Returns answer with citation markers `[1]`, `[2]`, plus citation metadata and validity check.

---

## Architecture

```
User Query
  → Router (rules + LLM) → should RAG?
  → Rewriter (CJK → English keywords)
  → Lexical (FTS5 BM25) + Vector (FAISS BGE)
  → Hybrid merge
  → [RAG] Context builder → DeepSeek → Citation verify
  → Response (results + ai_overview)
```

### Tech Stack

| Layer | Choice |
|---|---|
| API | FastAPI + Pydantic v2 |
| Lexical search | SQLite FTS5 + BM25 |
| Vector search | FAISS IVF + BGE-M3 embeddings (1024d) |
| LLM | DeepSeek V4 Flash (OpenAI-compatible) |
| Frontend | Single-page HTML (no framework) |
| Data | arXiv metadata sample; optional peS2o full-text experiment |

### Key Design Decisions

- No stemming or stop-word removal (BM25 IDF handles it naturally)
- Chunk text = Title + Abstract, plus body chunks when full text is available
- OR search + phrase boost (user can use `"..."` for exact phrase matching)
- FAISS is local and file-based; generated indexes are not committed
- AI Overview on demand, not every search (router decides)
- Citations validated but answer never rejected (frontend controls display)

---

## Project Structure

```
app/
  core/schemas.py          Pydantic models
  ingestion/               loader → normalize → chunk
  retrieval/               lexical (BM25), vector (FAISS), hybrid
  rag/                     llm_provider, prompt, context_builder,
                           citation, answer, router, rewriter
  api/                     routes_search, routes_ask
  eval/                    retrieval_eval, rag_eval
  main.py                  FastAPI app

scripts/                   download_arxiv, build_metadata_db,
                           build_fts, build_faiss, sample_corpus

data/                      raw/ (JSONL), indexes/ (SQLite, FAISS)
frontend/index.html        Search UI
tests/                     110+ tests
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Roadmap

| Version | Feature |
|---|---|
| v0.1 | FTS5 BM25 lexical search |
| v0.2 | FAISS vector search + hybrid retrieval |
| v0.3 | Citation-grounded RAG + /ask |
| v0.4 | Router + Rewriter + AI Overview |
| v0.5 | Frontend search page |
| v0.6 | Paper detail page |
| v1.0 | Docker + evaluation report |

---

## License

MIT
