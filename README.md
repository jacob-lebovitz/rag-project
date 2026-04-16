# PDF RAG Pipeline

A minimal, well-structured **Retrieval-Augmented Generation** backend for PDF knowledge bases, built with **FastAPI** and **Mistral AI**. It ingests PDFs, indexes them into a hybrid (dense + sparse) retriever, and answers user questions with grounded, cited responses. A lightweight single-page chat UI is included.

![UI screenshot placeholder](docs/ui.png)

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Pipeline Flow](#pipeline-flow)
4. [Design Considerations](#design-considerations)
5. [Getting Started](#getting-started)
6. [API Reference](#api-reference)
7. [Project Layout](#project-layout)
8. [Libraries Used](#libraries-used)
9. [Extension Ideas](#extension-ideas)

---

## Features

- **FastAPI** backend with OpenAPI docs auto-generated at `/docs`.
- **PDF ingestion** with robust text extraction (pdfplumber + pypdf fallback) and token-aware recursive chunking with overlap.
- **Hybrid retrieval**: dense (Mistral embeddings + FAISS) combined with sparse (BM25) via **Reciprocal Rank Fusion**.
- **Intent detection** that skips retrieval for small talk (greetings, acknowledgements), saving latency and cost.
- **Query transformation** — LLM-driven rewriting into self-contained search queries.
- **Grounded answers** with inline `[n]` citations pointing back to source passages (filename + page).
- **Simple chat UI** bundled as static assets — no build step, no framework.
- **Persistent index**: FAISS index + chunk metadata are written to disk and reloaded on startup.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            UI (static HTML/JS)                      │
│  • PDF upload (drag & drop)    • Chat with citations                │
└──────────────────────┬──────────────────────────────────────────────┘
                       │  HTTP
┌──────────────────────▼──────────────────────────────────────────────┐
│                            FastAPI                                  │
│                                                                     │
│   POST /ingest  ──► ┌──────────────────────────────────────┐        │
│                     │ pdfplumber → clean → recursive chunk │        │
│                     └──────────────┬───────────────────────┘        │
│                                    ▼                                │
│                     ┌──────────────────────────────┐                │
│                     │ Mistral embed (batched)      │                │
│                     └──────────────┬───────────────┘                │
│                                    ▼                                │
│                     ┌──────────────────────────────┐                │
│                     │  FAISS (IndexFlatIP, cosine) │                │
│                     │  BM25Okapi (keyword index)   │                │
│                     └──────────────────────────────┘                │
│                                                                     │
│   POST /query   ──► intent detect ──► rewrite ──► hybrid search     │
│                                    ──► RRF fuse  ──► LLM answer     │
│                                    ──► grounded response w/ sources │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

### Ingestion

```
PDF file ─► pdfplumber ─► per-page text ─► clean (dehyphenate, collapse ws)
           (fallback: pypdf)                        │
                                                    ▼
                          recursive splitter (paragraph → sentence → word)
                                                    │
                                            token-aware packing
                                                    │
                                        overlap window (64 tokens)
                                                    │
                                                    ▼
                                   Chunk {doc_id, filename, page, text}
                                                    │
                                  ┌─────────────────┴──────────────────┐
                                  ▼                                    ▼
                         Mistral embeddings                    BM25 tokenizer
                                  │                                    │
                                  ▼                                    ▼
                            FAISS index                          BM25Okapi
```

### Query

```
user question
      │
      ▼
┌─────────────────┐    small_talk    ┌────────────────────┐
│ intent detect   │ ───────────────► │ LLM chit-chat only │ ──► response
│ (rules + LLM)   │                  └────────────────────┘
└────────┬────────┘
         │ knowledge_base
         ▼
┌─────────────────┐
│ query rewrite   │  ← Mistral chat, self-contained search query
└────────┬────────┘
         ▼
┌──────────────────────────────────────┐
│     hybrid retrieval                 │
│                                      │
│   ┌──────────────┐   ┌────────────┐  │
│   │ dense (FAISS)│   │ BM25       │  │
│   └──────┬───────┘   └────────┬───┘  │
│          ▼                    ▼      │
│       Reciprocal Rank Fusion         │
│          ▼                           │
│       top-k final chunks             │
└────────┬─────────────────────────────┘
         ▼
┌─────────────────────────────────┐
│   grounded LLM answer           │
│   (prompt with cited passages)  │
└─────────────────────────────────┘
         │
         ▼
answer + [1][2] citations + source chunks
```

---

## Design Considerations

### PDF text extraction

- **Primary extractor:** [`pdfplumber`](https://github.com/jsvine/pdfplumber) — preserves layout reasonably well for text-based PDFs, handles columns and whitespace better than naive extractors.
- **Fallback:** [`pypdf`](https://github.com/py-pdf/pypdf) — runs automatically if pdfplumber fails or produces only empty pages.
- **Cleaning:** normalize line endings, collapse repeated whitespace, re-join hyphenated line breaks (`exam-\nple` → `example`), preserve paragraph breaks.
- **Scanned PDFs (image-only):** out of scope. Would require OCR (e.g. `pytesseract` or a hosted OCR API); the pipeline is structured so this would slot into `pdf_parser.py` transparently.
- **Tables / figures:** not parsed as structured data. Text inside tables is captured linearly. A future extension could use `pdfplumber.extract_tables()` and emit a separate chunk-kind flag.

### Chunking

- **Token-aware** (via `tiktoken.cl100k_base`) rather than character-based — this keeps chunks within the embedding model's token budget predictably.
- **Recursive hierarchy:** paragraphs → sentences → words. Keeps semantic units intact.
- **Overlap = 64 tokens** between adjacent chunks so answers that straddle a boundary can still be retrieved.
- **Chunks never cross page boundaries** — this gives every chunk a single, unambiguous citation target.
- **Default size: 512 tokens.** Small enough to stay under Mistral embed limits comfortably, large enough to carry meaningful context.

### Intent detection — *should we search the KB?*

Two-stage cascade:

1. **Rule-based** (fast, free):
   - Pure greetings / acknowledgements → `small_talk`.
   - Very short non-questions → `small_talk`.
   - Question mark OR leading Wh-word / command verb → `knowledge_base`.
2. **LLM fallback** for ambiguous cases — a short classification prompt. Defaults to `knowledge_base` if the call fails (false positives just waste a retrieval; false negatives skip it entirely).

Why cascade? Most real chit-chat is trivially classifiable, so we save both latency and API spend.

### Query transformation

- **Light normalization** always (strip, collapse whitespace).
- **LLM rewrite** for retrieval-time — asks Mistral to produce a concise, self-contained search query that resolves pronouns and removes pleasantries. Conservative temperature (0.0) to avoid inventing topics.
- Not implemented (left as extensions): HyDE, multi-query expansion, synonym expansion — these would further improve recall on harder queries.

### Semantic + keyword fusion

Embeddings are strong at semantic similarity but miss rare tokens — exact identifiers, acronyms, version numbers, code. BM25 is the mirror image. Running both and fusing gives the best of each.

**Fusion algorithm: Reciprocal Rank Fusion (RRF).**

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

Summed across all retrievers that contain `d`. `k = 60` per Cormack et al. (2009). RRF requires no score calibration, which matters because BM25 scores and cosine similarities live on completely different scales.

Alternatives considered:
- **Weighted score combination:** requires normalization per retriever and a tunable weight — brittle.
- **Cross-encoder re-ranking:** excellent quality, but adds a heavy model dependency and latency. The code is structured so a cross-encoder re-rank pass could drop in after RRF.

### Post-processing / re-ranking

- **RRF itself** functions as the primary re-rank.
- The retriever truncates to `TOP_K_FINAL` (default 5) before prompting the LLM — avoiding context bloat and irrelevant passages diluting the answer.
- The answer prompt instructs the LLM to cite `[n]` and say "I don't have that information" when the passages don't cover the question — mitigating hallucinations.

### Generation

- **Model:** `mistral-small-latest` by default. Overridable via `MISTRAL_CHAT_MODEL`.
- **Temperature 0.1** for answers (factual), 0.6 for small talk (friendlier).
- **System prompt** enforces grounding + citation discipline.
- **Max tokens 1024** per answer — enough for detailed responses without runaway generation.

---

## Getting Started

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# The provided Mistral API key is already set in .env.example.
# Edit .env to override the key or tune parameters.
```

### 3. Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

| URL | What |
| --- | --- |
| `http://localhost:8000/` | Chat UI |
| `http://localhost:8000/docs` | Interactive OpenAPI docs |
| `http://localhost:8000/health` | Health check + index stats |

### 4. Usage

**Upload PDFs** via the UI (drag-and-drop or file picker), or via curl:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf"
```

**Ask a question** via the UI, or via curl:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the paper say about retrieval fusion?"}'
```

**Reset the index** (testing only):

```bash
curl -X DELETE http://localhost:8000/ingest
```

---

## API Reference

### `POST /ingest`

Upload one or more PDFs for indexing.

- **Body (multipart/form-data):** `files=@file1.pdf&files=@file2.pdf` (repeatable field)
- **Response 201:**
  ```json
  {
    "status": "ok",
    "files": [
      {"filename": "paper.pdf", "doc_id": "a1b2c3d4e5f6", "num_pages": 12, "num_chunks": 38}
    ],
    "total_chunks": 38
  }
  ```

### `POST /query`

Answer a user question.

- **Body:**
  ```json
  {"question": "your question here", "top_k": 5, "include_sources": true}
  ```
  `top_k` is optional (defaults to `TOP_K_FINAL`).
- **Response:**
  ```json
  {
    "answer": "... grounded answer with [1][2] citations ...",
    "intent": "knowledge_base",
    "transformed_query": "rewritten search query",
    "sources": [
      {"doc_id": "...", "filename": "paper.pdf", "page": 3, "chunk_id": 12, "score": 0.812, "text": "..."}
    ]
  }
  ```

### `DELETE /ingest`

Wipe the index and uploaded files. Intended for local testing.

### `GET /health`

Returns index stats and the configured models.

---

## Project Layout

```
rag-pipeline/
├── app/
│   ├── main.py                 FastAPI app, CORS, UI mount, health
│   ├── config.py               Pydantic settings (env-driven)
│   ├── models.py               Pydantic schemas
│   ├── ingestion/
│   │   ├── pdf_parser.py       pdfplumber + pypdf fallback
│   │   └── chunker.py          recursive token-aware chunking
│   ├── retrieval/
│   │   ├── vector_store.py     FAISS IndexFlatIP + persistence
│   │   ├── keyword_store.py    BM25Okapi
│   │   └── hybrid.py           RRF fusion, retriever singleton
│   ├── query/
│   │   ├── intent.py           rule + LLM intent cascade
│   │   └── transform.py        LLM query rewrite
│   ├── generation/
│   │   ├── llm.py              Mistral SDK wrapper (embed + chat)
│   │   └── prompts.py          RAG + small-talk prompt templates
│   └── routes/
│       ├── ingest.py           POST/DELETE /ingest
│       └── query.py            POST /query
├── ui/
│   ├── index.html              single-page chat UI
│   ├── styles.css
│   └── app.js
├── data/                       persisted FAISS index + uploads (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Libraries Used

| Purpose | Library | Why |
| --- | --- | --- |
| Web framework | [FastAPI](https://fastapi.tiangolo.com/) | Async, auto OpenAPI docs, Pydantic validation |
| ASGI server | [Uvicorn](https://www.uvicorn.org/) | Production-ready ASGI |
| LLM + embeddings | [mistralai](https://github.com/mistralai/client-python) | Official Mistral Python SDK |
| PDF extraction | [pdfplumber](https://github.com/jsvine/pdfplumber) + [pypdf](https://github.com/py-pdf/pypdf) | Layout-aware primary + resilient fallback |
| Tokenization | [tiktoken](https://github.com/openai/tiktoken) | Fast, accurate token counting for chunk sizing |
| Vector search | [faiss-cpu](https://github.com/facebookresearch/faiss) | Industry-standard ANN / exact similarity |
| Keyword search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Simple, dependency-free BM25 |
| Async file I/O | [aiofiles](https://github.com/Tinche/aiofiles) | Non-blocking upload writes |
| Config | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Typed env-based config |

All free / open-source. No paid dependencies besides the Mistral API itself.

---

## Extension Ideas

These are deliberate omissions that would be the next natural improvements:

- **Streaming responses** via SSE for the answer, giving faster perceived latency in the UI.
- **Chat history** in the request — carry the last N turns so follow-up questions work ("tell me more about that").
- **Cross-encoder re-ranker** (e.g. `bge-reranker-v2-m3`) after RRF for higher-precision top-k.
- **HyDE** (Hypothetical Document Embeddings) for hard queries — embed an LLM-hallucinated answer and search with that.
- **Scanned-PDF OCR** path — plug `pytesseract` into `pdf_parser.py` when pdfplumber returns empty pages.
- **Delete by doc_id** in addition to full-index reset.
- **Persistent chat sessions** in a lightweight store (SQLite).
- **Observability:** request IDs, structured logging, metrics (Prometheus).
- **Evaluation harness:** a tiny set of (question, gold-passage, gold-answer) triples with recall@k and answer-faithfulness metrics.

---

## License

MIT. See `LICENSE` (add your preferred license if distributing).
