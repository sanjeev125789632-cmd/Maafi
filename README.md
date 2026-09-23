# Maafi RAG

Retrieval-augmented generation over a local document folder. Questions are answered by Claude using only passages retrieved from `data/docs`, with citations.

## Pipeline

```
data/docs/*.{txt,md,html}
   │  rag/loader.py     load + strip HTML
   ▼
   │  rag/chunker.py    ~200-word paragraph-aware chunks, 40-word overlap
   ▼
   │  rag/retriever.py  BM25 index  →  data/index.json
   ▼
question → top-k chunks → rag/generator.py (Claude, Citations API) → answer + sources
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # only needed for `ask` / `/api/ask`
```

## Usage

```bash
python -m rag ingest                     # build data/index.json from data/docs
python -m rag search "supported file types"   # retrieval only, no API call
python -m rag ask "Which model answers questions?"
python -m rag serve                      # web UI at http://127.0.0.1:8000
```

HTTP API:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | – | chunk count |
| POST | `/api/search` | `{"question", "top_k"?}` | ranked chunks |
| POST | `/api/ask` | `{"question", "top_k"?}` | answer, sources, citations |
| POST | `/api/reindex` | – | rebuilds the index from the corpus |

## Configuration

| Env var | Default |
|---|---|
| `RAG_MODEL` | `claude-opus-5` |
| `RAG_EFFORT` | `medium` |
| `RAG_CORPUS` | `data/docs` |
| `RAG_INDEX` | `data/index.json` |

Requests use adaptive thinking and the server-side refusal fallback (`fallbacks: "default"`); a `refusal` stop reason is reported instead of an empty answer.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
