"""HTTP API and web UI for the RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import DEFAULT_CORPUS, DEFAULT_INDEX, RAGPipeline, build_index

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchRequest(AskRequest):
    pass


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    corpus = Path(os.environ.get("RAG_CORPUS", DEFAULT_CORPUS))
    index_path = Path(os.environ.get("RAG_INDEX", DEFAULT_INDEX))
    state = {"pipeline": pipeline or RAGPipeline.from_path(index_path, corpus)}
    app = FastAPI(title="Maafi RAG")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(state["pipeline"].index)}

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict:
        hits = state["pipeline"].retrieve(req.question, req.top_k)
        return {"hits": [{**asdict(h.chunk), "score": round(h.score, 4)} for h in hits]}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        try:
            return asdict(state["pipeline"].ask(req.question, req.top_k))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except anthropic.AuthenticationError as e:
            raise HTTPException(500, "Claude API credentials are missing or invalid") from e
        except anthropic.RateLimitError as e:
            raise HTTPException(429, "Claude API rate limit hit; retry shortly") from e
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"Claude API error: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise HTTPException(502, "Could not reach the Claude API") from e
        except TypeError as e:  # raised by the SDK when no credentials resolve
            if "authentication" not in str(e):
                raise
            raise HTTPException(500, "Claude API credentials are not configured") from e

    @app.post("/api/reindex")
    def reindex() -> dict:
        old = state["pipeline"]
        index = build_index(corpus)
        index.save(index_path)
        state["pipeline"] = RAGPipeline(index, old._generator, old.top_k)
        return {"status": "ok", "chunks": len(index)}

    return app
