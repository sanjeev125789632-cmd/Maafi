"""Answer generation with Claude, grounded in retrieved chunks via the Citations API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import anthropic

from .retriever import Hit

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You answer questions using only the documents provided in the user turn. "
    "If the documents do not contain the answer, say you could not find it in the knowledge base "
    "instead of guessing. Keep answers short and direct."
)


@dataclass
class Citation:
    source: int  # 1-based index into Answer.sources
    title: str
    cited_text: str


@dataclass
class Generation:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False


class Generator(Protocol):
    def generate(self, question: str, hits: list[Hit]) -> Generation: ...


class ClaudeGenerator:
    def __init__(
        self,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 16000,
        effort: str | None = None,
    ):
        self.model = model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.effort = effort or os.environ.get("RAG_EFFORT", "medium")

    @staticmethod
    def build_content(question: str, hits: list[Hit]) -> list[dict]:
        docs = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": h.chunk.text},
                "title": f"{h.chunk.title} ({h.chunk.chunk_id})",
                "citations": {"enabled": True},
            }
            for h in hits
        ]
        return docs + [{"type": "text", "text": question}]

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        if not hits:
            return Generation("I could not find anything relevant in the knowledge base.")

        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": self.build_content(question, hits)}],
        )

        if response.stop_reason == "refusal":
            return Generation("The model declined to answer this question.", refused=True)

        parts: list[str] = []
        citations: list[Citation] = []
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            for c in getattr(block, "citations", None) or []:
                idx = getattr(c, "document_index", None)
                if idx is None:
                    continue
                citations.append(
                    Citation(source=idx + 1, title=c.document_title or "", cited_text=c.cited_text)
                )
                parts.append(f" [{idx + 1}]")
        return Generation(text="".join(parts).strip(), citations=citations)
