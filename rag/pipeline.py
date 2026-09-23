"""End-to-end RAG pipeline: ingest -> index -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_document
from .generator import Citation, ClaudeGenerator, Generator
from .loader import load_documents
from .retriever import BM25Index, Hit

DEFAULT_CORPUS = Path("data/docs")
DEFAULT_INDEX = Path("data/index.json")


@dataclass
class Source:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False


def build_index(corpus: str | Path, max_words: int = 200, overlap: int = 40) -> BM25Index:
    chunks = [
        c for doc in load_documents(corpus) for c in chunk_document(doc, max_words, overlap)
    ]
    return BM25Index(chunks)


class RAGPipeline:
    def __init__(self, index: BM25Index, generator: Generator | None = None, top_k: int = 5):
        self.index = index
        self._generator = generator
        self.top_k = top_k

    @property
    def generator(self) -> Generator:
        # Created lazily so retrieval-only use needs no API credentials.
        if self._generator is None:
            self._generator = ClaudeGenerator()
        return self._generator

    @classmethod
    def from_path(
        cls,
        index_path: str | Path = DEFAULT_INDEX,
        corpus: str | Path = DEFAULT_CORPUS,
        **kwargs,
    ) -> "RAGPipeline":
        index_path = Path(index_path)
        if index_path.exists():
            index = BM25Index.load(index_path)
        else:
            index = build_index(corpus)
            index.save(index_path)
        return cls(index, **kwargs)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.index.search(question, k or self.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        hits = self.retrieve(question, k)
        gen = self.generator.generate(question, hits)
        return Answer(
            question=question,
            answer=gen.text,
            sources=[
                Source(h.chunk.chunk_id, h.chunk.doc_id, h.chunk.title, h.chunk.text, round(h.score, 4))
                for h in hits
            ],
            citations=gen.citations,
            refused=gen.refused,
        )
