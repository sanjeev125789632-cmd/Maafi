"""Split documents into overlapping, paragraph-aware chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_document(doc: Document, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    # Pack whole paragraphs until the budget is hit; hard-split oversized paragraphs.
    words: list[str] = []
    pieces: list[list[str]] = []
    for para in (p.split() for p in doc.text.split("\n\n")):
        if not para:
            continue
        if words and len(words) + len(para) > max_words:
            pieces.append(words)
            words = words[-overlap:] if overlap else []
        words.extend(para)
        while len(words) > max_words:
            pieces.append(words[:max_words])
            words = words[max_words - overlap:]
    if words and (not pieces or words != pieces[-1][-len(words):]):
        pieces.append(words)

    return [
        Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, title=doc.title, text=" ".join(p))
        for i, p in enumerate(pieces)
    ]
