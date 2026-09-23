"""Maafi RAG: retrieval-augmented generation over a local document folder."""

from .chunker import Chunk, chunk_document
from .loader import Document, load_documents
from .pipeline import Answer, RAGPipeline
from .retriever import BM25Index, Hit

__all__ = [
    "Answer",
    "BM25Index",
    "Chunk",
    "Document",
    "Hit",
    "RAGPipeline",
    "chunk_document",
    "load_documents",
]
