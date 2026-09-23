"""Okapi BM25 lexical retriever with JSON persistence. No external dependencies."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunker import Chunk

_TOKEN = re.compile(r"[\w']+", re.U)
STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that the this to was "
    "were will with you your what which who how when where why do does did can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.title + " " + c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        if not terms:
            return []
        hits = []
        for i, tf in enumerate(self._tf):
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1))
            score = sum(
                self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if tf[t]
            )
            if score > 0:
                hits.append(Hit(self.chunks[i], score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def save(self, path: str | Path) -> None:
        data = {"k1": self.k1, "b": self.b, "chunks": [asdict(c) for c in self.chunks]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in data["chunks"]], k1=data["k1"], b=data["b"])
