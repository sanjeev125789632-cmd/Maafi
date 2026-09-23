"""Load plain-text documents (.txt, .md, .html) from a directory tree."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".html", ".htm"}

_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Document:
    doc_id: str  # path relative to the corpus root
    title: str
    text: str


def html_to_text(raw: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = _TAG.sub("\n", text)
    text = html.unescape(text)
    lines = (_WS.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _title(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return path.stem


def load_file(path: Path, root: Path) -> Document | None:
    if path.suffix.lower() not in SUPPORTED:
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = html_to_text(raw) if path.suffix.lower() in {".html", ".htm"} else raw
    text = text.strip()
    if not text:
        return None
    return Document(doc_id=path.relative_to(root).as_posix(), title=_title(path, text), text=text)


def load_documents(root: str | Path) -> list[Document]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            doc = load_file(path, root)
            if doc:
                docs.append(doc)
    return docs
