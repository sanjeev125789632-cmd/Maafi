from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import BM25Index, RAGPipeline, chunk_document, load_documents
from rag.generator import Citation, ClaudeGenerator, Generation
from rag.loader import Document, html_to_text
from rag.pipeline import build_index
from rag.server import create_app

CORPUS = Path(__file__).resolve().parent.parent / "data" / "docs"


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, hits):
        self.calls.append((question, hits))
        return Generation(
            text=f"answer from {len(hits)} chunks",
            citations=[Citation(1, hits[0].chunk.title, "x")] if hits else [],
        )


def test_html_to_text_strips_markup():
    raw = "<html><style>p{}</style><script>alert(1)</script><h1>Hi &amp; bye</h1><p>body</p></html>"
    assert html_to_text(raw) == "Hi & bye\nbody"


def test_load_documents_reads_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# Title A\n\ntext")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.html").write_text("<p>Hello</p>")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["a.md", "sub/c.html"]
    assert docs[0].title == "Title A"


def test_chunker_overlaps_and_covers_all_words():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_document(Document("d", "t", text), max_words=100, overlap=20)
    assert all(len(c.text.split()) <= 100 for c in chunks)
    seen = {w for c in chunks for w in c.text.split()}
    assert seen == set(text.split())
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "t", "x"), max_words=10, overlap=10)


def test_bm25_ranks_relevant_chunk_first_and_roundtrips(tmp_path):
    index = build_index(CORPUS)
    hits = index.search("which file types are supported html markdown", k=3)
    assert hits and "faq.md" in hits[0].chunk.chunk_id
    assert index.search("zzzz qqqq") == []

    path = tmp_path / "idx.json"
    index.save(path)
    loaded = BM25Index.load(path)
    assert len(loaded) == len(index)
    assert [h.chunk.chunk_id for h in loaded.search("BM25 index chunks")] == [
        h.chunk.chunk_id for h in index.search("BM25 index chunks")
    ]


def test_pipeline_ask_uses_retrieved_chunks():
    gen = FakeGenerator()
    pipe = RAGPipeline(build_index(CORPUS), generator=gen, top_k=2)
    ans = pipe.ask("How do I run the web interface?")
    assert len(ans.sources) == 2
    assert ans.answer == "answer from 2 chunks"
    assert gen.calls[0][0] == "How do I run the web interface?"
    with pytest.raises(ValueError):
        pipe.ask("   ")


def test_claude_generator_request_and_citation_parsing():
    class Block:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            cit = Block(document_index=0, document_title="FAQ", cited_text="Run python -m rag serve")
            return Block(
                stop_reason="end_turn",
                content=[
                    Block(type="thinking", thinking=""),
                    Block(type="text", text="Use the serve command.", citations=[cit]),
                ],
            )

    client = Block(beta=Block(messages=FakeMessages()))
    hits = build_index(CORPUS).search("web interface serve", k=2)
    out = ClaudeGenerator(model="claude-opus-5", client=client).generate("How?", hits)

    assert out.text == "Use the serve command. [1]"
    assert out.citations[0].title == "FAQ"
    content = captured["messages"][0]["content"]
    assert [c["type"] for c in content] == ["document"] * len(hits) + ["text"]
    assert all(c["citations"] == {"enabled": True} for c in content[:-1])
    assert captured["fallbacks"] == "default"


def test_claude_generator_handles_refusal_and_no_hits():
    class R:
        stop_reason = "refusal"
        content = []

    class M:
        def create(self, **kw):
            return R()

    class C:
        class beta:
            messages = M()

    hits = build_index(CORPUS).search("model", k=1)
    assert ClaudeGenerator(client=C()).generate("q", hits).refused
    assert "could not find" in ClaudeGenerator(client=C()).generate("q", []).text


def test_http_api(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CORPUS", str(CORPUS))
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "index.json"))
    pipe = RAGPipeline(build_index(CORPUS), generator=FakeGenerator())
    client = TestClient(create_app(pipe))

    assert client.get("/api/health").json()["chunks"] == len(pipe.index)
    assert "Maafi RAG" in client.get("/").text
    assert client.post("/api/search", json={"question": "API key"}).json()["hits"]
    body = client.post("/api/ask", json={"question": "Do I need an API key?"}).json()
    assert body["answer"].startswith("answer from")
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/reindex").json()["status"] == "ok"
    assert (tmp_path / "index.json").exists()
