"""Command line: ingest a corpus, search it, ask questions, or run the server."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import anthropic

from .pipeline import DEFAULT_CORPUS, DEFAULT_INDEX, RAGPipeline, build_index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description="Maafi retrieval-augmented generation")
    p.add_argument("--index", default=str(DEFAULT_INDEX), help="index file (default: %(default)s)")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="document folder (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="(re)build the index from the corpus")
    ing.add_argument("--chunk-words", type=int, default=200)
    ing.add_argument("--overlap", type=int, default=40)

    s = sub.add_parser("search", help="retrieve chunks without calling the model")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5)

    a = sub.add_parser("ask", help="retrieve and answer with Claude")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--json", action="store_true", help="print the full answer object")

    sv = sub.add_parser("serve", help="run the HTTP API and web UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        index = build_index(args.corpus, args.chunk_words, args.overlap)
        index.save(args.index)
        print(f"indexed {len(index)} chunks from {args.corpus} -> {args.index}")
        return 0

    if args.cmd == "serve":
        import os

        import uvicorn

        from .server import create_app

        os.environ["RAG_INDEX"] = args.index
        os.environ["RAG_CORPUS"] = args.corpus
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    pipeline = RAGPipeline.from_path(args.index, args.corpus)

    if args.cmd == "search":
        for h in pipeline.retrieve(args.question, args.k):
            print(f"{h.score:7.3f}  {h.chunk.chunk_id}\n         {h.chunk.text[:160]}")
        return 0

    try:
        ans = pipeline.ask(args.question, args.k)
    except TypeError as e:  # raised by the SDK when no credentials resolve
        if "authentication" not in str(e):
            raise
        print("error: set ANTHROPIC_API_KEY (or run `ant auth login`) to ask questions", file=sys.stderr)
        return 2
    except anthropic.APIError as e:
        print(f"error: Claude API request failed: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(asdict(ans), indent=2, ensure_ascii=False))
        return 0
    print(ans.answer)
    if ans.sources:
        print("\nSources:")
        for i, s in enumerate(ans.sources, 1):
            print(f"  [{i}] {s.chunk_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
