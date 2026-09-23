# About the Maafi RAG system

Maafi RAG is a retrieval-augmented generation system. It answers questions using documents stored in the data/docs folder instead of relying only on what the language model already knows.

## How a question is answered

First, every document in data/docs is split into overlapping chunks of about 200 words. The chunks are indexed with the BM25 ranking function and the index is saved to data/index.json.

When a question arrives, the retriever scores every chunk with BM25 and keeps the top five. Those chunks are sent to Claude as document blocks with citations enabled. Claude answers only from those passages and cites the passages it used. If the passages do not contain the answer, Claude says so instead of guessing.

## Adding knowledge

To add knowledge, drop .txt, .md or .html files into data/docs and run the ingest command: python -m rag ingest. The server can also rebuild the index without a restart through POST /api/reindex.
