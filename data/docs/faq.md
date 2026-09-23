# Frequently asked questions

## Which model answers the questions?

By default the generator uses the claude-opus-5 model. Set the RAG_MODEL environment variable to use a different Claude model, and RAG_EFFORT to change the effort level (default medium).

## Do I need an API key?

Answering questions calls the Claude API, so ANTHROPIC_API_KEY must be set. Ingesting documents and the search command work offline without any key.

## Which file types are supported?

Plain text (.txt), Markdown (.md, .markdown) and HTML (.html, .htm). Script and style blocks are removed from HTML before indexing.

## How do I run the web interface?

Run python -m rag serve and open http://127.0.0.1:8000 in a browser.
