# Simple Gemma RAG (LangChain + Ollama)

A minimal Retrieval-Augmented Generation (RAG) setup using LangChain, FAISS, and a local Gemma model served by Ollama.

## Prerequisites

- Python 3.10+
- Ollama installed and running: https://ollama.ai
- Models pulled:
  - `ollama pull gemma3:1b` (or another Gemma model you prefer)
  - `ollama pull nomic-embed-text` (embedding model)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ingest Documents

- Put your `.md` or `.txt` files in `data/` (a sample is included).
- Build the FAISS index:

```bash
python ingest.py
```

This creates a local index at `index/`.

## Query

Ask a question against the indexed docs:

```bash
python query.py "What are the main applications of LangChain?"
```

You can also run the small CLI wrapper:

```bash
python app.py
```

## Configuration

- LLM model: change `LLM_MODEL` in `app.py` (default: `gemma3:1b`).
- Embeddings model: change `EMBED_MODEL` in `app.py` (default: `nomic-embed-text`).
- Environment variable `OLLAMA_BASE_URL` can be set if Ollama isn’t at the default `http://127.0.0.1:11434`.

## Notes

- Only `.md` and `.txt` are loaded by default to keep dependencies light.
- For PDFs, you can add `pypdf` and a loader (e.g., `langchain_community.document_loaders.PyPDFLoader`).
- FAISS index is local only; for production, consider a managed vector DB.
