# LangChain Examples

Practical, minimal examples for building with LangChain and friends (LangGraph, FAISS, Ollama, etc.). Start locally, then adapt to your stack.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain 0.2.x](https://img.shields.io/badge/LangChain-0.2.x-1C3C3C)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-000000)](https://ollama.ai)

## Contents

- Main Applications of LangChain
- RAG Demo (Ollama + FAISS)
- Workflows Demo (Map-Reduce, LCEL)
- Tool-Using Agent Demo (Terminal)
- Troubleshooting

## Main Applications

- RAG (Retrieval-Augmented Generation): Answer questions over your docs, wikis, tickets, and codebases using chunking, embeddings, retrievers, re-ranking, and citations.
- Multi-step Workflows: Summarize, extract, translate, and classify at scale using deterministic chains and map-reduce patterns.
- Tool-Using Agents: Safely call APIs, databases, search, and internal tools with plan→act loops (often built with LangGraph for reliability).
- Structured Extraction: Produce typed JSON or fill schemas from semi-structured text via output parsers and validation.
- Conversational AI with Memory: Build chat experiences that remember context and can take actions through function/tool calling.
- Code & Data Assistants: Repo Q&A, refactoring helpers, SQL generation over warehouses/lakes, and “chat with your data.”

## RAG Demo (Ollama + FAISS)

A minimal Retrieval-Augmented Generation example lives in `RAG/` and lets you ask questions over local `.md`/`.txt` files using a Gemma model served by Ollama.

Prerequisites

- Install Ollama and pull models:
  - `ollama pull gemma3:1b`
  - `ollama pull nomic-embed-text`
- Python 3.10+

Quickstart

- Create a virtual environment and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r RAG/requirements.txt`
- Ingest sample docs and build the FAISS index:
  - `python RAG/ingest.py`
- Ask questions:
  - `python RAG/query.py "What are the main applications of LangChain?"`
  - or interactive: `python RAG/app.py` (streams tokens)

Configuration

- Models: `LLM_MODEL` (default `gemma3:1b`), `EMBED_MODEL` (default `nomic-embed-text`).
- Paths: `INDEX_DIR`, `DATA_DIR` (default to subfolders of `RAG/`).
- Ollama URL: `OLLAMA_BASE_URL` if not `http://127.0.0.1:11434`.

Project Structure

```text
RAG/
  README.md           # RAG-specific docs
  requirements.txt    # LangChain, FAISS, dotenv
  ingest.py           # Build local FAISS index from data/
  query.py            # Query the index with Gemma via Ollama
  app.py              # Simple interactive CLI
  data/               # Sample .md/.txt files
  index/              # Generated FAISS index (gitignored)
```

## Workflows Demo (Map-Reduce, LCEL)

Deterministic multi-step pipelines for summarization, structured extraction, translation, and classification using a local Gemma model via Ollama.

Prerequisites

- `ollama pull gemma3:1b`
- Python 3.10+

Quickstart (async)

- Create venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r Workflows/requirements.txt`
- Try examples with the included sample:
  - `python Workflows/summarize.py Workflows/data/multistep_sample.txt --concurrency 4`
  - `python Workflows/extract.py Workflows/data/multistep_sample.txt --concurrency 4`
  - `python Workflows/translate.py Workflows/data/multistep_sample.txt --lang es --concurrency 4`
  - `python Workflows/classify.py Workflows/data/multistep_sample.txt --labels tutorial reference tips --concurrency 4`

Configuration

- `LLM_MODEL` (default `gemma3:1b`), `OLLAMA_BASE_URL` for non-default hosts.
- Scripts support `--concurrency` to control async map parallelism.
- Adjust chunking via `--chunk_size`/`--chunk_overlap` where available.

## Tool-Using Agent Demo (Terminal)

A minimal ReAct-style agent lives in `Agents/`. It can shell out to search within this repository and list directory contents before answering.

Prerequisites

- `ollama pull gemma3:1b`
- Python 3.10+

Quickstart

- Create a virtual environment and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r Agents/requirements.txt`
- Launch the interactive CLI:
  - `python Agents/terminal_agent.py`
- Ask a single question without entering the loop:
  - `python Agents/terminal_agent.py --question "Where is repo_search defined?"`
- Hide the default intermediate trace:
  - `python Agents/terminal_agent.py --question "What are the main repo folders?" --hide-steps`
  - The CLI no longer injects fallback answers; outputs are agent-driven.

Configuration

- `LLM_MODEL` selects the Ollama chat model (default `gemma3:1b`). Set `LLM_TEMPERATURE` to adjust sampling (default `0.2`).
- `OLLAMA_BASE_URL` points to a non-default Ollama host if needed.
- Built-in tools include `repo_search` (text search), `list_repo` (directory listings), and `count_occurrences` (case-sensitive token counts). `repo_search` prefers `rg` (ripgrep) but falls back to a pure-Python scan when unavailable. Tool traces print by default—pass `--hide-steps` to suppress them. The CLI does not inject fallback answers.

## Troubleshooting

- Ollama not reachable: ensure the daemon is running; set `OLLAMA_BASE_URL`.
- No docs indexed: add `.md`/`.txt` files to `RAG/data/` and rerun `python RAG/ingest.py`.
- Import errors: verify you’re in the venv and ran `pip install -r RAG/requirements.txt`.

## References

- LangChain Docs: <https://python.langchain.com/>
- LangGraph: <https://langgraph.readthedocs.io/>
- Ollama: <https://ollama.ai/>
