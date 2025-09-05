# Multi-step Workflows (LangChain + Ollama)

Deterministic, inspectable LCEL workflows over documents: map-reduce summarization, structured extraction, translation, and classification using a local Gemma model via Ollama.

## Prerequisites

- Python 3.10+
- Ollama installed and running: <https://ollama.ai>
- Models:
  - `ollama pull gemma3:1b`

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place `.txt` or `.md` files under `data/`. A sample file is included.

## Usage

- Summarize (map-reduce, async):
  - `python summarize.py data/multistep_sample.txt`
- Extract structured fields (JSON, async):
  - `python extract.py data/multistep_sample.txt --chunk_size 1000 --chunk_overlap 120`
- Translate to a language (map-reduce, async):
  - `python translate.py data/multistep_sample.txt --lang es`
- Classify with a label set (async):
  - `python classify.py data/multistep_sample.txt --labels tutorial reference tips`

All scripts support `--concurrency N` to control parallel chunk processing.

## Configuration

- LLM: env `LLM_MODEL` (default `gemma3:1b`)
- OLLAMA base URL: `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- Chunking: adjust `chunk_size`/`chunk_overlap` in scripts
- For more reliable extraction, consider a slightly larger local model by setting `LLM_MODEL`.

## Notes

- Workflows use LCEL: prompt → model → parser, with async map steps powered by `ainvoke()` and `asyncio` plus a final reduce step.
- Control parallelism via `--concurrency`; tune chunk size/overlap per dataset.
- Keep prompts concise and explicit; prefer JSON outputs for extraction/classification.
