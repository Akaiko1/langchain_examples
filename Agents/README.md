# Terminal Agent (LangChain + Ollama)

A small LangChain ReAct agent that can search this repository via a shell tool and answer follow-up questions using a local Ollama model (Gemma by default).

## Prerequisites

- Python 3.10+
- Ollama running locally with a chat-capable model pulled, e.g.:
  ```bash
  ollama pull gemma3:1b
  ```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Agents/requirements.txt
```

## Usage

Interactive shell:

```bash
python Agents/terminal_agent.py
```

Ask a single question and exit:

```bash
python Agents/terminal_agent.py --question "Where is the classify workflow?"
```

Hide intermediate tool calls:

```bash
python Agents/terminal_agent.py --question "What are the main repo folders?" --hide-steps
```

Note: The agent fully drives reasoning; there are no CLI fallback syntheses.

## How it Works

- Loads `LLM_MODEL` from the environment (defaults to `gemma3:1b`).
- Exposes a `repo_search` tool that shells out to `rg` (ripgrep) when available, falling back to a pure-Python search over common text files.
- Provides a `list_repo` tool for listing directories/files and a `count_occurrences` tool for literal string frequency counts.
- Uses a ReAct-style prompt that encourages (but doesn’t force) tool usage. Intermediate tool steps are printed by default; pass `--hide-steps` to suppress them. The CLI no longer injects fallback answers—outputs are purely agent-driven.

Set `LLM_MODEL` (and optionally `OLLAMA_BASE_URL`) to point at a different local model if desired. You can also control sampling with `LLM_TEMPERATURE` (default `0.2`).
