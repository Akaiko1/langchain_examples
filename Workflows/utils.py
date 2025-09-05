import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_text_files(paths: List[str]) -> str:
    parts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


def get_llm():
    load_dotenv()
    model = os.getenv("LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model)

