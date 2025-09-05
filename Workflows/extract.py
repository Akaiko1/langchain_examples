import argparse
import asyncio
import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from utils import read_text_files, chunk_text, get_llm


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Input .txt/.md file")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()

    # Ensure .env variables (LLM_MODEL, OLLAMA_BASE_URL) are loaded
    load_dotenv()

    text = read_text_files([args.path])
    chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    class Info(BaseModel):
        topics: List[str] = Field(default_factory=list, description="High-level topics covered.")
        actions: List[str] = Field(default_factory=list, description="Key recommended or described actions.")
        entities: List[str] = Field(default_factory=list, description="Named entities like products, libraries, tools.")

    parser = PydanticOutputParser(pydantic_object=Info)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an information extraction assistant. "
                "Extract concise topics, actions, and entities from the passage. "
                "Return a JSON object with 'topics', 'actions', and 'entities' keys. "
                "If no items are found for a category, return an empty list.",
            ),
            (
                "user",
                "{format_instructions}\n\nPassage:\n{passage}",
            ),
        ]
    )

    llm = get_llm(temperature=0)
    chain = prompt | llm | parser

    sem = asyncio.Semaphore(max(1, args.concurrency))

    fmt = parser.get_format_instructions()

    async def run_extract(passage: str):
        async with sem:
            try:
                return await chain.ainvoke({
                    "format_instructions": fmt,
                    "passage": passage,
                })
            except Exception:
                # Fallback: ask again with a simpler instruction
                simple = ChatPromptTemplate.from_messages([
                    ("system", "Output JSON with keys: topics, actions, entities. Lists only."),
                    ("user", "Passage:\n{passage}"),
                ]) | llm | parser
                return await simple.ainvoke({"passage": passage})

    partials = await asyncio.gather(*[asyncio.create_task(run_extract(c)) for c in chunks])

    merged = {"topics": set(), "actions": set(), "entities": set()}
    for p in partials:
        try:
            obj = p if isinstance(p, Info) else Info(**json.loads(p))
        except Exception:
            continue
        for key in ("topics", "actions", "entities"):
            values = getattr(obj, key, [])
            for v in values or []:
                if isinstance(v, str) and v.strip():
                    merged[key].add(v.strip())

    # If everything empty, try a single-pass extraction over the full text as a fallback.
    if not any(merged.values()):
        try:
            full = await chain.ainvoke({
                "format_instructions": fmt,
                "passage": text[:12000],  # safety limit
            })
            if isinstance(full, Info):
                for key in ("topics", "actions", "entities"):
                    for v in getattr(full, key, []) or []:
                        if isinstance(v, str) and v.strip():
                            merged[key].add(v.strip())
        except Exception:
            pass

    result = {k: sorted(list(v)) for k, v in merged.items()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
