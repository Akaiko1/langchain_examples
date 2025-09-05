import argparse
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import read_text_files, chunk_text, get_llm


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Input .txt/.md file")
    ap.add_argument("--lang", default="es", help="Target language code or name")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    text = read_text_files([args.path])
    chunks = chunk_text(text)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the passage accurately and naturally."),
        ("user", "Translate to {lang}:\n\n{passage}"),
    ])
    chain = prompt | get_llm() | StrOutputParser()

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def run_translate(passage: str):
        async with sem:
            return await chain.ainvoke({"lang": args.lang, "passage": passage})

    translated = await asyncio.gather(*[asyncio.create_task(run_translate(c)) for c in chunks])
    print("\n\n".join(translated))


if __name__ == "__main__":
    asyncio.run(main())
