import argparse
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import read_text_files, chunk_text, get_llm


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Input .txt/.md files")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    text = read_text_files(args.paths)
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical summarizer."),
        ("user", "Summarize the following passage in 3-5 bullet points.\n\n{passage}"),
    ])
    map_chain = map_prompt | get_llm() | StrOutputParser()

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def run_map(passage: str):
        async with sem:
            return await map_chain.ainvoke({"passage": passage})

    summaries = await asyncio.gather(*[asyncio.create_task(run_map(c)) for c in chunks])

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", "You combine partial summaries into a single coherent summary."),
        ("user", "Combine the bullet lists into one crisp summary (6-8 bullets):\n\n{points}"),
    ])
    reduce_chain = reduce_prompt | get_llm() | StrOutputParser()

    combined = "\n\n".join(summaries)
    final_summary = await reduce_chain.ainvoke({"points": combined})
    print(final_summary)


if __name__ == "__main__":
    asyncio.run(main())
