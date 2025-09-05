import argparse
import asyncio
import collections
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import read_text_files, chunk_text, get_llm


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Input .txt/.md file")
    ap.add_argument("--labels", nargs="+", required=True, help="Label set")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    text = read_text_files([args.path])
    chunks = chunk_text(text)

    labels_str = ", ".join(args.labels)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the passage strictly into one label from the provided set."),
        ("user", "Labels: {labels}\n\nPassage:\n{passage}\n\nRespond with one label only."),
    ])
    chain = prompt | get_llm() | StrOutputParser()

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def run_classify(passage: str):
        async with sem:
            return await chain.ainvoke({"labels": labels_str, "passage": passage})

    preds = await asyncio.gather(*[asyncio.create_task(run_classify(c)) for c in chunks])
    preds = [p.strip().split()[0] for p in preds if p and isinstance(p, str)]
    counter = collections.Counter(preds)
    winner, _ = counter.most_common(1)[0] if counter else ("unknown", 0)

    print("Predictions per chunk:")
    for lbl, cnt in counter.most_common():
        print(f"- {lbl}: {cnt}")
    print(f"\nMajority label: {winner}")


if __name__ == "__main__":
    asyncio.run(main())
