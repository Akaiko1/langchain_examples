import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def format_docs(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        parts.append(f"Source: {src}\n{d.page_content}")
    return "\n\n".join(parts)


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        sys.exit(1)
    question = sys.argv[1]

    base = Path(__file__).parent
    index_dir = Path(os.getenv("INDEX_DIR", base / "index"))

    embeddings = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"))
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer. If unsure, say you don't know. Cite sources when possible."),
        ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely."),
    ])

    llm = ChatOllama(model=os.getenv("LLM_MODEL", "gemma3:1b"))

    chain = ({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser())

    answer = chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
