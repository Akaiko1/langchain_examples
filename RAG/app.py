import os
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# Defaults can be overridden via env vars or editing below
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(os.path.dirname(__file__), "index"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def format_docs(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        parts.append(f"Source: {src}\n{d.page_content}")
    return "\n\n".join(parts)


def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer. If unsure, say you don't know. Cite sources when possible."),
        ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely.")
    ])


def get_llm():
    return ChatOllama(model=LLM_MODEL)


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_retriever():
    embeddings = get_embeddings()
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 4})


def make_chain():
    retriever = get_retriever()
    prompt = build_prompt()
    llm = get_llm()
    chain = ({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser())
    return chain


def main():
    load_dotenv()
    print("Simple Gemma RAG (streaming). Type your question (or 'exit').")
    chain = make_chain()
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in {"exit", ":q", "quit"}:
            print("Bye.")
            break
        try:
            print("")
            for chunk in chain.stream(q):
                print(chunk, end="", flush=True)
            print("")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
