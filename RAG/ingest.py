import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", Path(__file__).parent / "index"))


def load_docs(data_dir: Path):
    docs = []
    for ext in ("*.md", "*.txt"):
        for p in data_dir.rglob(ext):
            loader = TextLoader(str(p), encoding="utf-8")
            for d in loader.load():
                # Attach source path for later citation
                d.metadata["source"] = str(p.relative_to(data_dir))
                docs.append(d)
    return docs


def main():
    load_dotenv()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from: {DATA_DIR}")
    docs = load_docs(DATA_DIR)
    if not docs:
        print("No documents found. Add .md or .txt files to the data/ folder.")
        return

    print(f"Loaded {len(docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks. Embedding + indexing...")

    embeddings = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"))
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(INDEX_DIR))

    print(f"Index saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()

