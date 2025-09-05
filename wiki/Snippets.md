# Snippets: LangChain RAG (Gemma + FAISS)

These short snippets match the code and versions used in `RAG/`.

## Minimal RAG Chain (LCEL)

```python
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(
        f"Source: {d.metadata.get('source','')}\n{d.page_content}" for d in docs
    )

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vs = FAISS.load_local("RAG/index", embeddings, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the context. If unsure, say you don't know. Cite sources."),
    ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely."),
])

llm = ChatOllama(model="gemma3:1b")

chain = ({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
} | prompt | llm | StrOutputParser())

print(chain.invoke("What are the main applications of LangChain?"))
```

## Streaming Tokens

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the context to answer briefly."),
    ("user", "Q: {question}\n\nContext:\n{context}"),
])

llm = ChatOllama(model="gemma3:1b")

for token in (prompt | llm | StrOutputParser()).stream({
    "question": "Summarize RAG in 2 lines",
    "context": "RAG augments LLMs with retrieved facts.",
}):
    print(token, end="", flush=True)
```

## Change Top-k and Use MMR

```python
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.3})
```

## Switch Models Quickly

```python
import os
from langchain_ollama import ChatOllama

llm = ChatOllama(model=os.getenv("LLM_MODEL", "gemma3:1b"), temperature=0.2)
```

## Return Answer With Citations

```python
from langchain_core.runnables import RunnableMap

def sources_only(docs):
    # unique, ordered list of sources
    seen, out = set(), []
    for d in docs:
        s = d.metadata.get("source", "")
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

base = RunnableMap({
    "docs": retriever,
    "question": RunnablePassthrough(),
})

answer = (base | RunnableLambda(lambda x: {
    "context": format_docs(x["docs"]),
    "question": x["question"],
}) | prompt | llm | StrOutputParser())

pipeline = base | RunnableLambda(lambda x: {
    "answer": answer.invoke(x),
    "sources": sources_only(x["docs"]),
})

print(pipeline.invoke("What is RAG?"))
```

## Custom Chunking For Ingest

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
```

## Add PDF Loading (optional deps)

```python
# pip install pypdf
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/file.pdf")
pdf_docs = loader.load()
```

## Programmatic Ingest From String

```python
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

docs = [Document(page_content="Hello RAG", metadata={"source": "memory"})]
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vs = FAISS.from_documents(docs, embeddings)
vs.save_local("RAG/index")
```

## Simple Metadata Filtering (client-side)

```python
docs = retriever.invoke("question here")
md_docs = [d for d in docs if str(d.metadata.get("source",""))
           .lower().endswith(".md")]
```

## Inspect Top Matches (debugging)

```python
docs = retriever.get_relevant_documents("What is RAG?")
for i, d in enumerate(docs, 1):
    print(i, d.metadata.get("source"), "\n", d.page_content[:200], "\n---\n")
```
