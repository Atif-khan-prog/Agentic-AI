import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

from embeddings import FastEmbedEmbeddings

DATABASE_PATH = "db/chroma_db"


def load_docs(folder="docs"):
    docs = []
    for path in Path(folder).rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": str(path)}))

    print(f"Loaded {len(docs)} documents" if docs else "No docs found")
    return docs


def split_docs(docs, chunk_size=1000, chunk_overlap=0):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks" if chunks else "No chunks created")
    return chunks


def create_vector_store(chunks, database_path=DATABASE_PATH):
    # Start fresh so re-running doesn't create duplicates
    shutil.rmtree(database_path, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=FastEmbedEmbeddings(),
        persist_directory=database_path,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Vector store created successfully")
    return vectorstore


def main():
    docs = load_docs()
    if not docs:
        return
    chunks = split_docs(docs)
    if not chunks:
        return
    create_vector_store(chunks)


if __name__ == "__main__":
    main()