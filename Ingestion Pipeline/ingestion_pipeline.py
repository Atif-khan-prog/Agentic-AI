import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def load_docs():
    loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()
    if len(documents) == 0 :
        print('no docs')
    else:
        print('whatsapp')

    return documents

def split_docs(docs, chunk_size = 1000, chunk_overlap=0):
    textSplitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = textSplitter.split_documents(docs)

    if len(chunks) == 0:
        print('No chunks bro')
    else:
        print('shi is chunked', len(chunks))

    return chunks

def create_vector_store(chunks, database_path ='db/chroma_db'):
     
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
    )

    vectorestore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=database_path,
        collection_metadata={'hnsw:space':'cosine'}
    )

    print('vector store has successfully created')
    return vectorestore



def main():
    #Load files
    docs = load_docs()
    #Create Chunks
    chunks = split_docs(docs)
    #Create vector store
    vectorstore = create_vector_store(chunks)

if __name__ == '__main__':
    main()