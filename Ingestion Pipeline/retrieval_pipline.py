import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from groq import Groq


database_path = 'db/chroma_db'

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


db = Chroma(
    persist_directory=database_path,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space': 'cosine'}
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_answer(query):

    retriever = db.as_retriever(
    search_kwargs={'k': 10}
)

    candidates = retriever.invoke(query)

    # Create pairs: [question, document]
    pairs = [
        [query, doc.page_content]
        for doc in candidates
    ]

    # Score each document against the question
    scores = reranker.predict(pairs)

    # Sort documents by score, highest first
    ranked_docs = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    # Take the best 3 documents
    relevant_ans = [
        doc for score, doc in ranked_docs[:3]
    ]

    # Augmentation: combine retrieved chunks into context
    context = '\n\n'.join(
        doc.page_content for doc in relevant_ans
    )

    # Create prompt containing context + question
    prompt = f"""
You are a RAG assistant.

Answer the question ONLY using the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Do not use your own knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generation: send augmented prompt to Ollama
    response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

    return response.choices[0].message.content

    
