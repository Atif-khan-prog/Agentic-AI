import os
from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

load_dotenv()

DATABASE_PATH = "db/chroma_db"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class FastEmbedEmbeddings(Embeddings):
    """Lightweight embeddings (ONNX, no PyTorch). Same model as before."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()


embedding_model = FastEmbedEmbeddings()

db = Chroma(
    persist_directory=DATABASE_PATH,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


def get_answer(query: str) -> str:
    # Retrieval: top 4 most similar chunks
    docs = db.similarity_search(query, k=4)

    # Augmentation: combine chunks into context
    context = "\n\n".join(doc.page_content for doc in docs)

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

    # Generation: send augmented prompt to Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

