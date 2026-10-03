from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """Lightweight embeddings (ONNX, no PyTorch). Same model as before."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()