# components/embedders/sentence_transformer_embedder.py

from sentence_transformers import SentenceTransformer
from core.interfaces import IEmbedder


class SentenceTransformerEmbedder(IEmbedder):
    """
    Local embeddings using sentence-transformers.
    Default model: all-MiniLM-L6-v2
      - 384 dimensions
      - Fast, lightweight, free
      - Downloads once (~90MB), cached after that
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"  [embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  [embedder] Ready")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=True)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]