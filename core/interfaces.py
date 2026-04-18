# core/interfaces.py

from abc import ABC, abstractmethod
from core.models import Document, Chunk, RetrievedChunk


class IDocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load raw text from a file path or directory."""
        pass


class IChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into smaller chunks."""
        pass


class IEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of strings to embedding vectors."""
        pass

    @abstractmethod
    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Used at query time."""
        pass


class IVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embedding vectors."""
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return top_k chunks most similar to the query embedding."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored data. Called before re-indexing."""
        pass


class ILLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        """
        Generate a response given a messages array.
        Messages follow OpenAI chat format:
        [{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        pass