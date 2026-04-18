# components/vector_stores/chroma_store.py

import chromadb
from core.interfaces import IVectorStore
from core.models import Chunk, RetrievedChunk


class ChromaVectorStore(IVectorStore):
    """
    Vector storage and similarity search using ChromaDB (in-memory).
    Uses cosine similarity.
    """

    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.client = chromadb.Client()  # in-memory, no disk persistence needed
        self.collection = self._get_or_create_collection()
        print(f"  [vector store] Collection '{collection_name}' ready")

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in chunks]
        )
        print(f"  [vector store] Stored {len(chunks)} chunks")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append(RetrievedChunk(
                chunk_id=results["ids"][0][i],
                doc_id=results["metadatas"][0][i].get("doc_id", ""),
                text=results["documents"][0][i],
                score=1 - results["distances"][0][i],  # distance → similarity
                metadata=results["metadatas"][0][i]
            ))

        return sorted(retrieved, key=lambda x: x.score, reverse=True)

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        print(f"  [vector store] Cleared")