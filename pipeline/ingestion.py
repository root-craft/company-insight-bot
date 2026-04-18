# pipeline/ingestion.py

from core.interfaces import IDocumentLoader, IChunker, IEmbedder, IVectorStore


class IngestionPipeline:
    """
    Orchestrates: load → chunk → embed → store.
    Depends only on interfaces — concrete classes injected from outside.
    """

    def __init__(
        self,
        loader: IDocumentLoader,
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, source: str) -> dict:
        try:
            print("\n[ingestion] Step 1: Loading documents...")
            documents = self.loader.load(source)
            if not documents:
                return {"status": "error", "message": "No documents loaded."}

            print("\n[ingestion] Step 2: Chunking...")
            chunks = self.chunker.chunk(documents)
            if not chunks:
                return {"status": "error", "message": "Chunking produced no chunks."}

            print("\n[ingestion] Step 3: Embedding...")
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed(texts)

            print("\n[ingestion] Step 4: Storing...")
            self.vector_store.clear()
            self.vector_store.add(chunks, embeddings)

            return {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks)
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}