# config.py

class Config:
    # Paths
    PDF_SOURCE_DIR: str = "./data/pdfs"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector store
    COLLECTION_NAME: str = "rag_collection"
    TOP_K: int = 5
    MIN_SCORE: float = 0.3

    # LLM
    OLLAMA_MODEL: str = "llama3.2"  # change if you pulled a different model