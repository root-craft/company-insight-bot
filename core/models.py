# core/models.py

from dataclasses import dataclass, field


@dataclass
class Document:
    """
    Output of a loader.
    Represents one PDF file's worth of raw text.
    """
    doc_id: str          # filename without extension, e.g. "google_overview"
    text: str            # full extracted text
    metadata: dict = field(default_factory=dict)  # e.g. {"source": "google_overview.pdf"}


@dataclass
class Chunk:
    """
    Output of a chunker.
    One piece of a Document, sized for embedding.
    """
    chunk_id: str        # e.g. "google_overview_3"
    doc_id: str          # which Document this came from
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """
    Output of a vector store search.
    A Chunk that matched a query, with its similarity score.
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float         # cosine similarity, 0.0 to 1.0
    metadata: dict = field(default_factory=dict)