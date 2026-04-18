# components/chunkers/recursive_chunker.py

from core.interfaces import IChunker
from core.models import Document, Chunk


class RecursiveCharacterChunker(IChunker):
    """
    Splits text by trying separators from largest semantic boundary
    (paragraph) down to smallest (character).
    Implements overlap so context isn't lost at chunk boundaries.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []

        for doc in documents:
            text_chunks = self._split_text(doc.text)
            for i, text in enumerate(text_chunks):
                all_chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{i}",
                    doc_id=doc.doc_id,
                    text=text,
                    metadata={**doc.metadata, "chunk_index": i}
                ))
            print(f"  [chunked] {doc.doc_id} → {len(text_chunks)} chunks")

        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        separator = self._find_separator(text)
        splits = text.split(separator) if separator else list(text)
        return self._merge_splits(splits, separator)

    def _find_separator(self, text: str) -> str:
        for sep in self.SEPARATORS:
            if sep in text:
                return sep
        return ""

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        chunks = []
        current_parts = []
        current_length = 0

        for part in splits:
            part_length = len(part)
            sep_length = len(separator) if current_parts else 0
            addition_length = sep_length + part_length

            if current_length + addition_length > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_parts, current_length = self._compute_overlap(
                    current_parts, separator
                )

            current_parts.append(part)
            current_length += addition_length

        last_chunk = separator.join(current_parts).strip()
        if last_chunk:
            chunks.append(last_chunk)

        return chunks

    def _compute_overlap(
        self, parts: list[str], separator: str
    ) -> tuple[list[str], int]:
        overlap_parts = []
        overlap_length = 0

        for part in reversed(parts):
            part_len = len(part) + len(separator)
            if overlap_length + part_len <= self.chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_length += part_len
            else:
                break

        return overlap_parts, overlap_length