# components/loaders/pdf_loader.py

import os
import fitz  # PyMuPDF
from core.interfaces import IDocumentLoader
from core.models import Document


class PDFDocumentLoader(IDocumentLoader):
    """
    Loads text from PDF files using PyMuPDF.
    Accepts a single .pdf path or a directory containing .pdf files.
    """

    def load(self, source: str) -> list[Document]:
        pdf_paths = self._resolve_paths(source)
        documents = []

        for path in pdf_paths:
            text = self._extract_text(path)
            if len(text.strip()) < 50:
                print(f"  [skip] {path} — too little text extracted")
                continue

            doc_id = os.path.splitext(os.path.basename(path))[0]
            documents.append(Document(
                doc_id=doc_id,
                text=text,
                metadata={"source": path, "filename": os.path.basename(path)}
            ))
            print(f"  [loaded] {doc_id} — {len(text)} characters")

        return documents

    def _resolve_paths(self, source: str) -> list[str]:
        if os.path.isfile(source) and source.endswith(".pdf"):
            return [source]
        elif os.path.isdir(source):
            paths = [
                os.path.join(source, f)
                for f in os.listdir(source)
                if f.endswith(".pdf")
            ]
            if not paths:
                raise ValueError(f"No .pdf files found in directory: {source}")
            return paths
        else:
            raise ValueError(f"Source must be a .pdf file or directory. Got: {source}")

    def _extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        page_texts = []

        for page in doc:
            raw = page.get_text()
            cleaned = self._clean(raw)
            if cleaned:
                page_texts.append(cleaned)

        doc.close()
        return "\n\n".join(page_texts)

    def _clean(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = [line.strip() for line in lines if len(line.strip()) >= 2]
        return "\n".join(cleaned)