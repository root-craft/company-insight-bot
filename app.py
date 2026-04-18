# app.py

import streamlit as st
from config import Config
from components.loaders.pdf_loader import PDFDocumentLoader
from components.chunkers.recursive_chunker import RecursiveCharacterChunker
from components.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from components.vector_stores.chroma_store import ChromaVectorStore
from components.llm_clients.ollama_client import OllamaLLMClient
from pipeline.ingestion import IngestionPipeline
from pipeline.query import QueryPipeline
from prompt.builder import PromptBuilder

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Company Research Chatbot",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Company Research Chatbot")
st.caption("Upload company PDFs to prepare for your HR interview.")

# ── Build components once per session ─────────────────────
@st.cache_resource
def build_components():
    cfg = Config()
    embedder = SentenceTransformerEmbedder(model_name=cfg.EMBEDDING_MODEL)
    vector_store = ChromaVectorStore(collection_name=cfg.COLLECTION_NAME)
    llm_client = OllamaLLMClient(model=cfg.OLLAMA_MODEL)
    prompt_builder = PromptBuilder()

    ingestion = IngestionPipeline(
        loader=PDFDocumentLoader(),
        chunker=RecursiveCharacterChunker(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP),
        embedder=embedder,
        vector_store=vector_store
    )

    query_pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        top_k=Config.TOP_K,
        min_score=Config.MIN_SCORE
    )

    return ingestion, query_pipeline

ingestion_pipeline, query_pipeline = build_components()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []      # full chat history

if "indexed" not in st.session_state:
    st.session_state["indexed"] = False    # whether PDFs have been ingested