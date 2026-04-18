# app.py

import streamlit as st
from config import Config
from components.loaders.pdf_loader import PDFDocumentLoader
from components.chunkers.recursive_chunker import RecursiveCharacterChunker
from components.embedders.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from components.vector_stores.chroma_store import ChromaVectorStore
from components.llm_clients.ollama_client import OllamaLLMClient
from pipeline.ingestion import IngestionPipeline
from pipeline.query import QueryPipeline
from prompt.builder import PromptBuilder

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Company Research Chatbot", page_icon="🏢", layout="centered"
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
        vector_store=vector_store,
    )

    query_pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        top_k=Config.TOP_K,
        min_score=Config.MIN_SCORE,
    )

    return ingestion, query_pipeline


ingestion_pipeline, query_pipeline = build_components()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # full chat history

if "indexed" not in st.session_state:
    st.session_state["indexed"] = False  # whether PDFs have been ingested

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Company Documents")
    st.caption("Point to a PDF file or a folder containing PDFs.")

    source_path = st.text_input("PDF file or folder path", value=Config.PDF_SOURCE_DIR)

    if st.button("Index Documents", use_container_width=True):
        with st.spinner("Loading, chunking, embedding... please wait."):
            result = ingestion_pipeline.run(source_path)

        if result["status"] == "success":
            st.success(
                f"✅ Indexed {result['documents_loaded']} document(s), "
                f"{result['chunks_created']} chunks."
            )
            st.session_state["indexed"] = True
            st.session_state["messages"] = []  # clear chat on re-index
        else:
            st.error(f"❌ {result['message']}")

    if st.session_state["indexed"]:
        st.divider()
        st.success("Documents loaded. Ask away!")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # ── System info ───────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ System Info")

    st.markdown(
        f"""
    | Setting | Value |
    |---|---|
    | Embedding model | `{Config.EMBEDDING_MODEL}` |
    | LLM | `{Config.OLLAMA_MODEL}` |
    | Chunk size | `{Config.CHUNK_SIZE}` chars |
    | Chunk overlap | `{Config.CHUNK_OVERLAP}` chars |
    | Top-K retrieval | `{Config.TOP_K}` chunks |
    | Min score | `{Config.MIN_SCORE}` |
    """
    )


# ── Main chat area ────────────────────────────────────────
if not st.session_state["indexed"]:
    st.info("👈 Load company PDFs from the sidebar to start chatting.")

else:
    # Suggested starter questions
    st.markdown("**Suggested questions:**")
    suggestions = [
        "What does this company do?",
        "What are the company's values?",
        "What is the culture like?",
        "Who leads the company?",
        "What products or services do they offer?",
    ]

    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state["pending_query"] = suggestion

    st.divider()

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Resolve query — typed or clicked suggestion
    query = st.chat_input("Ask about the company...")
    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")

    # Handle query
    if query:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        # Run pipeline and stream answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_pipeline.run(
                    query=query, chat_history=st.session_state["messages"][:-1]
                )

            st.markdown(result["answer"])

            # Show retrieved source chunks (collapsed by default)
            with st.expander("📎 Retrieved context", expanded=False):
                for chunk in result["retrieved_chunks"]:
                    source = chunk.metadata.get("filename", chunk.doc_id)
                    st.markdown(
                        f"**Source:** `{source}` | **Score:** `{chunk.score:.3f}`"
                    )
                    st.caption(chunk.text)
                    st.divider()

        st.session_state["messages"].append(
            {"role": "assistant", "content": result["answer"]}
        )
