# test_pipeline.py

from components.loaders.pdf_loader import PDFDocumentLoader
from components.chunkers.recursive_chunker import RecursiveCharacterChunker
from components.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from components.vector_stores.chroma_store import ChromaVectorStore
from components.llm_clients.ollama_client import OllamaLLMClient
from pipeline.ingestion import IngestionPipeline
from pipeline.query import QueryPipeline
from prompt.builder import PromptBuilder

PDF_SOURCE = ".\\data\\pdfs"  # folder, or change to a single file path

print("=" * 50)
print("Initialising components...")
print("=" * 50)

embedder = SentenceTransformerEmbedder()
vector_store = ChromaVectorStore()
llm_client = OllamaLLMClient(model="llama3.2")  # change if you pulled a different model
prompt_builder = PromptBuilder()

ingestion = IngestionPipeline(
    loader=PDFDocumentLoader(),
    chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50),
    embedder=embedder,
    vector_store=vector_store
)

query_pipeline = QueryPipeline(
    embedder=embedder,
    vector_store=vector_store,
    llm_client=llm_client,
    prompt_builder=prompt_builder,
    top_k=5,
    min_score=0.3
)

print("\n" + "=" * 50)
print("Running ingestion...")
print("=" * 50)

result = ingestion.run(PDF_SOURCE)
print(f"\nIngestion result: {result}")

if result["status"] != "success":
    print("Ingestion failed. Check your PDF path.")
    exit(1)

print("\n" + "=" * 50)
print("Running test queries...")
print("=" * 50)

test_questions = [
    "What does this company do?",
    "What are the company's core values?",
    "Who leads the company?",
]

for question in test_questions:
    result = query_pipeline.run(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Sources: {[c.metadata.get('filename') for c in result['retrieved_chunks']]}")
    print("-" * 40)