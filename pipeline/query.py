# pipeline/query.py

from core.interfaces import IEmbedder, IVectorStore, ILLMClient
from prompt.builder import PromptBuilder


class QueryPipeline:
    """
    Orchestrates: embed query → retrieve → build prompt → generate answer.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        llm_client: ILLMClient,
        prompt_builder: PromptBuilder,
        top_k: int = 5,
        min_score: float = 0.3
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.min_score = min_score

    def run(self, query: str, chat_history: list[dict] = None) -> dict:
        print(f"\n[query] '{query}'")

        print("[query] Step 1: Embedding query...")
        query_embedding = self.embedder.embed_one(query)

        print("[query] Step 2: Retrieving chunks...")
        retrieved = self.vector_store.search(query_embedding, top_k=self.top_k)

        # Filter by minimum score, but keep all results if none pass threshold
        filtered = [r for r in retrieved if r.score >= self.min_score]
        chunks_to_use = filtered if filtered else retrieved

        print(f"[query] Using {len(chunks_to_use)} chunks (scores: "
              f"{[round(c.score, 3) for c in chunks_to_use]})")

        print("[query] Step 3: Building prompt...")
        messages = self.prompt_builder.build(query, chunks_to_use, chat_history)

        print("[query] Step 4: Generating answer...")
        answer = self.llm_client.complete(messages)

        return {
            "answer": answer,
            "retrieved_chunks": chunks_to_use
        }