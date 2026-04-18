# prompt/builder.py

from core.models import RetrievedChunk

SYSTEM_PROMPT = """You are an expert research assistant helping job candidates prepare for HR interviews.

Your job is to answer questions about a company using ONLY the context documents provided to you.

Guidelines:
- Answer based ONLY on the provided context. Never use outside knowledge.
- When you use information from a specific document, mention it naturally in your answer.
  Example: "According to the company overview, they focus on..."
- If the context contains partial information, share what you found and clearly state what's missing.
- If the context contains NO relevant information at all, respond with:
  "I couldn't find information about that in the provided documents. 
   Try rephrasing your question or loading additional company documents."
- Keep answers concise but complete — a candidate reading before an interview needs clarity, not essays.
- Use a friendly, professional tone."""


class PromptBuilder:
    """
    Builds the messages array sent to the LLM.
    Combines: system prompt + chat history + retrieved context + user query.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        chat_history: list[dict] = None
    ) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]

        # Include last 3 turns of chat history (6 messages: 3 user + 3 assistant)
        if chat_history:
            messages.extend(chat_history[-6:])

        # Format retrieved chunks into a readable context block
        context_block = self._format_context(retrieved_chunks)

        user_message = f"""Use the following context to answer the question.

{context_block}

Question: {query}"""

        messages.append({"role": "user", "content": user_message})
        return messages

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("filename", chunk.doc_id)
            parts.append(
                f"--- Context {i + 1} (source: {source}, score: {chunk.score:.3f}) ---\n{chunk.text}"
            )

        return "\n\n".join(parts)