# prompt/builder.py

from core.models import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful assistant that helps job candidates prepare for HR interviews.
You answer questions about a company based ONLY on the provided context.

Rules:
- Use ONLY the information from the provided context to answer.
- If the context does not contain enough information, say: "I couldn't find that information in the provided documents."
- Frame answers from a candidate's perspective — what would be useful to know before an interview?
- Be concise but complete.
- Mention the source document when relevant."""


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