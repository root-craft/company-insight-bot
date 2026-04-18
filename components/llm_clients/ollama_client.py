# components/llm_clients/ollama_client.py

import ollama
from core.interfaces import ILLMClient


class OllamaLLMClient(ILLMClient):
    """
    LLM response generation via local Ollama.
    Ollama must be running: `ollama serve`
    Model must be pulled: `ollama pull llama3.2`
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        print(f"  [llm] Using Ollama model: {model}")

    def complete(self, messages: list[dict]) -> str:
        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        return response["message"]["content"]