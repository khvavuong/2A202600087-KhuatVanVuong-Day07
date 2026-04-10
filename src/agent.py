from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question or not question.strip():
            return "Please provide a valid question."

        results = self._store.search(question, top_k=top_k)

        if not results:
            return "I don't know the answer based on the available information."

        context_chunks = [r["content"] for r in results]
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        # 4. Call LLM
        response = self._llm_fn(prompt)

        return response.strip()
