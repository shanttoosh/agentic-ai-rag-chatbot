from typing import Protocol

from app.domain.models import Answer


class QuestionAnswerer(Protocol):
    """Answers a question from the knowledge base (implemented by the RAG pipeline)."""

    def answer(self, question: str, top_k: int) -> Answer: ...
