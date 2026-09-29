from typing import Protocol

from app.domain.models import RetrievalResult


class Retriever(Protocol):
    """Finds the chunks most relevant to a question."""

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]: ...
