from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.domain.models import DraftAnswer, RetrievalResult

OutputT = TypeVar("OutputT", bound=BaseModel)


class StructuredLLM(Protocol):
    """A language model that returns output validated against a Pydantic schema."""

    @property
    def model_name(self) -> str: ...

    def complete(self, *, system: str, prompt: str, output_type: type[OutputT]) -> OutputT:
        """Raises ``LLMRefusalError`` if the model declines, ``LLMError`` on failure."""
        ...


class AnswerGenerator(Protocol):
    """Writes an answer to the question using only the given context."""

    def generate(self, question: str, context: Sequence[RetrievalResult]) -> DraftAnswer: ...
