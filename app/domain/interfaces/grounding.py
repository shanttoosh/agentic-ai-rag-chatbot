from collections.abc import Sequence
from typing import Protocol

from app.domain.models import DraftAnswer, GroundingVerdict, RetrievalResult


class GroundingValidator(Protocol):
    """Checks that every claim in a draft answer is supported by the context."""

    def validate(
        self, question: str, draft: DraftAnswer, context: Sequence[RetrievalResult]
    ) -> GroundingVerdict: ...
