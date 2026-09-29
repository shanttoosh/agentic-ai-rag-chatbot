"""Answer, grounding and confidence models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.domain.models.retrieval import RetrievalResult


class AnswerStatus(StrEnum):
    """Outcome of a question; everything except ANSWERED returns the fallback text."""

    ANSWERED = "answered"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"  # no chunk passed the threshold
    NOT_IN_CONTEXT = "not_in_context"  # the generator found no answer in the chunks
    FAILED_VALIDATION = "failed_validation"  # the validator found unsupported claims
    DECLINED = "declined"  # the model declined to respond


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ConfidenceScore:
    """Output of ``app.rag.grounding.confidence.compute_confidence``.

    ``value`` is a heuristic in [0, 1] computed from the similarity scores (not a
    probability); the other fields are the retrieval statistics it was computed from.
    """

    value: float
    level: ConfidenceLevel
    top_score: float
    mean_score: float
    relevant_chunks: int
    retrieved_chunks: int

    @classmethod
    def none(
        cls, top_score: float = 0.0, mean_score: float = 0.0, retrieved: int = 0
    ) -> ConfidenceScore:
        return cls(
            value=0.0,
            level=ConfidenceLevel.NONE,
            top_score=top_score,
            mean_score=mean_score,
            relevant_chunks=0,
            retrieved_chunks=retrieved,
        )


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    """What the generator produced, before validation."""

    text: str
    answerable: bool
    cited_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundingVerdict:
    """Result of checking a draft answer against the retrieved context."""

    supported: bool
    unsupported_claims: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Answer:
    """Final result returned to callers."""

    question: str
    text: str
    status: AnswerStatus
    confidence: ConfidenceScore
    retrieved: tuple[RetrievalResult, ...]  # everything the vector store returned
    context_chunk_ids: frozenset[str]  # chunks that passed the threshold and went to the LLM
    cited_chunk_ids: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()

    @property
    def grounded(self) -> bool:
        return self.status is AnswerStatus.ANSWERED

    @property
    def cited_pages(self) -> tuple[int, ...]:
        by_id = {result.chunk.id: result.chunk.page_number for result in self.retrieved}
        return tuple(sorted({by_id[i] for i in self.cited_chunk_ids if i in by_id}))
