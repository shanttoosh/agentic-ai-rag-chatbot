from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.application.chat.service import MAX_QUESTION_CHARS
from app.domain.models import Answer, AnswerStatus, ConfidenceLevel
from app.schemas.retrieval import RetrievedContext

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class ChatRequest(BaseModel):
    question: Question = Field(examples=["What is Agentic AI?"])
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Chunks to retrieve (defaults to TOP_K)."
    )


class ConfidenceDetails(BaseModel):
    level: ConfidenceLevel
    top_score: float = Field(description="Highest cosine similarity among retrieved chunks.")
    mean_score: float = Field(description="Mean cosine similarity of the top-K chunks.")
    relevant_chunks: int = Field(description="Chunks at or above the retrieval threshold.")
    retrieved_chunks: int


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="True when the answer passed every grounding check.")
    status: AnswerStatus
    confidence: float = Field(
        ge=0,
        le=1,
        description="Retrieval-confidence heuristic in [0, 1]; not a calibrated probability. "
        "0 for fallback answers.",
    )
    confidence_details: ConfidenceDetails
    cited_pages: list[int]
    retrieved_context: list[RetrievedContext]

    @classmethod
    def from_answer(cls, answer: Answer) -> "ChatResponse":
        cited = set(answer.cited_chunk_ids)
        score = answer.confidence
        return cls(
            answer=answer.text,
            grounded=answer.grounded,
            status=answer.status,
            confidence=score.value,
            confidence_details=ConfidenceDetails(
                level=score.level,
                top_score=score.top_score,
                mean_score=score.mean_score,
                relevant_chunks=score.relevant_chunks,
                retrieved_chunks=score.retrieved_chunks,
            ),
            cited_pages=list(answer.cited_pages),
            retrieved_context=[
                RetrievedContext.from_result(
                    result,
                    above_threshold=result.chunk.id in answer.context_chunk_ids,
                    cited=result.chunk.id in cited,
                )
                for result in answer.retrieved
            ],
        )
