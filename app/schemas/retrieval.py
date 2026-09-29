from pydantic import BaseModel, Field

from app.domain.models import RetrievalResult


class RetrievedContext(BaseModel):
    chunk_id: str = Field(examples=["page21_chunk00"])
    page: int
    section: str | None = None
    score: float = Field(description="Cosine similarity between question and chunk.")
    above_threshold: bool = Field(
        description="Passed the retrieval threshold and was sent to the LLM."
    )
    cited: bool = Field(description="Cited by the final answer.")
    text: str

    @classmethod
    def from_result(
        cls, result: RetrievalResult, *, above_threshold: bool, cited: bool
    ) -> "RetrievedContext":
        return cls(
            chunk_id=result.chunk.id,
            page=result.chunk.page_number,
            section=result.chunk.section,
            score=round(result.score, 4),
            above_threshold=above_threshold,
            cited=cited,
            text=result.chunk.text,
        )
