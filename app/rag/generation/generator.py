"""Grounded answer generation."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.domain.interfaces import StructuredLLM
from app.domain.models import DraftAnswer, RetrievalResult
from app.rag.generation.prompts import GENERATION_PROMPT, generation_prompt, load_prompt


class GenerationOutput(BaseModel):
    """JSON schema the LLM must fill in."""

    model_config = ConfigDict(extra="forbid")  # -> additionalProperties: false

    answerable: bool = Field(
        description="True only if the excerpts contain enough information to answer."
    )
    answer: str = Field(
        description="Answer based only on the excerpts, with inline page references. "
        "Empty when answerable is false."
    )
    cited_excerpts: list[int] = Field(description="Ids of the excerpts the answer relies on.")


class GroundedAnswerGenerator:
    """Asks the LLM for an answer restricted to the retrieved excerpts."""

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def generate(self, question: str, context: Sequence[RetrievalResult]) -> DraftAnswer:
        output = self._llm.complete(
            system=load_prompt(GENERATION_PROMPT),
            prompt=generation_prompt(question, context),
            output_type=GenerationOutput,
        )
        return to_draft(output, context)


# Excerpt-id markers removed from answers: "【2】", "(id 3)", "(excerpts 1, 2)".
_EXCERPT_MARKERS = re.compile(r"\s*(?:【[^】]*】|\((?:ids?|excerpts?)\s*[\d,\s]+\))", re.IGNORECASE)


def strip_excerpt_markers(text: str) -> str:
    """Remove excerpt-id markers; page references like "(p. 12)" are kept."""
    return _EXCERPT_MARKERS.sub("", text).strip()


def to_draft(output: GenerationOutput, context: Sequence[RetrievalResult]) -> DraftAnswer:
    """Map cited excerpt numbers to chunk ids, dropping numbers outside the context."""
    cited = tuple(
        context[number - 1].chunk.id
        for number in dict.fromkeys(output.cited_excerpts)  # de-duplicate, keep order
        if 1 <= number <= len(context)
    )
    text = strip_excerpt_markers(output.answer)
    return DraftAnswer(
        text=text, answerable=output.answerable and bool(text), cited_chunk_ids=cited
    )
