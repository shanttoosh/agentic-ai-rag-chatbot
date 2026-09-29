"""Typed state shared by the LangGraph nodes.

Each node returns a partial dict; LangGraph merges it into the state. Only ``question``
and ``top_k`` are present at the start; the rest are filled in as the graph runs.
"""

from __future__ import annotations

from typing import Required, TypedDict

from app.domain.models import (
    AnswerStatus,
    ConfidenceScore,
    DraftAnswer,
    GroundingVerdict,
    RetrievalResult,
)


class RAGState(TypedDict, total=False):
    question: Required[str]
    top_k: Required[int]
    retrieved: list[RetrievalResult]  # everything the vector store returned (top-K)
    context: list[RetrievalResult]  # the subset above the threshold, sent to the LLM
    confidence: ConfidenceScore
    draft: DraftAnswer
    verdict: GroundingVerdict
    answer: str
    status: AnswerStatus
