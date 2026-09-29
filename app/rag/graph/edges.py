"""Conditional routing between graph nodes (pure functions of the state)."""

from __future__ import annotations

from typing import Final, Literal

from app.domain.models import AnswerStatus
from app.rag.graph.state import RAGState

RETRIEVE: Final = "retrieve"
CHECK_RETRIEVAL: Final = "check_retrieval"
GENERATE: Final = "generate"
VALIDATE: Final = "validate"
FINALIZE: Final = "finalize"
FALLBACK: Final = "fallback"


def after_check_retrieval(state: RAGState) -> Literal["generate", "fallback"]:
    """``generate`` if at least one chunk passed the threshold, else ``fallback``."""
    return GENERATE if state.get("context") else FALLBACK


def after_generate(state: RAGState) -> Literal["validate", "fallback"]:
    draft = state.get("draft")
    if state.get("status") is AnswerStatus.DECLINED or draft is None or not draft.answerable:
        return FALLBACK
    return VALIDATE


def after_validate(state: RAGState) -> Literal["finalize", "fallback"]:
    verdict = state.get("verdict")
    return FINALIZE if verdict is not None and verdict.supported else FALLBACK


def fallback_status(state: RAGState) -> AnswerStatus:
    """Fallback status, derived from which state keys are set."""
    if state.get("status") is AnswerStatus.DECLINED:
        return AnswerStatus.DECLINED
    if not state.get("context"):
        return AnswerStatus.LOW_RETRIEVAL_SCORE
    draft = state.get("draft")
    if draft is None or not draft.answerable:
        return AnswerStatus.NOT_IN_CONTEXT
    return AnswerStatus.FAILED_VALIDATION
