"""Chat application service."""

from __future__ import annotations

import pytest

from app.application.chat.service import ChatService, normalize_question
from app.core.exceptions import InvalidQuestionError
from app.domain.models import Answer, AnswerStatus, ConfidenceScore


class RecordingAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def answer(self, question: str, top_k: int) -> Answer:
        self.calls.append((question, top_k))
        return Answer(
            question=question,
            text="fallback",
            status=AnswerStatus.LOW_RETRIEVAL_SCORE,
            confidence=ConfidenceScore.none(),
            retrieved=(),
            context_chunk_ids=frozenset(),
        )


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_questions_are_rejected(question: str) -> None:
    with pytest.raises(InvalidQuestionError):
        normalize_question(question)


def test_normalizes_whitespace() -> None:
    assert normalize_question("  What   is\nAgentic AI? ") == "What is Agentic AI?"


def test_rejects_oversized_question() -> None:
    with pytest.raises(InvalidQuestionError, match="at most"):
        normalize_question("x" * 2001)


def test_uses_default_top_k() -> None:
    answerer = RecordingAnswerer()
    ChatService(answerer, default_top_k=5).ask(" What is Agentic AI? ")
    assert answerer.calls == [("What is Agentic AI?", 5)]


def test_explicit_top_k() -> None:
    answerer = RecordingAnswerer()
    ChatService(answerer, default_top_k=5).ask("q", top_k=8)
    assert answerer.calls == [("q", 8)]


@pytest.mark.parametrize("top_k", [0, 21])
def test_rejects_out_of_range_top_k(top_k: int) -> None:
    with pytest.raises(InvalidQuestionError, match="top_k"):
        ChatService(RecordingAnswerer(), default_top_k=5).ask("q", top_k=top_k)


def test_empty_question_never_reaches_pipeline() -> None:
    answerer = RecordingAnswerer()
    with pytest.raises(InvalidQuestionError):
        ChatService(answerer, default_top_k=5).ask("  ")
    assert answerer.calls == []
