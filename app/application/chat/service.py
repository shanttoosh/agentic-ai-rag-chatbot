"""Chat use case: validate the question, run the pipeline, log the outcome."""

from __future__ import annotations

import logging
import time

from app.core.exceptions import InvalidQuestionError
from app.domain.interfaces import QuestionAnswerer
from app.domain.models import Answer

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000


def normalize_question(question: str) -> str:
    """Collapse whitespace; reject empty or oversized questions."""
    normalized = " ".join(question.split())
    if not normalized:
        raise InvalidQuestionError("The question must not be empty")
    if len(normalized) > MAX_QUESTION_CHARS:
        raise InvalidQuestionError(f"The question must be at most {MAX_QUESTION_CHARS} characters")
    return normalized


class ChatService:
    def __init__(self, answerer: QuestionAnswerer, default_top_k: int, max_top_k: int = 20) -> None:
        if not 1 <= default_top_k <= max_top_k:
            raise ValueError("default_top_k must be between 1 and max_top_k")
        self._answerer = answerer
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        normalized = normalize_question(question)
        k = self._default_top_k if top_k is None else top_k
        if not 1 <= k <= self._max_top_k:
            raise InvalidQuestionError(f"top_k must be between 1 and {self._max_top_k}")

        started = time.perf_counter()
        answer = self._answerer.answer(normalized, k)
        logger.info(
            "chat.answered",
            extra={
                "status": answer.status.value,
                "grounded": answer.grounded,
                "confidence": answer.confidence.value,
                "top_score": answer.confidence.top_score,
                "top_k": k,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return answer
