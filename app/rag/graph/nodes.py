"""Graph nodes. Each node does one step and returns only the state keys it changes."""

from __future__ import annotations

import logging
from typing import Any

from app.core.constants import FALLBACK_ANSWER
from app.core.exceptions import LLMRefusalError
from app.domain.interfaces import AnswerGenerator, GroundingValidator, Retriever
from app.domain.models import AnswerStatus, ConfidenceScore, GroundingVerdict
from app.rag.graph.edges import fallback_status
from app.rag.graph.state import RAGState
from app.rag.grounding.confidence import ConfidencePolicy, compute_confidence
from app.rag.retrieval.filters import relevant_results

logger = logging.getLogger(__name__)

StateUpdate = dict[str, Any]


class RAGNodes:
    """Holds the injected dependencies; every public method is one graph node."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        validator: GroundingValidator,
        policy: ConfidencePolicy,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._validator = validator
        self._policy = policy

    def retrieve(self, state: RAGState) -> StateUpdate:
        return {"retrieved": self._retriever.retrieve(state["question"], state["top_k"])}

    def check_retrieval(self, state: RAGState) -> StateUpdate:
        """Keep only chunks above the threshold and score the retrieval."""
        retrieved = state.get("retrieved", [])
        context = relevant_results(retrieved, self._policy.threshold)
        confidence = compute_confidence([r.score for r in retrieved], self._policy)
        logger.info(
            "retrieval.checked",
            extra={
                "relevant": len(context),
                "retrieved": len(retrieved),
                "top_score": confidence.top_score,
                "threshold": self._policy.threshold,
            },
        )
        return {"context": context, "confidence": confidence}

    def generate(self, state: RAGState) -> StateUpdate:
        try:
            draft = self._generator.generate(state["question"], state["context"])
        except LLMRefusalError:
            logger.warning("generation.declined")
            return {"status": AnswerStatus.DECLINED}
        return {"draft": draft}

    def validate(self, state: RAGState) -> StateUpdate:
        try:
            verdict = self._validator.validate(state["question"], state["draft"], state["context"])
        except LLMRefusalError:
            # a declined validation counts as unsupported
            logger.warning("validation.declined")
            verdict = GroundingVerdict(supported=False, reason="validator declined")
        return {"verdict": verdict}

    def finalize(self, state: RAGState) -> StateUpdate:
        return {"answer": state["draft"].text, "status": AnswerStatus.ANSWERED}

    def fallback(self, state: RAGState) -> StateUpdate:
        """Set the fallback answer and status; confidence 0, raw retrieval stats kept."""
        status = fallback_status(state)
        raw = state.get("confidence") or ConfidenceScore.none()
        logger.info("answer.fallback", extra={"status": status.value})
        return {
            "answer": FALLBACK_ANSWER,
            "status": status,
            "confidence": ConfidenceScore.none(
                top_score=raw.top_score, mean_score=raw.mean_score, retrieved=raw.retrieved_chunks
            ),
        }
