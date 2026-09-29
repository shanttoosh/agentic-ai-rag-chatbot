"""Grounding validation: is every claim in the draft answer backed by the context?"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.domain.interfaces import StructuredLLM
from app.domain.models import DraftAnswer, GroundingVerdict, RetrievalResult
from app.rag.generation.prompts import GROUNDING_PROMPT, grounding_prompt, load_prompt

logger = logging.getLogger(__name__)


class ValidationOutput(BaseModel):
    """JSON schema the LLM judge must fill in."""

    model_config = ConfigDict(extra="forbid")  # -> additionalProperties: false

    unsupported_claims: list[str] = Field(
        description="Claims in the answer that the excerpts don't support. Empty if none."
    )
    supported: bool = Field(description="True only if every claim in the answer is supported.")


def precheck(draft: DraftAnswer, context: Sequence[RetrievalResult]) -> GroundingVerdict | None:
    """Cheap deterministic checks; returns a failing verdict, or None to continue."""
    if not draft.text:
        return GroundingVerdict(supported=False, reason="empty answer")
    if not draft.cited_chunk_ids:
        return GroundingVerdict(supported=False, reason="answer cites no retrieved excerpt")
    context_ids = {result.chunk.id for result in context}
    if not set(draft.cited_chunk_ids) <= context_ids:
        return GroundingVerdict(supported=False, reason="answer cites excerpts outside the context")
    return None


class LLMGroundingValidator:
    """Runs ``precheck``, then an LLM judge over the excerpts the generator saw.

    The draft is supported only if the judge returns ``supported=True`` and an empty
    ``unsupported_claims`` list.
    """

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def validate(
        self, question: str, draft: DraftAnswer, context: Sequence[RetrievalResult]
    ) -> GroundingVerdict:
        failed = precheck(draft, context)
        if failed is not None:
            logger.info("grounding.precheck_failed", extra={"reason": failed.reason})
            return failed

        output = self._llm.complete(
            system=load_prompt(GROUNDING_PROMPT),
            prompt=grounding_prompt(question, draft.text, context),
            output_type=ValidationOutput,
        )
        claims = tuple(claim.strip() for claim in output.unsupported_claims if claim.strip())
        supported = output.supported and not claims
        verdict = GroundingVerdict(
            supported=supported,
            unsupported_claims=claims,
            reason="llm judge: supported" if supported else "llm judge: unsupported claims",
        )
        logger.info(
            "grounding.validated",
            extra={"supported": supported, "unsupported_claims": len(claims)},
        )
        return verdict
