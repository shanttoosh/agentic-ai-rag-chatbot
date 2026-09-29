"""Groq adapter (OpenAI-compatible chat completions) with strict JSON-schema outputs."""

from __future__ import annotations

import logging
from typing import Literal

import groq
from pydantic import ValidationError

from app.core.config import Effort
from app.core.exceptions import LLMError
from app.domain.interfaces import OutputT

logger = logging.getLogger(__name__)

_MAX_COMPLETION_TOKENS = 4096  # includes the reasoning tokens of gpt-oss models

ReasoningEffort = Literal["low", "medium", "high"]


def reasoning_effort_for(model: str, effort: Effort) -> ReasoningEffort | None:
    """gpt-oss models accept low/medium/high reasoning effort; other models get none."""
    if not model.startswith("openai/gpt-oss"):
        return None
    if effort in ("low", "medium"):
        return effort
    return "high"  # high / xhigh / max


class GroqStructuredLLM:
    """Implements ``StructuredLLM`` with Groq chat completions.

    ``response_format`` is ``output_type``'s JSON schema in strict mode, temperature is 0,
    and the reply is parsed with ``output_type.model_validate_json``.
    """

    def __init__(self, client: groq.Groq, model: str, effort: Effort) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort_for(model, effort)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, *, system: str, prompt: str, output_type: type[OutputT]) -> OutputT:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_type.__name__,
                        "schema": output_type.model_json_schema(),
                        "strict": True,
                    },
                },
                reasoning_effort=self._reasoning_effort,
            )
        except groq.APIConnectionError as exc:
            raise LLMError("Could not reach the language model service") from exc
        except groq.RateLimitError as exc:
            raise LLMError("The language model service is rate limiting requests") from exc
        except (groq.AuthenticationError, groq.PermissionDeniedError) as exc:
            raise LLMError("The language model service rejected the credentials") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"The language model request failed (HTTP {exc.status_code})") from exc

        choice = response.choices[0]
        logger.info(
            "llm.completed",
            extra={
                "model": response.model,
                "schema": output_type.__name__,
                "finish_reason": choice.finish_reason,
                "input_tokens": response.usage.prompt_tokens if response.usage else None,
                "output_tokens": response.usage.completion_tokens if response.usage else None,
            },
        )
        if choice.finish_reason != "stop" or not choice.message.content:
            raise LLMError(
                f"The language model returned no complete answer "
                f"(finish_reason={choice.finish_reason})"
            )
        try:
            return output_type.model_validate_json(choice.message.content)
        except ValidationError as exc:
            raise LLMError("The language model returned output that failed validation") from exc
