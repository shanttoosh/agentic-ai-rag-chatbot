"""Claude (Anthropic API) adapter with structured outputs."""

from __future__ import annotations

import logging

import anthropic
from anthropic.types.beta import BetaOutputConfigParam
from pydantic import ValidationError

from app.core.config import Effort
from app.core.exceptions import LLMError, LLMRefusalError
from app.domain.interfaces import OutputT

logger = logging.getLogger(__name__)

# Server-side refusal fallback (fallbacks="default"); sent only for these models.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
_MAX_TOKENS = 16000


class ClaudeStructuredLLM:
    """Implements ``StructuredLLM`` with ``client.beta.messages.parse``.

    The output is constrained to ``output_type``'s JSON schema and parsed by the SDK;
    ``effort`` is sent as ``output_config.effort``.
    """

    def __init__(self, client: anthropic.Anthropic, model: str, effort: Effort) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._use_fallback = model in _FALLBACK_MODELS

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, *, system: str, prompt: str, output_type: type[OutputT]) -> OutputT:
        output_config: BetaOutputConfigParam = {"effort": self._effort}
        try:
            response = self._client.beta.messages.parse(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                output_format=output_type,
                betas=[_FALLBACK_BETA] if self._use_fallback else anthropic.omit,
                fallbacks="default" if self._use_fallback else anthropic.omit,
            )
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the language model service") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The language model service is rate limiting requests") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMError("The language model service rejected the credentials") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The language model request failed (HTTP {exc.status_code})") from exc
        except ValidationError as exc:
            raise LLMError("The language model returned output that failed validation") from exc

        logger.info(
            "llm.completed",
            extra={
                "model": response.model,
                "schema": output_type.__name__,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
        if response.stop_reason == "refusal":
            raise LLMRefusalError("The language model declined to answer")
        if response.parsed_output is None:
            raise LLMError(
                f"The language model returned no structured output "
                f"(stop_reason={response.stop_reason})"
            )
        return response.parsed_output
