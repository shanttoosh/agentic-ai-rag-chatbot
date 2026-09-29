"""LLM adapters with fake SDK clients: request shape, parsing and error translation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from app.core.exceptions import LLMError, LLMRefusalError
from app.infrastructure.llm.claude import ClaudeStructuredLLM
from app.infrastructure.llm.groq_llm import GroqStructuredLLM, reasoning_effort_for
from app.rag.generation.generator import GenerationOutput
from app.rag.grounding.validator import ValidationOutput


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool


# ------------------------------------------------------------------------------ Groq


def _groq_client(
    content: str | None = '{"supported": true}',
    finish_reason: str = "stop",
    error: Exception | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if error is not None:
            raise error
        return SimpleNamespace(
            model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason, message=SimpleNamespace(content=content)
                )
            ],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_groq_sends_strict_schema_and_parses_output() -> None:
    client, calls = _groq_client()
    llm = GroqStructuredLLM(client, "openai/gpt-oss-120b", "medium")

    result = llm.complete(system="sys", prompt="user", output_type=Verdict)

    assert result == Verdict(supported=True)
    request = calls[0]
    assert request["temperature"] == 0
    assert request["reasoning_effort"] == "medium"
    assert request["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    schema_spec = request["response_format"]["json_schema"]
    assert request["response_format"]["type"] == "json_schema"
    assert schema_spec["strict"] is True
    assert schema_spec["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("output_type", [GenerationOutput, ValidationOutput])
def test_output_schemas_are_strict_compatible(output_type: type[BaseModel]) -> None:
    schema = output_type.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("openai/gpt-oss-120b", "low", "low"),
        ("openai/gpt-oss-20b", "max", "high"),
        ("qwen/qwen3.8-27b", "medium", None),
    ],
)
def test_reasoning_effort_mapping(model: str, effort: Any, expected: str | None) -> None:
    assert reasoning_effort_for(model, effort) == expected


def test_groq_truncated_output_is_an_error() -> None:
    client, _ = _groq_client(content='{"supp', finish_reason="length")
    with pytest.raises(LLMError, match="finish_reason=length"):
        GroqStructuredLLM(client, "openai/gpt-oss-20b", "low").complete(
            system="s", prompt="p", output_type=Verdict
        )


def test_groq_invalid_json_is_an_error() -> None:
    client, _ = _groq_client(content='{"unexpected": 1}')
    with pytest.raises(LLMError, match="failed validation"):
        GroqStructuredLLM(client, "openai/gpt-oss-20b", "low").complete(
            system="s", prompt="p", output_type=Verdict
        )


def _status_error(cls: type[groq.APIStatusError], code: int) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (_status_error(groq.RateLimitError, 429), "rate limiting"),
        (_status_error(groq.AuthenticationError, 401), "rejected the credentials"),
        (_status_error(groq.InternalServerError, 500), "HTTP 500"),
        (groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")), "reach"),
    ],
)
def test_groq_errors_are_translated(error: Exception, message: str) -> None:
    client, _ = _groq_client(error=error)
    with pytest.raises(LLMError, match=message) as caught:
        GroqStructuredLLM(client, "openai/gpt-oss-20b", "low").complete(
            system="s", prompt="p", output_type=Verdict
        )
    assert caught.value.__cause__ is error  # full provider error kept for the logs


# ---------------------------------------------------------------------------- Claude


def _claude_client(
    stop_reason: str = "end_turn", parsed: BaseModel | None = None
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def parse(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            parsed_output=parsed,
        )

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse))), calls


def test_claude_uses_structured_output_effort_and_fallbacks() -> None:
    client, calls = _claude_client(parsed=Verdict(supported=True))
    result = ClaudeStructuredLLM(client, "claude-opus-5-5", "medium").complete(
        system="sys", prompt="user", output_type=Verdict
    )
    assert result == Verdict(supported=True)
    request = calls[0]
    assert request["output_format"] is Verdict
    assert request["output_config"] == {"effort": "medium"}
    assert request["fallbacks"] == "default"


def test_claude_refusal_raises() -> None:
    client, _ = _claude_client(stop_reason="refusal")
    with pytest.raises(LLMRefusalError):
        ClaudeStructuredLLM(client, "claude-opus-5-5", "low").complete(
            system="s", prompt="p", output_type=Verdict
        )
