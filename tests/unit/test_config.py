"""Settings validation and missing-variable errors."""

from __future__ import annotations

import pytest

from app.core.config import Settings, load_settings
from app.core.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PINECONE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "VALIDATOR_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_THRESHOLD",
        "CONFIDENCE_SCORE_CEILING",
        "TOP_K",
        "LLM_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_are_valid() -> None:
    settings = _settings()
    assert settings.top_k == 5
    assert settings.pdf_path.is_absolute()
    assert settings.llm_provider == "groq"
    assert settings.effective_llm_model == "openai/gpt-oss-120b"
    assert settings.effective_validator_model == "openai/gpt-oss-20b"


def test_provider_picks_default_model() -> None:
    claude = _settings(llm_provider="anthropic")
    assert claude.effective_llm_model == "claude-opus-5-5"
    assert claude.effective_validator_model == "claude-opus-5-5"
    assert _settings(validator_model="custom").effective_validator_model == "custom"
    assert _settings(llm_model="openai/gpt-oss-20b").effective_llm_model == "openai/gpt-oss-20b"


def test_missing_groq_key_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        _settings().require_secret("groq")


def test_missing_pinecone_key_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="PINECONE_API_KEY"):
        _settings().require_secret("pinecone")


def test_keys_are_secret() -> None:
    settings = _settings(pinecone_api_key="pc-secret-value")
    assert "pc-secret-value" not in repr(settings)
    assert settings.require_secret("pinecone") == "pc-secret-value"


def test_invalid_values_become_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_OVERLAP", "5000")
    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
        load_settings()


def test_threshold_must_be_below_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.9")
    with pytest.raises(ConfigurationError, match="RETRIEVAL_THRESHOLD"):
        load_settings()


def test_rejects_unknown_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EFFORT", "extreme")
    with pytest.raises(ConfigurationError, match="LLM_EFFORT"):
        load_settings()
