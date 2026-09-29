"""Environment-based settings (``.env`` is read automatically)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import DOCUMENT_NAME, DOCUMENT_URL
from app.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Effort = Literal["low", "medium", "high", "xhigh", "max"]
LLMProvider = Literal["groq", "anthropic"]

DEFAULT_LLM_MODELS: dict[LLMProvider, str] = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-opus-5-5",
}
# Validator model per provider; providers not listed use the generation model.
DEFAULT_VALIDATOR_MODELS: dict[LLMProvider, str] = {"groq": "openai/gpt-oss-20b"}


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials (optional here; checked by `require_secret` where they're needed)
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None

    # --- Document
    pdf_url: str = DOCUMENT_URL
    pdf_path: Path = Path("data/raw") / DOCUMENT_NAME
    processed_dir: Path = Path("data/processed")

    # --- Chunking
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Embeddings (Pinecone Inference)
    embedding_model: str = "llama-text-embed-v2"
    embedding_dimension: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=64, ge=1, le=96)

    # --- Vector DB (Pinecone serverless)
    pinecone_index_name: str = "agentic-ai-ebook"
    pinecone_namespace: str = "agentic-ai-ebook"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Retrieval
    top_k: int = Field(default=5, ge=1, le=20)
    # Cosine-similarity bounds for llama-text-embed-v2 (values from scripts/evaluate.py).
    retrieval_threshold: float = Field(default=0.33, ge=0.0, le=1.0)
    confidence_score_ceiling: float = Field(default=0.70, gt=0.0, le=1.0)

    # --- LLM
    llm_provider: LLMProvider = "groq"
    llm_model: str | None = None  # defaults to DEFAULT_LLM_MODELS[llm_provider]
    llm_effort: Effort = "medium"
    validator_model: str | None = None  # defaults per provider, else the generation model
    validator_effort: Effort = "low"
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_threshold >= self.confidence_score_ceiling:
            raise ValueError("RETRIEVAL_THRESHOLD must be below CONFIDENCE_SCORE_CEILING")
        self.pdf_path = _absolute(self.pdf_path)
        self.processed_dir = _absolute(self.processed_dir)
        return self

    @property
    def effective_llm_model(self) -> str:
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def effective_validator_model(self) -> str:
        return (
            self.validator_model
            or DEFAULT_VALIDATOR_MODELS.get(self.llm_provider)
            or self.effective_llm_model
        )

    def require_secret(self, name: Literal["groq", "anthropic", "pinecone"]) -> str:
        """Return an API key, or raise ConfigurationError naming the missing variable."""
        secret: SecretStr | None = getattr(self, f"{name}_api_key")
        if secret is None or not secret.get_secret_value():
            raise ConfigurationError(
                f"Missing required environment variable: {name.upper()}_API_KEY"
            )
        return secret.get_secret_value()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    """Build settings from the environment, converting validation errors."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']).upper() or 'settings'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(f"Invalid configuration: {problems}") from exc
