"""Application exception hierarchy.

``AppError.message`` is returned to API clients. Provider SDK errors are attached as
``__cause__`` (``raise ... from exc``) and appear only in the logs.
"""


class AppError(Exception):
    """Base class for all application errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(AppError):
    """A required setting is missing or invalid."""


class InvalidQuestionError(AppError):
    """The question is empty or otherwise unusable."""


class DocumentError(AppError):
    """Base class for problems with the source document."""


class DocumentDownloadError(DocumentError):
    """The PDF could not be downloaded."""


class DocumentParseError(DocumentError):
    """The PDF could not be opened or has no extractable text."""


class EmbeddingError(AppError):
    """The embedding provider failed."""


class VectorStoreError(AppError):
    """The vector database failed or is not ready."""


class LLMError(AppError):
    """The language model call failed."""


class LLMRefusalError(LLMError):
    """The language model declined to respond."""
