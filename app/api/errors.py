"""Map application exceptions to JSON error responses.

The response body is ``{"error": <code>, "message": <AppError.message>}``; the exception
and its chained cause are logged.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError,
    ConfigurationError,
    DocumentError,
    EmbeddingError,
    InvalidQuestionError,
    LLMError,
    VectorStoreError,
)
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)

# Most specific first: the first matching class wins.
_ERROR_MAP: tuple[tuple[type[AppError], int, str], ...] = (
    (InvalidQuestionError, status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_question"),
    (ConfigurationError, status.HTTP_503_SERVICE_UNAVAILABLE, "configuration_error"),
    (VectorStoreError, status.HTTP_503_SERVICE_UNAVAILABLE, "vector_store_unavailable"),
    (EmbeddingError, status.HTTP_502_BAD_GATEWAY, "embedding_error"),
    (LLMError, status.HTTP_502_BAD_GATEWAY, "llm_error"),
    (DocumentError, status.HTTP_500_INTERNAL_SERVER_ERROR, "document_error"),
)


def _classify(exc: AppError) -> tuple[int, str]:
    for error_type, http_status, code in _ERROR_MAP:
        if isinstance(exc, error_type):
            return http_status, code
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"


def _json(http_status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=code, message=message).model_dump()
    return JSONResponse(status_code=http_status, content=body)


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AppError):  # registered for AppError, but typed as Exception
        return await unhandled_error_handler(request, exc)
    http_status, code = _classify(exc)
    log = logger.warning if http_status < 500 else logger.error
    log(
        "request.failed",
        extra={"path": request.url.path, "error": code, "status_code": http_status},
        exc_info=http_status >= 500,
    )
    return _json(http_status, code, exc.message)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("request.unhandled_error", extra={"path": request.url.path})
    return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "Internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
