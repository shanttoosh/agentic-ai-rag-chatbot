from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    error: str  # machine-readable code, e.g. "llm_error"
    message: str  # human-readable message (AppError.message)
