"""FastAPI dependencies. Services are created once in the lifespan and read from app state."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.application.chat.service import ChatService
from app.core.exceptions import ConfigurationError


def get_chat_service(request: Request) -> ChatService:
    service: ChatService | None = getattr(request.app.state, "chat_service", None)
    if service is None:
        startup_error: ConfigurationError | None = getattr(request.app.state, "startup_error", None)
        raise startup_error or ConfigurationError("The chat service is not initialised")
    return service


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
