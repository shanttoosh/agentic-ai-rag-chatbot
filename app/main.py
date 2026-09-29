"""FastAPI entry point: ``uvicorn app.main:app --reload``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager, nullcontext

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.application.chat.service import ChatService
from app.bootstrap import open_chat_service
from app.core.config import load_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

ServiceFactory = Callable[[], AbstractContextManager[ChatService]]


def _service_from_environment() -> AbstractContextManager[ChatService]:
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)
    return open_chat_service(settings)


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    """Build the app. Pass ``chat_service`` to inject one (tests); otherwise it's wired
    from environment settings at startup and its clients are closed at shutdown."""
    factory: ServiceFactory = (
        (lambda: nullcontext(chat_service)) if chat_service else _service_from_environment
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.chat_service = None
        app.state.startup_error = None
        with ExitStack() as resources:
            try:
                app.state.chat_service = resources.enter_context(factory())
                logger.info("app.started")
            except ConfigurationError as exc:
                # /api/chat re-raises this error (HTTP 503); /health and /docs keep working.
                app.state.startup_error = exc
                logger.error("app.misconfigured", extra={"reason": exc.message})
            yield
        logger.info("app.stopped")

    app = FastAPI(
        title="Agentic AI RAG Chatbot",
        description="Answers questions strictly from the Agentic AI eBook "
        "(LangGraph + Pinecone + Groq LLM).",
        version="1.0.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


configure_logging()
app = create_app()
