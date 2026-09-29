from fastapi import APIRouter

from app.api.dependencies import ChatServiceDep
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import ErrorResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"description": "Invalid request (e.g. empty question)"},
        502: {"model": ErrorResponse, "description": "Embedding or LLM provider failed"},
        503: {
            "model": ErrorResponse,
            "description": "Missing configuration or vector DB unavailable",
        },
    },
)
def chat(payload: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """Answer a question from the Agentic AI eBook.

    Sync handler (``def``): FastAPI runs it in its thread pool.
    """
    answer = service.ask(payload.question, payload.top_k)
    return ChatResponse.from_answer(answer)
