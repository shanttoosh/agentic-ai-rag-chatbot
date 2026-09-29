"""POST /api/chat: request validation, response schema, error mapping."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.application.chat.service import ChatService
from app.core.constants import FALLBACK_ANSWER
from app.core.exceptions import LLMError, VectorStoreError
from app.domain.models import Answer, AnswerStatus, ConfidenceLevel, ConfidenceScore
from app.main import create_app
from tests.fakes import make_result

RETRIEVED = (
    make_result(0.52, "page18_chunk00", page=18),
    make_result(0.18, "page05_chunk00", page=5),
)


class StubAnswerer:
    def __init__(self, outcome: Answer | Exception) -> None:
        self.outcome = outcome

    def answer(self, question: str, top_k: int) -> Answer:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _grounded_answer() -> Answer:
    return Answer(
        question="What is Agentic AI?",
        text="Agentic AI refers to systems capable of autonomous decision-making (p. 18).",
        status=AnswerStatus.ANSWERED,
        confidence=ConfidenceScore(0.71, ConfidenceLevel.MEDIUM, 0.52, 0.35, 1, 2),
        retrieved=RETRIEVED,
        context_chunk_ids=frozenset({"page18_chunk00"}),
        cited_chunk_ids=("page18_chunk00",),
    )


@contextmanager
def _client(outcome: Answer | Exception) -> Generator[TestClient, None, None]:
    app = create_app(ChatService(StubAnswerer(outcome), default_top_k=5))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client() -> Iterator[TestClient]:
    with _client(_grounded_answer()) as test_client:
        yield test_client


def test_response_schema(client: TestClient) -> None:
    response = client.post("/api/chat", json={"question": "What is Agentic AI?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Agentic AI refers to")
    assert body["grounded"] is True
    assert body["status"] == "answered"
    assert body["confidence"] == 0.71
    assert body["confidence_details"]["level"] == "medium"
    assert body["cited_pages"] == [18]
    first, second = body["retrieved_context"]
    assert first == {
        "chunk_id": "page18_chunk00",
        "page": 18,
        "section": None,
        "score": 0.52,
        "above_threshold": True,
        "cited": True,
        "text": RETRIEVED[0].chunk.text,
    }
    assert second["above_threshold"] is False
    assert second["cited"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "    "},
        {},
        {"question": "q", "top_k": 0},
        {"question": "x" * 2001},
    ],
)
def test_invalid_requests_are_422(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/api/chat", json=payload).status_code == 422


def test_fallback_answer() -> None:
    fallback = Answer(
        question="What is the capital of France?",
        text=FALLBACK_ANSWER,
        status=AnswerStatus.LOW_RETRIEVAL_SCORE,
        confidence=ConfidenceScore.none(top_score=0.12, mean_score=0.1, retrieved=5),
        retrieved=(),
        context_chunk_ids=frozenset(),
    )
    with _client(fallback) as client:
        body = client.post("/api/chat", json={"question": "What is the capital of France?"}).json()
    assert body["answer"] == FALLBACK_ANSWER
    assert body["grounded"] is False
    assert body["confidence"] == 0.0
    assert body["status"] == "low_retrieval_score"


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (LLMError("The language model service rejected the credentials"), 502, "llm_error"),
        (VectorStoreError("Pinecone query failed"), 503, "vector_store_unavailable"),
    ],
)
def test_provider_errors_are_mapped(error: Exception, status_code: int, code: str) -> None:
    with _client(error) as client:
        response = client.post("/api/chat", json={"question": "What is Agentic AI?"})
    assert response.status_code == status_code
    assert response.json() == {"error": code, "message": str(error)}


def test_unexpected_errors_do_not_leak_details() -> None:
    with _client(RuntimeError("secret internal detail sk-123")) as client:
        response = client.post("/api/chat", json={"question": "What is Agentic AI?"})
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "message": "Internal server error"}


def test_missing_configuration_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "")
    with TestClient(create_app()) as client:
        response = client.post("/api/chat", json={"question": "What is Agentic AI?"})
    assert response.status_code == 503
    assert response.json()["error"] == "configuration_error"
    assert "PINECONE_API_KEY" in response.json()["message"]
