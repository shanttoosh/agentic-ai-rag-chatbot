"""Streamlit UI rendered headlessly with AppTest; the backend is replaced by stubs."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.core.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "ui" / "streamlit_app.py")

RESPONSE: dict[str, Any] = {
    "answer": "Agentic AI refers to systems capable of autonomous decision-making (p. 18).",
    "grounded": True,
    "status": "answered",
    "confidence": 0.71,
    "confidence_details": {
        "level": "medium",
        "top_score": 0.52,
        "mean_score": 0.35,
        "relevant_chunks": 1,
        "retrieved_chunks": 2,
    },
    "cited_pages": [18],
    "retrieved_context": [
        {
            "chunk_id": "page18_chunk00",
            "page": 18,
            "section": None,
            "score": 0.52,
            "above_threshold": True,
            "cited": True,
            "text": "Agentic AI refers to systems capable of autonomous decision-making.",
        },
        {
            "chunk_id": "page05_chunk00",
            "page": 5,
            "section": None,
            "score": 0.18,
            "above_threshold": False,
            "cited": False,
            "text": "Table of Contents",
        },
    ],
}


@pytest.fixture
def api_up(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        requests.append({"url": url, "json": json})
        return httpx.Response(200, json=RESPONSE, request=httpx.Request("POST", url))

    def fake_get(url: str, timeout: float) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    return requests


def test_asking_renders_answer_confidence_and_context(api_up: list[dict[str, Any]]) -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input(key="question").input("What is Agentic AI?")
    app.main.button[0].click().run()  # the form's "Ask" button (sidebar has the samples)

    assert not app.exception
    assert api_up[0]["url"].endswith("/api/chat")
    assert api_up[0]["json"] == {"question": "What is Agentic AI?", "top_k": 5}
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Retrieval confidence"] == "0.71"
    assert metrics["Grounded"] == "✓ Yes"
    assert any(e.label.startswith("Retrieved context: 2 chunks") for e in app.expander)


def test_sample_question_button_asks_immediately(api_up: list[dict[str, Any]]) -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.button[0].click().run()

    assert not app.exception
    assert api_up[0]["json"]["question"] == "What is Agentic AI?"
    assert app.text_input(key="question").value == "What is Agentic AI?"


def test_unreachable_api_shows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx, "get", refuse)
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input(key="question").input("What is Agentic AI?")
    app.main.button[0].click().run()

    assert not app.exception
    assert any("Can't reach the API" in error.value for error in app.error)
