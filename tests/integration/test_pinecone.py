"""Live checks against the real Pinecone index. Skipped unless explicitly enabled:

    RUN_LIVE_TESTS=1 pytest tests/integration/test_pinecone.py

Requires PINECONE_API_KEY and a completed `python scripts/ingest.py`.
"""

from __future__ import annotations

import os

import pytest

from app.bootstrap import open_retriever, open_vector_store
from app.core.config import Settings, load_settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1", reason="live test; set RUN_LIVE_TESTS=1 to run"
)


@pytest.fixture(scope="module")
def settings() -> Settings:
    settings = load_settings()
    if settings.pinecone_api_key is None:
        pytest.skip("PINECONE_API_KEY not set")
    return settings


def test_index_holds_the_ebook(settings: Settings) -> None:
    with open_vector_store(settings) as store:
        assert store.count() > 50


def test_query_returns_scored_chunks_with_metadata(settings: Settings) -> None:
    with open_retriever(settings) as retriever:
        results = retriever.retrieve("What is Agentic AI?", top_k=5)

    assert len(results) == 5
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    top = results[0].chunk
    assert top.id.startswith("page") and top.page_number > 0 and top.text
    assert top.document_name == "Ebook-Agentic-AI.pdf"
    assert {8, 18} & {r.chunk.page_number for r in results}
