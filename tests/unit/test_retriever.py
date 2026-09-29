"""Retrieval and retrieval filtering."""

from __future__ import annotations

import pytest

from app.domain.models import EmbeddedChunk
from app.rag.retrieval.filters import relevant_results, top_score
from app.rag.retrieval.retriever import VectorRetriever
from tests.fakes import FakeEmbedder, InMemoryVectorStore, make_chunk, make_result


def test_retriever_embeds_query_and_returns_best_first() -> None:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    chunks = [make_chunk(f"page0{i}_chunk00", f"text {i}", page=i) for i in range(1, 6)]
    store.upsert([EmbeddedChunk(chunk, embedder.embed_query(chunk.text)) for chunk in chunks], "fp")

    results = VectorRetriever(embedder, store).retrieve("text 3", top_k=3)

    assert embedder.queries[-1] == "text 3"
    assert len(results) == 3
    assert results[0].chunk.id == "page03_chunk00"  # identical text -> cosine 1.0
    assert results[0].score == pytest.approx(1.0)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_result_format_carries_chunk_metadata() -> None:
    result = make_result(0.42, "page12_chunk03", page=12)
    assert result.chunk.id == "page12_chunk03"
    assert result.chunk.page_number == 12
    assert result.chunk.source == "Agentic AI eBook"
    assert result.chunk.document_name == "Ebook-Agentic-AI.pdf"


def test_relevant_results_filters_and_sorts() -> None:
    results = [make_result(0.2, "a"), make_result(0.5, "b"), make_result(0.35, "c")]
    kept = relevant_results(results, threshold=0.3)
    assert [r.chunk.id for r in kept] == ["b", "c"]


def test_threshold_is_inclusive() -> None:
    assert len(relevant_results([make_result(0.3)], threshold=0.3)) == 1


def test_top_score() -> None:
    assert top_score([make_result(0.2), make_result(0.6)]) == 0.6
    assert top_score([]) == 0.0
