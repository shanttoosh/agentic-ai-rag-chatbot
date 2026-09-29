"""Ingestion: batching, idempotency via fingerprints, stale-vector cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.application.ingestion.chunker import ParagraphChunker
from app.application.ingestion.service import IngestionService
from app.core.exceptions import DocumentParseError
from app.domain.models import Document, Page
from tests.fakes import FakeEmbedder, InMemoryVectorStore


def _document(content_hash: str = "a" * 64, pages: int = 6) -> Document:
    return Document(
        name="book.pdf",
        source="Test Book",
        content_hash=content_hash,
        pages=tuple(
            Page(number=n, blocks=(f"Page {n} explains how autonomous agents plan and act.",))
            for n in range(1, pages + 1)
        ),
    )


class UnusedLoader:
    def load(self, path: Path) -> Document:
        raise AssertionError("loader should not be used when a Document is passed")


def _service(
    store: InMemoryVectorStore, embedder: FakeEmbedder, chunk_size: int = 500
) -> IngestionService:
    return IngestionService(
        UnusedLoader(), ParagraphChunker(chunk_size, 50), embedder, store, batch_size=4
    )


def test_first_run_embeds_and_upserts_in_batches() -> None:
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    report = _service(store, embedder).ingest(_document())

    assert not report.skipped
    assert report.chunks == 6 and store.count() == 6
    assert [len(batch) for batch in embedder.document_batches] == [4, 2]
    assert store.index_created
    stored_text = next(iter(store.records.values()))[0]
    assert stored_text.chunk.document_name == "book.pdf"


def test_second_run_is_skipped() -> None:
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    service = _service(store, embedder)
    first = service.ingest(_document())
    second = service.ingest(_document())

    assert second.skipped
    assert second.fingerprint == first.fingerprint
    assert len(embedder.document_batches) == 2  # nothing re-embedded


def test_force_reingests() -> None:
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    service = _service(store, embedder)
    service.ingest(_document())
    assert not service.ingest(_document(), force=True).skipped
    assert len(embedder.document_batches) == 4


def test_changed_document_is_reingested_and_stale_vectors_removed() -> None:
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    service = _service(store, embedder)
    service.ingest(_document(pages=6))
    report = service.ingest(_document(content_hash="b" * 64, pages=4))

    assert not report.skipped
    assert report.stale_deleted == 2
    assert sorted(store.records) == [f"page0{n}_chunk00" for n in range(1, 5)]


def test_chunking_config_is_part_of_fingerprint() -> None:
    embedder = FakeEmbedder()
    small = _service(InMemoryVectorStore(), embedder, chunk_size=300)
    large = _service(InMemoryVectorStore(), embedder, chunk_size=600)
    assert small.fingerprint(_document()) != large.fingerprint(_document())


def test_incomplete_previous_run_is_redone() -> None:
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    service = _service(store, embedder)
    service.ingest(_document())
    store.delete(["page03_chunk00"])  # simulate a partially failed earlier run
    assert not service.ingest(_document()).skipped
    assert store.count() == 6


def test_document_without_chunks_fails() -> None:
    empty = Document("empty.pdf", "Test", "c" * 64, (Page(1, ("ok",)),))
    with pytest.raises(DocumentParseError):
        _service(InMemoryVectorStore(), FakeEmbedder()).ingest(empty)
