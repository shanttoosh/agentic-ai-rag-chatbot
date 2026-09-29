"""Ingestion use case: document -> chunks -> embeddings -> vector store."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.batching import batched
from app.core.constants import INGESTION_VERSION
from app.core.exceptions import DocumentParseError
from app.domain.interfaces import Chunker, DocumentLoader, Embedder, VectorStore
from app.domain.models import Document, DocumentChunk, EmbeddedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    document_name: str
    pages: int
    chunks: int
    fingerprint: str
    skipped: bool  # True if the index already held this fingerprint and nothing was written
    stale_deleted: int = 0


class IngestionService:
    """Chunks, embeds and upserts a document into the vector store.

    Every vector is tagged with a fingerprint of (PDF hash, chunker signature, embedding
    model, dimension, ``INGESTION_VERSION``). If the stored vectors already carry the
    current fingerprint, ``ingest`` returns without embedding anything.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        batch_size: int = 64,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def ingest_file(self, path: Path, *, force: bool = False) -> IngestionReport:
        return self.ingest(self._loader.load(path), force=force)

    def ingest(self, document: Document, *, force: bool = False) -> IngestionReport:
        fingerprint = self.fingerprint(document)
        # chunks are held in memory; embeddings are computed and upserted per batch
        chunks = list(self._chunker.chunk(document))
        if not chunks:
            raise DocumentParseError(f"{document.name} produced no text chunks")

        self._store.ensure_index()
        if not force and self._is_current(chunks, fingerprint):
            logger.info(
                "ingestion.skipped",
                extra={"document": document.name, "fingerprint": fingerprint},
            )
            return IngestionReport(
                document.name, document.page_count, len(chunks), fingerprint, True
            )

        self._index_chunks(chunks, fingerprint)
        stale = self._delete_stale({chunk.id for chunk in chunks})
        logger.info(
            "ingestion.completed",
            extra={
                "document": document.name,
                "chunks": len(chunks),
                "stale_deleted": stale,
                "fingerprint": fingerprint,
            },
        )
        return IngestionReport(
            document.name, document.page_count, len(chunks), fingerprint, False, stale
        )

    def fingerprint(self, document: Document) -> str:
        parts = (
            document.content_hash,
            self._chunker.signature,
            self._embedder.model_name,
            str(self._embedder.dimension),
            INGESTION_VERSION,
        )
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _is_current(self, chunks: list[DocumentChunk], fingerprint: str) -> bool:
        # current = the last-upserted chunk has this fingerprint and the counts match
        return self._store.fingerprint_of(
            chunks[-1].id
        ) == fingerprint and self._store.count() == len(chunks)

    def _index_chunks(self, chunks: list[DocumentChunk], fingerprint: str) -> None:
        for number, batch in enumerate(batched(chunks, self._batch_size), start=1):
            vectors = self._embedder.embed_documents([chunk.embedding_text for chunk in batch])
            records = [
                EmbeddedChunk(chunk=chunk, vector=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self._store.upsert(records, fingerprint)
            logger.info("ingestion.batch_upserted", extra={"batch": number, "size": len(batch)})

    def _delete_stale(self, current_ids: set[str]) -> int:
        """Remove vectors from earlier ingestions whose ids no longer exist."""
        stale = [record_id for record_id in self._store.list_ids() if record_id not in current_ids]
        for batch in batched(stale, 1000):
            self._store.delete(batch)
        return len(stale)
