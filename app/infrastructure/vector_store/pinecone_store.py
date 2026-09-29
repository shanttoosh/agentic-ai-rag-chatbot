"""Pinecone serverless vector store."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from pinecone import Index, NotFoundError, Pinecone, PineconeError, ServerlessSpec

from app.core.batching import batched
from app.core.exceptions import VectorStoreError
from app.domain.models import DocumentChunk, EmbeddedChunk, RetrievalResult

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 100


class PineconeVectorStore:
    """One Pinecone namespace holding the chunks of one document collection."""

    def __init__(
        self,
        client: Pinecone,
        index_name: str,
        namespace: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self._index_name = index_name
        self._namespace = namespace
        self._dimension = dimension
        self._spec = ServerlessSpec(cloud=cloud, region=region)
        self._index_handle: Index | None = None
        self._handle_lock = threading.Lock()  # guards lazy creation of the index handle

    # --------------------------------------------------------------- lifecycle

    def ensure_index(self) -> None:
        with self._errors("index setup"):
            if self._client.has_index(self._index_name):
                return
            logger.info("vector_store.creating_index", extra={"index": self._index_name})
            self._client.create_index(  # blocks until the index is ready
                name=self._index_name,
                dimension=self._dimension,
                metric="cosine",
                spec=self._spec,
            )

    def close(self) -> None:
        with self._handle_lock:
            if self._index_handle is not None:
                self._index_handle.close()
                self._index_handle = None

    # ------------------------------------------------------------------ writes

    def upsert(self, records: Sequence[EmbeddedChunk], fingerprint: str) -> None:
        vectors = [
            {
                "id": record.chunk.id,
                "values": list(record.vector),
                "metadata": _to_metadata(record.chunk, fingerprint),
            }
            for record in records
        ]
        with self._errors("upsert"):
            for batch in batched(vectors, _UPSERT_BATCH):
                self._index.upsert(vectors=batch, namespace=self._namespace, show_progress=False)

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self._errors("delete"):
            self._index.delete(ids=list(ids), namespace=self._namespace)

    # ------------------------------------------------------------------- reads

    def query(self, vector: Sequence[float], top_k: int) -> list[RetrievalResult]:
        with self._errors("query"):
            response = self._index.query(
                vector=list(vector),
                top_k=top_k,
                include_metadata=True,
                namespace=self._namespace,
            )
        results = [
            RetrievalResult(
                chunk=_from_metadata(match.id, match.metadata), score=float(match.score)
            )
            for match in response.matches
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)

    def count(self) -> int:
        with self._errors("stats"):
            stats = self._index.describe_index_stats()
        summary = stats.namespaces.get(self._namespace)
        return summary.vector_count if summary else 0

    def fingerprint_of(self, record_id: str) -> str | None:
        with self._errors("fetch"):
            response = self._index.fetch(ids=[record_id], namespace=self._namespace)
        vector = response.vectors.get(record_id)
        if vector is None or not vector.metadata:
            return None
        value = vector.metadata.get("fingerprint")
        return str(value) if value is not None else None

    def list_ids(self) -> Iterator[str]:
        with self._errors("list"):
            for page in self._index.list(namespace=self._namespace):
                for item in page.vectors:
                    if item.id is not None:
                        yield item.id

    # ---------------------------------------------------------------- internals

    @property
    def _index(self) -> Index:
        """Index handle, created on first use."""
        with self._handle_lock:
            if self._index_handle is None:
                with self._errors("connect"):
                    handle = self._client.Index(self._index_name)
                if not isinstance(handle, Index):
                    raise VectorStoreError("Unexpected Pinecone index client type")
                self._index_handle = handle
            return self._index_handle

    @contextmanager
    def _errors(self, action: str) -> Generator[None, None, None]:
        """Translate SDK errors into VectorStoreError with a client-safe message."""
        try:
            yield
        except NotFoundError as exc:
            raise VectorStoreError(
                f"Pinecone index '{self._index_name}' was not found. "
                "Run `python scripts/ingest.py` first."
            ) from exc
        except PineconeError as exc:
            raise VectorStoreError(f"Pinecone {action} failed") from exc


def _to_metadata(chunk: DocumentChunk, fingerprint: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "source": chunk.source,
        "document_name": chunk.document_name,
        "fingerprint": fingerprint,
    }
    # optional fields are omitted when unset (Pinecone rejects null metadata values)
    if chunk.chapter:
        metadata["chapter"] = chunk.chapter
    if chunk.section:
        metadata["section"] = chunk.section
    return metadata


def _from_metadata(record_id: str, metadata: Mapping[str, Any] | None) -> DocumentChunk:
    if not metadata or "text" not in metadata:
        raise VectorStoreError(
            "The index contains records without chunk metadata. "
            "Re-run `python scripts/ingest.py --force`."
        )
    return DocumentChunk(
        id=record_id,
        text=str(metadata["text"]),
        page_number=int(metadata.get("page_number", 0)),
        chunk_index=int(metadata.get("chunk_index", 0)),
        source=str(metadata.get("source", "")),
        document_name=str(metadata.get("document_name", "")),
        chapter=metadata.get("chapter"),
        section=metadata.get("section"),
    )
