from collections.abc import Iterator, Sequence
from typing import Protocol

from app.domain.models import EmbeddedChunk, RetrievalResult


class VectorStore(Protocol):
    """Stores chunk vectors for one document collection and searches them."""

    def ensure_index(self) -> None:
        """Create the index if it doesn't exist yet."""
        ...

    def upsert(self, records: Sequence[EmbeddedChunk], fingerprint: str) -> None:
        """Insert or overwrite records, tagging each with the ingestion fingerprint."""
        ...

    def query(self, vector: Sequence[float], top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` nearest chunks, best first."""
        ...

    def count(self) -> int:
        """Number of vectors in the collection."""
        ...

    def fingerprint_of(self, record_id: str) -> str | None:
        """Ingestion fingerprint stored on ``record_id``, or None if it doesn't exist."""
        ...

    def list_ids(self) -> Iterator[str]:
        """Every record id in the collection."""
        ...

    def delete(self, ids: Sequence[str]) -> None: ...
