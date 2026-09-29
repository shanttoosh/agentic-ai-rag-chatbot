from collections.abc import Iterator
from typing import Protocol

from app.domain.models import Document, DocumentChunk


class Chunker(Protocol):
    """Splits a document into retrievable chunks."""

    @property
    def signature(self) -> str:
        """Identifies the chunking configuration; a change triggers re-ingestion."""
        ...

    def chunk(self, document: Document) -> Iterator[DocumentChunk]: ...
