"""Chunk models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrievable piece of a document, with the metadata stored alongside its vector."""

    id: str  # e.g. "page12_chunk03"
    text: str
    page_number: int
    chunk_index: int  # position within its page, starting at 0
    source: str
    document_name: str
    chapter: str | None = None
    section: str | None = None

    @property
    def heading(self) -> str:
        """Where the chunk sits in the book, e.g. "02 Anatomy ... > 2.3 Defining ..."."""
        return " > ".join(part for part in (self.chapter, self.section) if part)

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model: ``heading``, a blank line, then the text."""
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk paired with its embedding vector, ready to upsert."""

    chunk: DocumentChunk
    vector: Sequence[float]
