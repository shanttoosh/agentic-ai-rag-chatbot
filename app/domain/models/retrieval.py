"""Retrieval result model."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.models.chunk import DocumentChunk


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A chunk returned by semantic search.

    ``score`` is the cosine similarity between the query and chunk embeddings, as
    returned by the vector store.
    """

    chunk: DocumentChunk
    score: float
