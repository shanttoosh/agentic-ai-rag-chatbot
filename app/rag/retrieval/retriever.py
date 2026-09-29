"""Semantic retriever: embed the question, search the vector store."""

from __future__ import annotations

import logging

from app.domain.interfaces import Embedder, VectorStore
from app.domain.models import RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        vector = self._embedder.embed_query(question)
        results = self._store.query(vector, top_k)
        logger.info(
            "retrieval.completed",
            extra={
                "top_k": top_k,
                "returned": len(results),
                "scores": [round(result.score, 4) for result in results],
            },
        )
        return results
