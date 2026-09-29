"""Embeddings via Pinecone Inference (hosted models, same API key as the index)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pinecone import DenseEmbedding, Pinecone, PineconeError

from app.core.batching import batched
from app.core.exceptions import EmbeddingError

InputType = Literal["passage", "query"]


class PineconeEmbedder:
    """Embeds text with a Pinecone-hosted model such as ``llama-text-embed-v2``.

    Documents are sent with ``input_type="passage"``, queries with ``input_type="query"``.
    """

    def __init__(
        self,
        client: Pinecone,
        model: str,
        dimension: int,
        batch_size: int = 64,  # the API accepts at most 96 inputs per request
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(texts, self._batch_size):
            vectors.extend(self._embed(batch, "passage"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def _embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        try:
            result = self._client.inference.embed(
                model=self._model,
                inputs=texts,
                parameters={
                    "input_type": input_type,
                    "truncate": "END",
                    "dimension": self._dimension,
                },
            )
        except PineconeError as exc:
            raise EmbeddingError("The embedding service request failed") from exc

        vectors = [list(item.values) for item in result.data if isinstance(item, DenseEmbedding)]
        if len(vectors) != len(texts) or any(len(v) != self._dimension for v in vectors):
            raise EmbeddingError("The embedding service returned unexpected vectors")
        return vectors
