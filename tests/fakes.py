"""In-memory fakes for the domain interfaces."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.core.exceptions import LLMRefusalError
from app.domain.interfaces import OutputT
from app.domain.models import DocumentChunk, EmbeddedChunk, RetrievalResult


def make_chunk(
    chunk_id: str = "page01_chunk00",
    text: str = "Agentic AI refers to systems capable of autonomous decision-making.",
    page: int = 1,
    section: str | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        text=text,
        page_number=page,
        chunk_index=0,
        source="Agentic AI eBook",
        document_name="Ebook-Agentic-AI.pdf",
        chapter="01 Introduction to Agentic AI",
        section=section,
    )


def make_result(
    score: float, chunk_id: str = "page01_chunk00", page: int = 1, text: str | None = None
) -> RetrievalResult:
    chunk = (
        make_chunk(chunk_id=chunk_id, page=page)
        if text is None
        else make_chunk(chunk_id, text, page)
    )
    return RetrievalResult(chunk=chunk, score=score)


class FakeEmbedder:
    """Deterministic pseudo-embeddings derived from a hash of the text."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(byte - 127.5) / 127.5 for byte in digest[: self._dimension]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


@dataclass
class InMemoryVectorStore:
    """Cosine-similarity search over a dict; mirrors PineconeVectorStore's behaviour."""

    records: dict[str, tuple[EmbeddedChunk, str]] = field(default_factory=dict)
    upsert_calls: int = 0
    index_created: bool = False

    def ensure_index(self) -> None:
        self.index_created = True

    def upsert(self, records: Sequence[EmbeddedChunk], fingerprint: str) -> None:
        self.upsert_calls += 1
        for record in records:
            self.records[record.chunk.id] = (record, fingerprint)

    def query(self, vector: Sequence[float], top_k: int) -> list[RetrievalResult]:
        scored = [
            RetrievalResult(chunk=record.chunk, score=_cosine(vector, record.vector))
            for record, _ in self.records.values()
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]

    def count(self) -> int:
        return len(self.records)

    def fingerprint_of(self, record_id: str) -> str | None:
        entry = self.records.get(record_id)
        return entry[1] if entry else None

    def list_ids(self) -> Iterator[str]:
        yield from list(self.records)

    def delete(self, ids: Sequence[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class FakeLLM:
    """Returns queued outputs by schema type and records every call."""

    def __init__(self, *outputs: BaseModel | Exception) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def complete(self, *, system: str, prompt: str, output_type: type[OutputT]) -> OutputT:
        self.calls.append({"system": system, "prompt": prompt, "output_type": output_type})
        for index, output in enumerate(self._outputs):
            if isinstance(output, Exception | output_type):
                self._outputs.pop(index)
                if isinstance(output, Exception):
                    raise output
                return output
        raise AssertionError(f"FakeLLM has no queued {output_type.__name__}")


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append((question, top_k))
        return self._results[:top_k]


def refusal() -> LLMRefusalError:
    return LLMRefusalError("The language model declined to answer")
