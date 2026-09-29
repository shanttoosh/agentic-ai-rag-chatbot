from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns text into dense vectors; documents and queries use separate methods."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
