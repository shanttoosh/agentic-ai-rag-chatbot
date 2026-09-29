"""Source document models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Page:
    """One PDF page as a sequence of cleaned text blocks, in reading order."""

    number: int  # 1-based
    blocks: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded source document."""

    name: str  # file name, e.g. "Ebook-Agentic-AI.pdf"
    source: str  # human-readable source label, e.g. "Agentic AI eBook"
    content_hash: str  # sha256 of the file bytes
    pages: tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)
