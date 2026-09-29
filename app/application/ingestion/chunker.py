"""Chunking: pages -> sentence units -> overlapping chunks.

1. Chunks never cross a page boundary; each chunk has one page number.
2. A numbered section heading ("2.3 Defining Characteristics of an Agent") starts a new
   chunk, with no overlap from the previous one.
3. Sentences are packed greedily up to ``chunk_size`` characters. A full chunk's last
   sentences (up to ``chunk_overlap`` characters) are repeated at the start of the next.
4. Sentences longer than ``chunk_size`` are split on spaces.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from app.domain.models import Document, DocumentChunk, Page

SECTION_HEADING = re.compile(r"^\d{1,2}\.\d{1,2}\.?\s+[A-Z].{2,90}$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“•])")


@dataclass(frozen=True, slots=True)
class _Unit:
    """A sentence (or piece of a long sentence) with its block and section."""

    text: str
    starts_block: bool  # first unit of a PDF text block -> joined with "\n", else " "
    is_heading: bool
    section: str | None  # section heading in effect when this unit appears


class ParagraphChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        chapters: Sequence[tuple[int, str]] = (),
        min_alnum_chars: int = 20,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._chapters = sorted(chapters)
        self._chapter_starts = [start for start, _ in self._chapters]
        self._min_alnum = min_alnum_chars

    @property
    def signature(self) -> str:
        chapters = ",".join(str(start) for start in self._chapter_starts)
        return f"paragraph-v1|size={self._size}|overlap={self._overlap}|chapters={chapters}"

    def chunk(self, document: Document) -> Iterator[DocumentChunk]:
        section: str | None = None
        for page in document.pages:
            if page.number in self._chapter_starts:
                section = None  # reset the section at each chapter start
            units = list(self._units(page, section))
            if units:
                section = units[-1].section
            yield from self._page_chunks(document, page, units)

    # ------------------------------------------------------------------ internals

    def chapter_for_page(self, page_number: int) -> str | None:
        position = bisect.bisect_right(self._chapter_starts, page_number) - 1
        return self._chapters[position][1] if position >= 0 else None

    def _units(self, page: Page, section: str | None) -> Iterator[_Unit]:
        for block in page.blocks:
            is_heading = bool(SECTION_HEADING.match(block))
            if is_heading:
                section = block
            for index, sentence in enumerate(self._sentences(block)):
                yield _Unit(sentence, index == 0, is_heading, section)

    def _sentences(self, block: str) -> Iterator[str]:
        for sentence in _SENTENCE_BOUNDARY.split(block):
            if len(sentence) <= self._size:
                yield sentence
            else:
                yield from _split_words(sentence, self._size)

    def _page_chunks(
        self, document: Document, page: Page, units: list[_Unit]
    ) -> Iterator[DocumentChunk]:
        chapter = self.chapter_for_page(page.number)
        index = 0
        for group in self._pack(units):
            if all(unit.is_heading for unit in group):
                continue  # skip heading-only groups (a heading at the bottom of a page)
            text = _join(group)
            if sum(ch.isalnum() for ch in text) < self._min_alnum:
                continue
            yield DocumentChunk(
                id=f"page{page.number:02d}_chunk{index:02d}",
                text=text,
                page_number=page.number,
                chunk_index=index,
                source=document.source,
                document_name=document.name,
                chapter=chapter,
                # all units in a group share one section (no overlap across headings)
                section=group[0].section,
            )
            index += 1

    def _pack(self, units: list[_Unit]) -> Iterator[list[_Unit]]:
        current: list[_Unit] = []
        has_fresh = False  # does `current` hold anything beyond carried-over overlap?
        for unit in units:
            if unit.is_heading and not unit.starts_block:
                current.append(unit)  # continuation of a (long) heading block
                continue
            if unit.is_heading and has_fresh:
                yield current
                current, has_fresh = [], False
            elif has_fresh and _joined_length([*current, unit]) > self._size:
                yield current
                current = _tail(current, self._overlap)
                if _joined_length([*current, unit]) > self._size:
                    current = []
            elif unit.is_heading:
                current = []  # drop carried overlap: new section, new chunk
            current.append(unit)
            has_fresh = True
        if has_fresh:
            yield current


def _join(units: Sequence[_Unit]) -> str:
    parts = [units[0].text]
    for unit in units[1:]:
        parts.append(("\n" if unit.starts_block else " ") + unit.text)
    return "".join(parts)


def _joined_length(units: Sequence[_Unit]) -> int:
    return sum(len(unit.text) for unit in units) + max(len(units) - 1, 0)


def _tail(units: Sequence[_Unit], budget: int) -> list[_Unit]:
    """Longest suffix of ``units`` whose joined length fits in ``budget`` characters."""
    tail: list[_Unit] = []
    for unit in reversed(units):
        if _joined_length([unit, *tail]) > budget:
            break
        tail.insert(0, unit)
    return tail


def _split_words(text: str, size: int) -> Iterator[str]:
    """Split on spaces into pieces of at most ``size`` characters."""
    piece = ""
    for word in text.split():
        while len(word) > size:  # a single "word" longer than a chunk: hard-split it
            if piece:
                yield piece
                piece = ""
            yield word[:size]
            word = word[size:]
        if not word:
            continue
        candidate = f"{piece} {word}" if piece else word
        if len(candidate) > size:
            yield piece
            piece = word
        else:
            piece = candidate
    if piece:
        yield piece
