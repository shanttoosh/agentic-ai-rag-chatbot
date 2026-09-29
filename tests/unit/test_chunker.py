"""Structure-aware chunking."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from app.application.ingestion.chunker import ParagraphChunker
from app.core.constants import EBOOK_CHAPTERS
from app.domain.models import Document, Page
from app.infrastructure.documents.pdf_loader import PyMuPDFLoader


def _document(*pages: tuple[str, ...]) -> Document:
    return Document(
        name="book.pdf",
        source="Test Book",
        content_hash="0" * 64,
        pages=tuple(Page(number=i, blocks=blocks) for i, blocks in enumerate(pages, start=1)),
    )


def _sentences(count: int, prefix: str = "Sentence") -> str:
    return " ".join(f"{prefix} number {i} talks about autonomous agents." for i in range(count))


class TestChunkIdsAndMetadata:
    def test_ids_and_page_numbers(self) -> None:
        document = _document(
            ("First page content about agents.",), ("Second page content about tools.",)
        )
        chunks = list(ParagraphChunker(200, 20).chunk(document))

        assert [c.id for c in chunks] == ["page01_chunk00", "page02_chunk00"]
        assert [c.page_number for c in chunks] == [1, 2]
        assert all(c.source == "Test Book" and c.document_name == "book.pdf" for c in chunks)

    def test_chunk_index_counts_within_page(self) -> None:
        document = _document((_sentences(12),))
        chunks = list(ParagraphChunker(200, 40).chunk(document))
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_signature_changes_with_config(self) -> None:
        assert ParagraphChunker(500, 50).signature != ParagraphChunker(600, 50).signature


class TestSizesAndOverlap:
    def test_chunks_respect_size(self) -> None:
        chunks = list(ParagraphChunker(200, 40).chunk(_document((_sentences(20),))))
        assert all(len(c.text) <= 200 for c in chunks)

    def test_consecutive_chunks_overlap(self) -> None:
        chunks = list(ParagraphChunker(200, 60).chunk(_document((_sentences(20),))))
        for previous, current in pairwise(chunks):
            last_sentence = previous.text.rsplit(". ", 1)[-1]
            assert current.text.startswith(last_sentence)

    def test_no_overlap_when_disabled(self) -> None:
        chunks = list(ParagraphChunker(200, 0).chunk(_document((_sentences(20),))))
        joined = " ".join(c.text for c in chunks)
        assert joined.count("number 5 ") == 1

    def test_long_sentence_is_split_on_words(self) -> None:
        long_sentence = "word " * 150  # 750 chars, no sentence boundary
        chunks = list(ParagraphChunker(200, 20).chunk(_document((long_sentence.strip(),))))
        assert len(chunks) >= 4
        assert all(len(c.text) <= 200 for c in chunks)

    def test_tiny_noise_chunks_are_dropped(self) -> None:
        chunks = list(
            ParagraphChunker(200, 20).chunk(
                _document(("ok",), ("Real content about agentic systems.",))
            )
        )
        assert [c.page_number for c in chunks] == [2]

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
    def test_invalid_config(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            ParagraphChunker(size, overlap)


class TestSections:
    def test_heading_starts_new_chunk_without_overlap(self) -> None:
        document = _document(
            (
                "1.1 Intro Section",
                "Intro text about agents.",
                "1.2 Next Section",
                "Next text about tools.",
            )
        )
        chunks = list(ParagraphChunker(1000, 200).chunk(document))

        assert [c.section for c in chunks] == ["1.1 Intro Section", "1.2 Next Section"]
        assert chunks[1].text.startswith("1.2 Next Section")
        assert "Intro text" not in chunks[1].text

    def test_section_carries_across_pages(self) -> None:
        document = _document(
            ("2.1 Pillars Section", "Perception text."), ("Reasoning text continues here.",)
        )
        chunks = list(ParagraphChunker(1000, 100).chunk(document))
        assert chunks[1].section == "2.1 Pillars Section"

    def test_heading_only_chunk_is_skipped_but_labels_next_page(self) -> None:
        document = _document(
            ("Body text on page one about agents.", "2.2 Building Blocks"),
            ("Environment, agent, goals and actions.",),
        )
        chunks = list(ParagraphChunker(1000, 100).chunk(document))
        assert [c.page_number for c in chunks] == [1, 2]
        assert chunks[1].section == "2.2 Building Blocks"
        assert all(c.text != "2.2 Building Blocks" for c in chunks)

    def test_chapter_start_resets_section(self) -> None:
        document = _document(
            ("4.5 Practical Steps", "Steps text here."), ("Readiness chapter text begins.",)
        )
        chunker = ParagraphChunker(
            1000, 100, chapters=[(1, "04 Orchestrating"), (2, "05 Readiness")]
        )
        chunks = list(chunker.chunk(document))
        assert (chunks[0].chapter, chunks[0].section) == ("04 Orchestrating", "4.5 Practical Steps")
        assert (chunks[1].chapter, chunks[1].section) == ("05 Readiness", None)

    def test_embedding_text_includes_heading(self) -> None:
        document = _document(
            ("1.3 Capabilities", "Autonomy operates independently."),
        )
        chunk = next(ParagraphChunker(1000, 100, chapters=[(1, "01 Intro")]).chunk(document))
        assert chunk.embedding_text.startswith("01 Intro > 1.3 Capabilities\n\n")


def test_real_ebook_chunks(ebook_path: Path) -> None:
    document = PyMuPDFLoader("Agentic AI eBook").load(ebook_path)
    chunks = list(ParagraphChunker(1000, 150, chapters=EBOOK_CHAPTERS).chunk(document))

    assert 80 <= len(chunks) <= 140
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert all(1 <= c.page_number <= 60 for c in chunks)
    by_id = {c.id: c for c in chunks}
    assert by_id["page21_chunk00"].section == "2.3 Defining Characteristics of an Agent"
    assert all(c.section is None for c in chunks if 47 <= c.page_number <= 58)
