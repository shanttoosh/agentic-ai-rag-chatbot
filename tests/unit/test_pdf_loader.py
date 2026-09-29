"""PDF extraction and cleaning."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import DocumentDownloadError, DocumentParseError
from app.infrastructure.documents.cleaning import (
    RUNNING_FOOTER,
    clean_block,
    clean_blocks,
    is_noise,
)
from app.infrastructure.documents.downloader import download_pdf
from app.infrastructure.documents.pdf_loader import PyMuPDFLoader


class TestCleaning:
    def test_collapses_whitespace(self) -> None:
        assert clean_block("  Agentic\n  AI\t systems  ") == "Agentic AI systems"

    def test_fixes_bullet_glyph_and_apostrophe(self) -> None:
        assert clean_block("բ Konverge AI�s expertise") == "• Konverge AI's expertise"

    def test_rejoins_broken_hyphenation(self) -> None:
        assert (
            clean_block("Decision -Making and decision -making")
            == "Decision-Making and decision-making"
        )

    def test_keeps_spaced_dashes(self) -> None:
        assert clean_block("Agentic AI - an advanced form") == "Agentic AI - an advanced form"

    @pytest.mark.parametrize("block", ["", RUNNING_FOOTER, "07", "123"])
    def test_noise_blocks(self, block: str) -> None:
        assert is_noise(block)

    def test_content_is_not_noise(self) -> None:
        assert not is_noise("2.1 The Core Pillars")

    def test_clean_blocks_drops_noise(self) -> None:
        assert clean_blocks([RUNNING_FOOTER, " Hello \n world ", "42"]) == ("Hello world",)


def _write_pdf(path: Path, pages: list[list[str]]) -> Path:
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line)
            y += 40
    doc.save(path)
    doc.close()
    return path


class TestPyMuPDFLoader:
    def test_extracts_pages_in_order_and_removes_noise(self, tmp_path: Path) -> None:
        pdf = _write_pdf(
            tmp_path / "book.pdf",
            [
                [RUNNING_FOOTER, "1.1 First Section", "Agents act autonomously.", "04"],
                ["Second page text."],
            ],
        )
        document = PyMuPDFLoader("Test Book").load(pdf)

        assert document.name == "book.pdf"
        assert document.source == "Test Book"
        assert [page.number for page in document.pages] == [1, 2]
        assert document.pages[0].blocks == ("1.1 First Section", "Agents act autonomously.")
        assert document.pages[1].text == "Second page text."
        assert len(document.content_hash) == 64

    def test_same_file_same_hash(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "book.pdf", [["Stable content."]])
        loader = PyMuPDFLoader("Test Book")
        assert loader.load(pdf).content_hash == loader.load(pdf).content_hash

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentParseError, match="not found"):
            PyMuPDFLoader("Test Book").load(tmp_path / "missing.pdf")

    def test_not_a_pdf(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.pdf"
        bogus.write_text("this is not a pdf")
        with pytest.raises(DocumentParseError):
            PyMuPDFLoader("Test Book").load(bogus)

    def test_pdf_without_text(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path / "blank.pdf", [[]])
        with pytest.raises(DocumentParseError, match="No extractable text"):
            PyMuPDFLoader("Test Book").load(pdf)

    def test_real_ebook(self, ebook_path: Path) -> None:
        document = PyMuPDFLoader("Agentic AI eBook").load(ebook_path)
        assert document.page_count == 60
        assert "autonomous decision-making" in document.pages[17].text  # page 18
        full_text = "\n".join(page.text for page in document.pages)
        assert RUNNING_FOOTER not in full_text
        assert "բ" not in full_text and "�" not in full_text


class TestDownloader:
    def test_skips_existing_file(self, tmp_path: Path) -> None:
        existing = tmp_path / "book.pdf"
        existing.write_bytes(b"%PDF-1.7 already here")
        assert download_pdf("https://invalid.example/book.pdf", existing) == existing

    def test_downloads_pdf(self, tmp_path: Path) -> None:
        source = _write_pdf(tmp_path / "source.pdf", [["Hello"]])
        target = tmp_path / "raw" / "book.pdf"
        download_pdf(source.as_uri(), target)
        assert target.read_bytes() == source.read_bytes()

    def test_rejects_non_pdf(self, tmp_path: Path) -> None:
        source = tmp_path / "page.html"
        source.write_text("<html>not found</html>")
        target = tmp_path / "book.pdf"
        with pytest.raises(DocumentDownloadError, match="not a PDF"):
            download_pdf(source.as_uri(), target)
        assert not target.exists()

    def test_unreachable_url(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentDownloadError):
            download_pdf((tmp_path / "missing.pdf").as_uri(), tmp_path / "book.pdf")
