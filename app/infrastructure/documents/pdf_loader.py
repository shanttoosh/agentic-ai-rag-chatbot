"""PDF loading with PyMuPDF."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pymupdf

from app.core.exceptions import DocumentParseError
from app.domain.models import Document, Page
from app.infrastructure.documents.cleaning import clean_blocks

logger = logging.getLogger(__name__)

_TEXT_BLOCK = 0  # PyMuPDF block type: 0 = text, 1 = image


class PyMuPDFLoader:
    """Extracts cleaned text blocks page by page.

    Blocks come from ``get_text("blocks", sort=True)`` (top-to-bottom, left-to-right);
    image blocks are skipped and each text block goes through ``clean_blocks``.
    """

    def __init__(self, source: str) -> None:
        self._source = source

    def load(self, path: Path) -> Document:
        if not path.is_file():
            raise DocumentParseError(f"PDF not found: {path.name}")
        data = path.read_bytes()
        pages = tuple(self._iter_pages(data, path.name))
        if all(page.is_empty for page in pages):
            raise DocumentParseError(f"No extractable text in {path.name}")
        logger.info("document.loaded", extra={"document": path.name, "pages": len(pages)})
        return Document(
            name=path.name,
            source=self._source,
            content_hash=hashlib.sha256(data).hexdigest(),
            pages=pages,
        )

    @staticmethod
    def _iter_pages(data: bytes, name: str) -> Iterator[Page]:
        try:
            # pymupdf.open is an alias of pymupdf.Document, whose __init__ is untyped.
            with pymupdf.open(stream=data, filetype="pdf") as pdf:  # type: ignore[no-untyped-call]
                for index, page in enumerate(pdf):
                    raw = [
                        block[4]
                        for block in page.get_text("blocks", sort=True)
                        if block[6] == _TEXT_BLOCK
                    ]
                    yield Page(number=index + 1, blocks=clean_blocks(raw))
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise DocumentParseError(f"Could not parse {name} as a PDF") from exc
