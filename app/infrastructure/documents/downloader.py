"""Download the source PDF."""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from app.core.exceptions import DocumentDownloadError

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
_USER_AGENT = "agentic-ai-rag/1.0 (+https://github.com)"


def download_pdf(
    url: str, destination: Path, *, timeout: float = 60.0, force: bool = False
) -> Path:
    """Download ``url`` to ``destination`` unless it exists (or ``force`` is set).

    Writes to ``<destination>.part``, checks the ``%PDF-`` header, then renames the file
    into place; the partial file is removed on failure.
    """
    if destination.exists() and not force:
        logger.info("document.download_skipped", extra={"path": str(destination)})
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        with partial.open("rb") as check:
            if check.read(len(_PDF_MAGIC)) != _PDF_MAGIC:
                raise DocumentDownloadError(f"Downloaded file from {url} is not a PDF")
        partial.replace(destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DocumentDownloadError(f"Could not download the PDF from {url}") from exc
    finally:
        partial.unlink(missing_ok=True)

    logger.info(
        "document.downloaded",
        extra={"url": url, "path": str(destination), "bytes": destination.stat().st_size},
    )
    return destination
