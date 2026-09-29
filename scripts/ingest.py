"""Ingest the eBook: download -> extract -> clean -> chunk -> embed -> upsert to Pinecone.

    python scripts/ingest.py            # skips embedding if the index is already current
    python scripts/ingest.py --force    # re-embed and overwrite

Also writes the chunks to data/processed/chunks.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from app.bootstrap import build_chunker, open_ingestion_service
from app.core.config import Settings
from app.core.constants import DOCUMENT_SOURCE
from app.domain.models import DocumentChunk
from app.infrastructure.documents.downloader import download_pdf
from app.infrastructure.documents.pdf_loader import PyMuPDFLoader
from scripts._cli import run


def export_chunks(chunks: Iterable[DocumentChunk], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            count += 1
    return count


def main(settings: Settings) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    args = parser.parse_args()

    pdf_path = download_pdf(settings.pdf_url, settings.pdf_path)
    document = PyMuPDFLoader(DOCUMENT_SOURCE).load(pdf_path)
    exported = export_chunks(
        build_chunker(settings).chunk(document), settings.processed_dir / "chunks.jsonl"
    )
    print(
        f"Extracted {document.page_count} pages -> {exported} chunks (data/processed/chunks.jsonl)"
    )

    with open_ingestion_service(settings) as service:
        report = service.ingest(document, force=args.force)

    if report.skipped:
        print(
            f"Index already up to date (fingerprint {report.fingerprint}); nothing to do. "
            "Use --force to re-embed."
        )
    else:
        print(
            f"Upserted {report.chunks} chunks into '{settings.pinecone_index_name}/"
            f"{settings.pinecone_namespace}' (fingerprint {report.fingerprint}); "
            f"removed {report.stale_deleted} stale vectors."
        )
    return 0


if __name__ == "__main__":
    run(main)
