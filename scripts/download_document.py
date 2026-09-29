"""Download the Agentic AI eBook to data/raw/ (skipped if already present).

python scripts/download_document.py [--force]
"""

from __future__ import annotations

import argparse

from app.core.config import Settings
from app.infrastructure.documents.downloader import download_pdf
from scripts._cli import run


def main(settings: Settings) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()

    path = download_pdf(settings.pdf_url, settings.pdf_path, force=args.force)
    print(f"PDF ready: {path} ({path.stat().st_size / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    run(main)
