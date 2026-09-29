from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.rag.grounding.confidence import ConfidencePolicy

EBOOK_PATH: Path = PROJECT_ROOT / "data" / "raw" / "Ebook-Agentic-AI.pdf"


@pytest.fixture
def policy() -> ConfidencePolicy:
    return ConfidencePolicy(threshold=0.30, ceiling=0.60)


@pytest.fixture
def ebook_path() -> Path:
    if not EBOOK_PATH.exists():
        pytest.skip("eBook PDF not downloaded (run scripts/download_document.py)")
    return EBOOK_PATH
