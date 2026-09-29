from pathlib import Path
from typing import Protocol

from app.domain.models import Document


class DocumentLoader(Protocol):
    """Reads a file into a :class:`Document` of cleaned pages."""

    def load(self, path: Path) -> Document: ...
