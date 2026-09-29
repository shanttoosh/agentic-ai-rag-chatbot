"""Pure functions that remove PDF extraction artifacts from text blocks."""

from __future__ import annotations

import re

# Running footer printed on every page of the eBook.
RUNNING_FOOTER = "AGENTIC AI FOR EXECUTIVES"

_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_BROKEN_HYPHEN = re.compile(r"(\w) -(\w)")  # "decision -making" -> "decision-making"
_REPLACEMENTS = {
    "�": "'",  # mis-encoded apostrophe ("AI�s")
    "բ": "•",  # bullet glyph extracted as the Armenian letter "բ"
    " ": " ",  # non-breaking space
}


def clean_block(text: str) -> str:
    """Normalise one extracted text block: fix glyphs, collapse whitespace, rejoin hyphens."""
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = " ".join(text.split())
    return _BROKEN_HYPHEN.sub(r"\1-\2", text)


def is_noise(block: str) -> bool:
    """True for blocks that carry no content: empty, running footer, bare page number."""
    return not block or block == RUNNING_FOOTER or bool(_PAGE_NUMBER.fullmatch(block))


def clean_blocks(raw_blocks: list[str]) -> tuple[str, ...]:
    cleaned = (clean_block(block) for block in raw_blocks)
    return tuple(block for block in cleaned if not is_noise(block))
