"""Prompt loading and formatting.

System prompts are read from ``app/prompts/*.txt``. User messages wrap each input in
XML-style tags, and all inserted text is HTML-escaped.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from html import escape
from importlib.resources import files

from app.domain.models import RetrievalResult

GENERATION_PROMPT = "system.txt"
GROUNDING_PROMPT = "grounding.txt"


@cache
def load_prompt(name: str) -> str:
    return files("app.prompts").joinpath(name).read_text(encoding="utf-8").strip()


def format_excerpts(context: Sequence[RetrievalResult]) -> str:
    """Format the context as ``<excerpt id page section>`` blocks, ids from 1."""
    parts = []
    for number, result in enumerate(context, start=1):
        chunk = result.chunk
        attributes = f'id="{number}" page="{chunk.page_number}"'
        if chunk.section:
            attributes += f' section="{escape(chunk.section)}"'
        parts.append(f"<excerpt {attributes}>\n{escape(chunk.text, quote=False)}\n</excerpt>")
    return "<excerpts>\n" + "\n".join(parts) + "\n</excerpts>"


def generation_prompt(question: str, context: Sequence[RetrievalResult]) -> str:
    return f"{format_excerpts(context)}\n\n<question>{escape(question, quote=False)}</question>"


def grounding_prompt(question: str, answer: str, context: Sequence[RetrievalResult]) -> str:
    return (
        f"{format_excerpts(context)}\n\n"
        f"<question>{escape(question, quote=False)}</question>\n\n"
        f"<draft_answer>{escape(answer, quote=False)}</draft_answer>"
    )
