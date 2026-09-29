"""Constants shared across layers."""

from typing import Final

FALLBACK_ANSWER: Final = (
    "I couldn't find enough information about this in the provided Agentic AI eBook."
)

DOCUMENT_SOURCE: Final = "Agentic AI eBook"
DOCUMENT_NAME: Final = "Ebook-Agentic-AI.pdf"
DOCUMENT_URL: Final = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

# (start page, chapter title) from the table of contents on page 5. The chunker uses it
# to set each chunk's chapter and to reset the current section at chapter starts.
EBOOK_CHAPTERS: Final[tuple[tuple[int, str], ...]] = (
    (1, "Front matter"),
    (7, "01 Introduction to Agentic AI"),
    (17, "02 Anatomy of an Agentic AI System"),
    (29, "03 Multi-Agent Systems"),
    (37, "04 Orchestrating Agentic AI Systems"),
    (47, "05 Your Readiness for Agentic AI"),
    (54, "06 Practical Applications of Agentic AI"),
    (59, "About the Authors"),
)

# Part of the ingestion fingerprint: changing it forces re-ingestion.
INGESTION_VERSION: Final = "1"
