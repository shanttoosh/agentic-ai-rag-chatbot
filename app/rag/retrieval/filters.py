"""Pure retrieval-quality filters."""

from collections.abc import Sequence

from app.domain.models import RetrievalResult


def relevant_results(results: Sequence[RetrievalResult], threshold: float) -> list[RetrievalResult]:
    """Results whose similarity clears the threshold, best first."""
    kept = [result for result in results if result.score >= threshold]
    return sorted(kept, key=lambda result: result.score, reverse=True)


def top_score(results: Sequence[RetrievalResult]) -> float:
    return max((result.score for result in results), default=0.0)
