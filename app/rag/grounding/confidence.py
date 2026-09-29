"""Retrieval-confidence heuristic, computed from the top-K cosine similarities::

norm(s)    = clip((s - threshold) / (ceiling - threshold), 0, 1)
top        = norm(max score)
mean       = norm(mean of top-K scores)
coverage   = min(relevant / target_relevant, 1)   # relevant: score >= threshold

confidence = 0.5 * top + 0.3 * mean + 0.2 * coverage   (0 when nothing is relevant)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from app.domain.models import ConfidenceLevel, ConfidenceScore


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    threshold: float
    ceiling: float
    top_weight: float = 0.5
    mean_weight: float = 0.3
    coverage_weight: float = 0.2
    target_relevant: int = 3
    high_cutoff: float = 0.75
    medium_cutoff: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.threshold < self.ceiling <= 1:
            raise ValueError("expected 0 <= threshold < ceiling <= 1")
        if abs(self.top_weight + self.mean_weight + self.coverage_weight - 1) > 1e-9:
            raise ValueError("confidence weights must sum to 1")
        if self.target_relevant < 1:
            raise ValueError("target_relevant must be at least 1")


def normalize_score(score: float, floor: float, ceiling: float) -> float:
    """Map ``floor -> 0`` and ``ceiling -> 1`` linearly, clipped to [0, 1]."""
    if ceiling <= floor:
        raise ValueError("ceiling must be greater than floor")
    return min(max((score - floor) / (ceiling - floor), 0.0), 1.0)


def confidence_level(value: float, policy: ConfidencePolicy) -> ConfidenceLevel:
    if value >= policy.high_cutoff:
        return ConfidenceLevel.HIGH
    if value >= policy.medium_cutoff:
        return ConfidenceLevel.MEDIUM
    if value > 0:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.NONE


def compute_confidence(scores: Sequence[float], policy: ConfidencePolicy) -> ConfidenceScore:
    if not scores:
        return ConfidenceScore.none()
    top, mean = round(max(scores), 4), round(fmean(scores), 4)
    relevant = sum(score >= policy.threshold for score in scores)
    if relevant == 0:
        return ConfidenceScore.none(top_score=top, mean_score=mean, retrieved=len(scores))

    value = (
        policy.top_weight * normalize_score(top, policy.threshold, policy.ceiling)
        + policy.mean_weight * normalize_score(mean, policy.threshold, policy.ceiling)
        + policy.coverage_weight * min(relevant / policy.target_relevant, 1.0)
    )
    value = round(value, 3)
    return ConfidenceScore(
        value=value,
        level=confidence_level(value, policy),
        top_score=top,
        mean_score=mean,
        relevant_chunks=relevant,
        retrieved_chunks=len(scores),
    )
