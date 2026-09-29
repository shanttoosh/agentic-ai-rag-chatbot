"""Score normalisation and the retrieval-confidence heuristic."""

from __future__ import annotations

import pytest

from app.domain.models import ConfidenceLevel
from app.rag.grounding.confidence import (
    ConfidencePolicy,
    compute_confidence,
    confidence_level,
    normalize_score,
)


class TestNormalizeScore:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(0.10, 0.0), (0.30, 0.0), (0.45, 0.5), (0.60, 1.0), (0.95, 1.0)],
    )
    def test_maps_floor_to_zero_and_ceiling_to_one(self, score: float, expected: float) -> None:
        assert normalize_score(score, 0.30, 0.60) == pytest.approx(expected)

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError):
            normalize_score(0.5, 0.6, 0.3)


class TestComputeConfidence:
    def test_no_scores(self, policy: ConfidencePolicy) -> None:
        score = compute_confidence([], policy)
        assert score.value == 0.0 and score.level is ConfidenceLevel.NONE

    def test_all_below_threshold_is_zero(self, policy: ConfidencePolicy) -> None:
        score = compute_confidence([0.25, 0.2, 0.1], policy)
        assert score.value == 0.0
        assert score.relevant_chunks == 0
        assert score.top_score == 0.25  # raw stats are still reported

    def test_strong_retrieval_is_high(self, policy: ConfidencePolicy) -> None:
        score = compute_confidence([0.70, 0.65, 0.62, 0.61, 0.60], policy)
        assert score.value == pytest.approx(1.0)
        assert score.level is ConfidenceLevel.HIGH
        assert score.relevant_chunks == 5

    def test_formula(self, policy: ConfidencePolicy) -> None:
        # top 0.45 -> 0.5; mean 0.35 -> 1/6; 2 relevant of target 3 -> 2/3
        score = compute_confidence([0.45, 0.35, 0.25], policy)
        expected = 0.5 * 0.5 + 0.3 * (0.05 / 0.30) + 0.2 * (2 / 3)
        assert score.value == pytest.approx(expected, abs=1e-3)
        assert score.relevant_chunks == 2
        assert score.retrieved_chunks == 3

    def test_raw_scores_are_rounded_on_fallback(self, policy: ConfidencePolicy) -> None:
        score = compute_confidence([0.0201124363, 0.0161, 0.0135], policy)
        assert (score.top_score, score.mean_score) == (0.0201, 0.0166)

    def test_is_monotonic_in_top_score(self, policy: ConfidencePolicy) -> None:
        values = [compute_confidence([top, 0.32], policy).value for top in (0.35, 0.45, 0.55)]
        assert values == sorted(values)

    def test_stays_in_unit_interval(self, policy: ConfidencePolicy) -> None:
        for scores in ([0.99] * 10, [0.31], [1.0, 0.0]):
            assert 0.0 <= compute_confidence(scores, policy).value <= 1.0


class TestPolicy:
    def test_levels(self, policy: ConfidencePolicy) -> None:
        assert confidence_level(0.8, policy) is ConfidenceLevel.HIGH
        assert confidence_level(0.6, policy) is ConfidenceLevel.MEDIUM
        assert confidence_level(0.2, policy) is ConfidenceLevel.LOW
        assert confidence_level(0.0, policy) is ConfidenceLevel.NONE

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            ConfidencePolicy(threshold=0.3, ceiling=0.6, top_weight=0.9)

    def test_threshold_below_ceiling(self) -> None:
        with pytest.raises(ValueError):
            ConfidencePolicy(threshold=0.7, ceiling=0.6)
