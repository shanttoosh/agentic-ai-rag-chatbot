"""Evaluation dataset loading and metrics (pure functions, no I/O besides ``load_cases``)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from app.domain.models import Answer


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    answerable: bool
    expected_pages: tuple[int, ...] = ()
    sample: bool = False  # answer included in the report's "Sample answers"
    near_miss: bool = False


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    case: EvalCase
    answer: Answer
    latency_ms: int

    @property
    def retrieved_pages(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(r.chunk.page_number for r in self.answer.retrieved))

    @property
    def passed(self) -> bool:
        """Answerable -> grounded answer; unanswerable -> safe fallback."""
        return self.answer.grounded == self.case.answerable

    @property
    def page_hit(self) -> bool | None:
        """Did retrieval surface at least one expected page? None for unanswerable cases."""
        if not self.case.expected_pages:
            return None
        return bool(set(self.case.expected_pages) & set(self.retrieved_pages))


@dataclass(frozen=True, slots=True)
class EvalSummary:
    total: int
    answerable_accuracy: float  # share of answerable questions that got a grounded answer
    fallback_accuracy: float  # share of unanswerable questions that got the fallback
    page_hit_rate: float  # share of answerable questions whose expected page was retrieved
    mean_confidence_answered: float
    mean_latency_ms: float
    suggested_threshold: float | None


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=item["id"],
            question=item["question"],
            answerable=bool(item["answerable"]),
            expected_pages=tuple(item.get("expected_pages", [])),
            sample=bool(item.get("sample", False)),
            near_miss=bool(item.get("near_miss", False)),
        )
        for item in raw["cases"]
    ]


def _share(flags: Sequence[bool]) -> float:
    return round(sum(flags) / len(flags), 3) if flags else 0.0


def suggest_threshold(outcomes: Sequence[EvalOutcome]) -> float | None:
    """Midpoint between two top-score groups, or None if they overlap.

    Positives: answerable cases whose expected page was retrieved. Negatives: unanswerable
    cases that are not marked ``near_miss``.
    """
    positives = [
        o.answer.confidence.top_score for o in outcomes if o.case.answerable and o.page_hit
    ]
    negatives = [
        o.answer.confidence.top_score
        for o in outcomes
        if not o.case.answerable and not o.case.near_miss
    ]
    if not positives or not negatives or min(positives) <= max(negatives):
        return None
    return round((min(positives) + max(negatives)) / 2, 3)


def summarize(outcomes: Sequence[EvalOutcome]) -> EvalSummary:
    answerable = [o for o in outcomes if o.case.answerable]
    unanswerable = [o for o in outcomes if not o.case.answerable]
    answered = [o.answer.confidence.value for o in outcomes if o.answer.grounded]
    return EvalSummary(
        total=len(outcomes),
        answerable_accuracy=_share([o.passed for o in answerable]),
        fallback_accuracy=_share([o.passed for o in unanswerable]),
        page_hit_rate=_share([bool(o.page_hit) for o in answerable]),
        mean_confidence_answered=round(fmean(answered), 3) if answered else 0.0,
        mean_latency_ms=round(fmean(o.latency_ms for o in outcomes), 1) if outcomes else 0.0,
        suggested_threshold=suggest_threshold(outcomes),
    )


def _pages(pages: Sequence[int]) -> str:
    return ", ".join(map(str, pages)) or "-"


def to_markdown(outcomes: Sequence[EvalOutcome], summary: EvalSummary, threshold: float) -> str:
    suggested = (
        summary.suggested_threshold if summary.suggested_threshold is not None else "n/a (overlap)"
    )
    lines = [
        "# Evaluation results",
        "",
        f"- Questions: {summary.total}",
        f"- Answerable -> grounded answer: {summary.answerable_accuracy:.0%}",
        f"- Unanswerable -> safe fallback: {summary.fallback_accuracy:.0%}",
        f"- Expected page retrieved (answerable): {summary.page_hit_rate:.0%}",
        f"- Mean confidence of grounded answers: {summary.mean_confidence_answered}",
        f"- Mean latency: {summary.mean_latency_ms:.0f} ms",
        f"- Retrieval threshold used: {threshold}; suggested from this run: {suggested}",
        "",
        "| id | question | expected | status | pass | top score | confidence "
        "| retrieved pages | cited pages |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        expected = "answer" if o.case.answerable else "fallback"
        lines.append(
            f"| {o.case.id} | {o.case.question} | {expected} | {o.answer.status.value} | "
            f"{'yes' if o.passed else 'NO'} | {o.answer.confidence.top_score:.3f} | "
            f"{o.answer.confidence.value:.2f} | {', '.join(map(str, o.retrieved_pages))} | "
            f"{', '.join(map(str, o.answer.cited_pages)) or '-'} |"
        )
    lines += ["", "## Sample answers", ""]
    for o in outcomes:
        if not o.case.sample and o.case.answerable:
            continue
        lines += [
            f"### {o.case.question}",
            "",
            o.answer.text,
            "",
            f"*Status:* `{o.answer.status.value}` · "
            f"*confidence:* {o.answer.confidence.value:.2f} ({o.answer.confidence.level.value}) · "
            f"*top score:* {o.answer.confidence.top_score:.3f} · "
            f"*cited pages:* {_pages(o.answer.cited_pages)} · "
            f"*retrieved pages:* {_pages(o.retrieved_pages)}",
            "",
        ]
    return "\n".join(lines)
