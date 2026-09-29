"""Run the evaluation set through the full pipeline and report the results.

    python scripts/evaluate.py

Writes evaluation/results/latest.json and latest.md. Requires ingestion to have run.
Each question makes up to two LLM calls (generate + validate).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from app.bootstrap import open_chat_service
from app.core.config import PROJECT_ROOT, Settings
from evaluation.metrics import EvalOutcome, load_cases, summarize, to_markdown
from scripts._cli import run

DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "rag_test_cases.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def main(settings: Settings) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--only", nargs="*", help="case ids to run (default: all)")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.only:
        cases = [case for case in cases if case.id in set(args.only)]

    outcomes: list[EvalOutcome] = []
    with open_chat_service(settings) as service:
        for case in cases:
            started = time.perf_counter()
            answer = service.ask(case.question)
            latency = round((time.perf_counter() - started) * 1000)
            outcome = EvalOutcome(case=case, answer=answer, latency_ms=latency)
            outcomes.append(outcome)
            print(
                f"[{'PASS' if outcome.passed else 'FAIL'}] {case.id} {answer.status.value:<20} "
                f"top={answer.confidence.top_score:.3f} conf={answer.confidence.value:.2f} "
                f"pages={list(outcome.retrieved_pages)}  {case.question}"
            )

    summary = summarize(outcomes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.md").write_text(
        to_markdown(outcomes, summary, settings.retrieval_threshold), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(
            {
                "summary": asdict(summary),
                "results": [
                    {
                        "id": o.case.id,
                        "question": o.case.question,
                        "answerable": o.case.answerable,
                        "status": o.answer.status.value,
                        "passed": o.passed,
                        "answer": o.answer.text,
                        "confidence": o.answer.confidence.value,
                        "top_score": o.answer.confidence.top_score,
                        "retrieved_pages": list(o.retrieved_pages),
                        "cited_pages": list(o.answer.cited_pages),
                        "unsupported_claims": list(o.answer.unsupported_claims),
                        "latency_ms": o.latency_ms,
                    }
                    for o in outcomes
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nanswerable->grounded {summary.answerable_accuracy:.0%} | "
        f"unanswerable->fallback {summary.fallback_accuracy:.0%} | "
        f"page hit {summary.page_hit_rate:.0%} | "
        f"threshold {settings.retrieval_threshold} (suggested: {summary.suggested_threshold})"
    )
    print(f"Wrote {RESULTS_DIR / 'latest.md'}")
    return 0 if all(o.passed for o in outcomes) else 2


if __name__ == "__main__":
    run(main)
