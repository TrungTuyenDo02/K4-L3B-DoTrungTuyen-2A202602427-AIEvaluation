"""Exercise 3.4 — compare RAGAS and DeepEval on the same 20 benchmark inputs.

Both frameworks score the SAME saved traces (question, actual answer,
retrieved chunks, expected answer) with the SAME judge model, alongside the
lab's word-overlap heuristic from ``template.py``.

This script is optional bonus work. Its dependencies are NOT part of the lab
``requirements.txt``; install them in a separate virtual environment:

    python -m venv .venv-bonus
    .venv-bonus/Scripts/python -m pip install -r bonus/requirements-bonus.txt
    .venv-bonus/Scripts/python bonus/compare_frameworks.py
    .venv-bonus/Scripts/python bonus/compare_frameworks.py --resume  # only re-score failed calls

Requires OPENAI_API_KEY in the repo ``.env`` (makes paid OpenAI calls).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from template import RAGASEvaluator  # noqa: E402

JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CONCURRENCY = 5
MAX_ATTEMPTS = 3
OUTPUT_PATH = REPO_ROOT / "artifacts" / "framework_comparison.json"
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def load_cases() -> list[dict[str, Any]]:
    golden = json.loads((REPO_ROOT / "golden_dataset.json").read_text(encoding="utf-8"))
    actual = json.loads(
        (REPO_ROOT / "artifacts" / "actual_answers.json").read_text(encoding="utf-8")
    )
    expected_by_id = {pair["id"]: pair for pair in golden["qa_pairs"]}
    cases = []
    for record in actual["answers"]:
        gold = expected_by_id[record["id"]]
        cases.append(
            {
                "id": record["id"],
                "difficulty": gold["difficulty"],
                "question": record["question"],
                "answer": record["actual_answer"],
                "expected": gold["expected_answer"],
                "retrieved": [context["text"] for context in record["retrieved_contexts"]],
                "gold_context": "\n\n".join(context["text"] for context in gold["contexts"]),
            }
        )
    return cases


def answers_generated_at() -> str | None:
    """Identify which RAG run (actual_answers.json) the scores belong to."""
    actual = json.loads(
        (REPO_ROOT / "artifacts" / "actual_answers.json").read_text(encoding="utf-8")
    )
    return actual.get("generated_at")


def score_lab_heuristic(case: dict[str, Any]) -> dict[str, float]:
    """Word-overlap scores from the lab core, as computed by evaluate_answers.py."""
    evaluator = RAGASEvaluator()
    return {
        "faithfulness": evaluator.evaluate_faithfulness(case["answer"], case["gold_context"]),
        "answer_relevancy": evaluator.evaluate_relevance(case["answer"], case["question"]),
        "context_recall": evaluator.evaluate_context_recall(case["retrieved"], case["expected"]),
        "context_precision": evaluator.evaluate_context_precision(
            case["retrieved"], case["expected"]
        ),
    }


async def score_ragas(
    cases: list[dict[str, Any]], pending: set[tuple[str, str]] | None = None
) -> dict[str, dict[str, Any]]:
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI()
    llm = llm_factory(JUDGE_MODEL, client=client)
    embeddings = embedding_factory("openai", model=EMBEDDING_MODEL, client=client)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_recall": ContextRecall(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
    }
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run_one(case: dict[str, Any], name: str) -> tuple[str, str, Any]:
        kwargs: dict[str, Any] = {"user_input": case["question"]}
        if name in ("faithfulness", "answer_relevancy"):
            kwargs["response"] = case["answer"]
        if name != "answer_relevancy":
            kwargs["retrieved_contexts"] = case["retrieved"]
        if name in ("context_recall", "context_precision"):
            kwargs["reference"] = case["expected"]
        async with semaphore:
            try:
                result = await _with_retries(lambda: metrics[name].ascore(**kwargs))
                return case["id"], name, {"score": float(result.value), "reason": None}
            except Exception as exc:  # keep the run going; record the failure
                return case["id"], name, {"score": None, "reason": f"ERROR: {exc}"}

    tasks = [
        run_one(case, name)
        for case in cases
        for name in METRICS
        if pending is None or (case["id"], name) in pending
    ]
    output: dict[str, dict[str, Any]] = {case["id"]: {} for case in cases}
    for case_id, name, value in await asyncio.gather(*tasks):
        output[case_id][name] = value
    return output


async def score_deepeval(
    cases: list[dict[str, Any]], pending: set[tuple[str, str]] | None = None
) -> dict[str, dict[str, Any]]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    metric_classes = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
        "context_recall": ContextualRecallMetric,
        "context_precision": ContextualPrecisionMetric,
    }
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run_one(case: dict[str, Any], name: str) -> tuple[str, str, Any]:
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=case["answer"],
            expected_output=case["expected"],
            retrieval_context=case["retrieved"],
        )
        # Metric objects keep per-run state, so use a fresh instance per case.
        metric = metric_classes[name](model=JUDGE_MODEL, include_reason=True)
        async with semaphore:
            try:
                await _with_retries(lambda: metric.a_measure(test_case, _show_indicator=False))
                return case["id"], name, {"score": float(metric.score), "reason": metric.reason}
            except Exception as exc:
                return case["id"], name, {"score": None, "reason": f"ERROR: {exc}"}

    tasks = [
        run_one(case, name)
        for case in cases
        for name in METRICS
        if pending is None or (case["id"], name) in pending
    ]
    output: dict[str, dict[str, Any]] = {case["id"]: {} for case in cases}
    for case_id, name, value in await asyncio.gather(*tasks):
        output[case_id][name] = value
    return output


async def _with_retries(coro_factory: Any) -> Any:
    """Retry transient API/network failures with exponential backoff."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            await asyncio.sleep(2 ** attempt)
    return None


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


async def main_async() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing from .env")
        return 2
    cases = load_cases()
    resume = "--resume" in sys.argv[1:]
    previous: dict[str, Any] = {}
    if resume and OUTPUT_PATH.exists():
        previous = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    previous_rows = {row["id"]: row for row in previous.get("results", [])}
    previous_run = previous.get("summary", {}).get("actual_answers_generated_at")
    if previous_run and previous_run != answers_generated_at():
        print("ERROR: saved comparison belongs to a different RAG run; rerun without --resume")
        return 2

    def pending_for(framework: str) -> set[tuple[str, str]] | None:
        if not resume:
            return None
        return {
            (case["id"], metric)
            for case in cases
            for metric in METRICS
            if previous_rows.get(case["id"], {}).get(framework, {}).get(metric, {}).get("score") is None
        }

    def merge(framework: str, fresh: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        merged = {case["id"]: dict(previous_rows.get(case["id"], {}).get(framework, {})) for case in cases}
        for case_id, scores in fresh.items():
            merged[case_id].update(scores)
        return merged

    previous_runtime = previous.get("summary", {}).get("runtime_seconds", {})
    started = time.perf_counter()
    ragas_scores = merge("ragas", await score_ragas(cases, pending_for("ragas")))
    ragas_seconds = time.perf_counter() - started + (previous_runtime.get("ragas", 0) if resume else 0)

    started = time.perf_counter()
    deepeval_scores = merge("deepeval", await score_deepeval(cases, pending_for("deepeval")))
    deepeval_seconds = time.perf_counter() - started + (previous_runtime.get("deepeval", 0) if resume else 0)

    rows = []
    for case in cases:
        rows.append(
            {
                "id": case["id"],
                "difficulty": case["difficulty"],
                "lab_heuristic": score_lab_heuristic(case),
                "ragas": ragas_scores[case["id"]],
                "deepeval": deepeval_scores[case["id"]],
            }
        )

    summary: dict[str, Any] = {
        "actual_answers_generated_at": answers_generated_at(),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "runtime_seconds": {"ragas": round(ragas_seconds, 1), "deepeval": round(deepeval_seconds, 1)},
        "averages": {},
        "errors": {"ragas": 0, "deepeval": 0},
    }
    for metric in METRICS:
        summary["averages"][metric] = {
            "lab_heuristic": _mean([row["lab_heuristic"][metric] for row in rows]),
            "ragas": _mean([row["ragas"][metric]["score"] for row in rows]),
            "deepeval": _mean([row["deepeval"][metric]["score"] for row in rows]),
        }
    for framework in ("ragas", "deepeval"):
        summary["errors"][framework] = sum(
            1 for row in rows for metric in METRICS if row[framework][metric]["score"] is None
        )

    output = OUTPUT_PATH
    output.write_text(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    print("| ID | Faith lab/RAGAS/DE | Relevancy lab/RAGAS/DE | Recall lab/RAGAS/DE | Precision lab/RAGAS/DE |")
    print("|---|---|---|---|---|")
    for row in rows:
        cells = [
            " / ".join(
                fmt(value)
                for value in (
                    row["lab_heuristic"][metric],
                    row["ragas"][metric]["score"],
                    row["deepeval"][metric]["score"],
                )
            )
            for metric in METRICS
        ]
        print(f"| {row['id']} | " + " | ".join(cells) + " |")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
