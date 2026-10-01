"""Exercise 3.5 — measure retrieval metrics before/after ``rerank_by_overlap``.

Reorders the SAME retrieved chunks from ``artifacts/actual_answers.json`` and
recomputes Context Recall / Context Precision with the evaluation core.

The reranker query is the customer question only: using the expected answer
at inference time would leak gold data into the system under evaluation.

Run from the repo root:
    python bonus/rerank_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from template import RAGASEvaluator, rerank_by_overlap  # noqa: E402


def main() -> int:
    golden = json.loads((REPO_ROOT / "golden_dataset.json").read_text(encoding="utf-8"))
    actual = json.loads(
        (REPO_ROOT / "artifacts" / "actual_answers.json").read_text(encoding="utf-8")
    )
    expected_by_id = {pair["id"]: pair["expected_answer"] for pair in golden["qa_pairs"]}
    evaluator = RAGASEvaluator()

    rows: list[dict[str, object]] = []
    for record in actual["answers"]:
        chunks = [context["text"] for context in record["retrieved_contexts"]]
        expected = expected_by_id[record["id"]]
        reranked = rerank_by_overlap(chunks, record["question"])
        assert sorted(reranked) == sorted(chunks), "reranking must keep the same set"
        rows.append(
            {
                "id": record["id"],
                "recall_before": evaluator.evaluate_context_recall(chunks, expected),
                "recall_after": evaluator.evaluate_context_recall(reranked, expected),
                "precision_before": evaluator.evaluate_context_precision(chunks, expected),
                "precision_after": evaluator.evaluate_context_precision(reranked, expected),
                "order_changed": reranked != chunks,
            }
        )

    print("| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision | Order changed |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        delta = row["precision_after"] - row["precision_before"]
        print(
            f"| {row['id']} | {row['recall_before']:.3f} | {row['recall_after']:.3f} | "
            f"{row['precision_before']:.3f} | {row['precision_after']:.3f} | "
            f"{delta:+.3f} | {'yes' if row['order_changed'] else 'no'} |"
        )
    count = len(rows)
    for key in ("recall_before", "recall_after", "precision_before", "precision_after"):
        print(f"avg {key}: {sum(row[key] for row in rows) / count:.3f}")

    output = REPO_ROOT / "artifacts" / "rerank_results.json"
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
