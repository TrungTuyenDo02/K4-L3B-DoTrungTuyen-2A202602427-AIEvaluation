"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        Retrieval-side metrics are intentionally excluded: they diagnose the
        retriever, not the quality of the final answer.
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a score into [low, high]."""
    return max(low, min(high, value))


def _overlap_ratio(source: str, reference: str, empty_score: float = 1.0) -> float:
    """Return |tokens(source) ∩ tokens(reference)| / |tokens(reference)|.

    ``empty_score`` is returned when ``reference`` has no content tokens, which
    avoids division by zero.
    """
    reference_tokens = _tokenize(reference)
    if not reference_tokens:
        return empty_score
    overlap = _tokenize(source) & reference_tokens
    return _clamp(len(overlap) / len(reference_tokens))


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        # Denominator = answer tokens: every answer word should come from context.
        return _overlap_ratio(context, answer)

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # Denominator = question tokens: how much of the question is addressed.
        return _overlap_ratio(answer, question)

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # Denominator = expected tokens: how much of the reference is covered.
        return _overlap_ratio(answer, expected)

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        return _clamp(len(expected_tokens & union_tokens) / len(expected_tokens))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        # Step 1: binary relevance label for each chunk, in retriever order.
        relevance_flags = [
            len(_tokenize(chunk) & expected_tokens) / len(expected_tokens)
            >= relevance_threshold
            for chunk in contexts
        ]
        total_relevant = sum(relevance_flags)
        if total_relevant == 0:
            return 0.0

        # Steps 2-3: sum Precision@k only at ranks k that hold a relevant chunk.
        relevant_seen = 0
        precision_sum = 0.0
        for k, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                relevant_seen += 1
                precision_sum += relevant_seen / k
        return _clamp(precision_sum / total_relevant)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(
                question=question,
                expected_answer=expected,
                context=context,
                retrieved_contexts=list(contexts) if contexts is not None else [],
            ),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    query_tokens = _tokenize(query)
    # sorted() is stable, so ties keep the retriever's original order.
    return sorted(
        contexts,
        key=lambda chunk: len(_tokenize(chunk) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        criteria_lines = "\n".join(
            f"- {name}: {description}" for name, description in rubric.items()
        )
        output_example = json.dumps({name: 0.0 for name in rubric})
        prompt = (
            "You are an impartial evaluator. Score the answer against each "
            "criterion on a scale from 0.0 (fails) to 1.0 (fully satisfies). "
            "Judge only content quality; do not reward length.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Criteria:\n{criteria_lines}\n\n"
            "Respond with a JSON object only, mapping each criterion to a score, "
            f"for example: {output_example}"
        )

        raw_response = self.judge_llm_fn(prompt)
        parsed = self._parse_json_object(raw_response)
        # Accept both {"criterion": score} and {"scores": {"criterion": score}}.
        if isinstance(parsed.get("scores"), dict):
            parsed = parsed["scores"]

        scores: dict[str, float] = {}
        for criterion in rubric:
            value = parsed.get(criterion)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[criterion] = _clamp(float(value))
            else:
                scores[criterion] = 0.5  # fallback when unparseable / missing
        return {"scores": scores, "reasoning": raw_response}

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        """Extract the first JSON object from ``text``; return {} on failure."""
        if not isinstance(text, str):
            return {}
        candidates = [text]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                return value
        return {}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        per_response_means: list[float] = []
        all_scores: list[float] = []
        for entry in scores_batch:
            values = [
                float(v)
                for v in entry.get("scores", {}).values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if values:
                per_response_means.append(sum(values) / len(values))
                all_scores.extend(values)

        if not all_scores:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        overall_mean = sum(all_scores) / len(all_scores)
        # Positional: the first-presented response beats every later response.
        positional_bias = len(per_response_means) >= 2 and all(
            per_response_means[0] > later for later in per_response_means[1:]
        )
        return {
            "positional_bias": positional_bias,
            "leniency_bias": overall_mean > 0.8,
            "severity_bias": overall_mean < 0.3,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context or "",
                expected=pair.expected_answer,
                # An empty list means "no retrieval trace", so keep metrics None.
                contexts=pair.retrieved_contexts or None,
            )
            result.qa_pair = pair  # keep metadata such as the dataset ID
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        def _mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        recall_values = [r.context_recall for r in results if r.context_recall is not None]
        precision_values = [
            r.context_precision for r in results if r.context_precision is not None
        ]
        failure_types = Counter(
            r.failure_type for r in results if not r.passed and r.failure_type
        )
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_faithfulness": _mean([r.faithfulness for r in results]) or 0.0,
            "avg_relevance": _mean([r.relevance for r in results]) or 0.0,
            "avg_completeness": _mean([r.completeness for r in results]) or 0.0,
            "avg_context_recall": _mean(recall_values),
            "avg_context_precision": _mean(precision_values),
            "failure_types": dict(failure_types),
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions

        Drops of exactly 0.05 or less are treated as run-to-run noise.
        """
        max_allowed_drop = 0.05
        metrics = ("faithfulness", "relevance", "completeness")

        def _avg(results: list[EvalResult], metric: str) -> float:
            if not results:
                return 0.0
            return sum(getattr(r, metric) for r in results) / len(results)

        report: dict[str, Any] = {}
        regressions: list[str] = []
        for metric in metrics:
            new_avg = _avg(new_results, metric)
            baseline_avg = _avg(baseline_results, metric)
            report[f"new_avg_{metric}"] = new_avg
            report[f"baseline_avg_{metric}"] = baseline_avg
            if baseline_avg - new_avg > max_allowed_drop:
                regressions.append(metric)
        report["regressions"] = regressions
        report["passed"] = not regressions
        return report

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            r
            for r in results
            if min(r.faithfulness, r.relevance, r.completeness) < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    # Targeted fix per failure type, shared by suggestions and the log.
    TYPE_SUGGESTIONS: dict[str, str] = {
        "hallucination": (
            "Add a grounding guardrail: instruct the generator to answer only "
            "from retrieved chunks and reject drafts whose faithfulness to the "
            "retrieved context is below 0.5"
        ),
        "irrelevant": (
            "Tighten the system prompt so the answer addresses the user's exact "
            "question first, and add few-shot examples of on-intent answers"
        ),
        "incomplete": (
            "Add few-shot examples of complete answers that keep every amount, "
            "date, condition and exception, and raise top-k or merge adjacent "
            "chunks so all required evidence reaches the generator"
        ),
        "off_topic": (
            "Add an intent/scope check before generation so out-of-scope or "
            "ambiguous requests get a scoped refusal or clarifying question"
        ),
    }

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        return dict(Counter(f.failure_type or "unclassified" for f in failures))

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        lowest = min(scores.values())
        lowest_metrics = [name for name, value in scores.items() if value == lowest]
        all_below_pass = all(value < 0.5 for value in scores.values())

        # No single weakest dimension -> the whole pipeline needs review.
        if all_below_pass or len(lowest_metrics) > 1:
            return "Multiple issues detected — review full pipeline"
        if lowest_metrics[0] == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        if lowest_metrics[0] == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        return (
            "Answer is missing key information — increase context window "
            "or improve generation"
        )

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        With one suggestion per failure, suggestions are matched by position.
        With a shorter (prioritized) list, each failure gets the suggestion that
        targets its failure_type, falling back to position, then to triage.
        """

        def _cell(text: str) -> str:
            return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()

        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for index, failure in enumerate(failures):
            failure_id = f"F{index + 1:03d}"
            dataset_id = failure.qa_pair.metadata.get("id") if failure.qa_pair else None
            if dataset_id:
                failure_id = f"{failure_id} ({dataset_id})"
            suggestion = self._match_suggestion(failure, index, failures, suggestions)
            lines.append(
                f"| {_cell(failure_id)} | {_cell(failure.failure_type or 'unclassified')} "
                f"| {_cell(self.find_root_cause(failure))} | {_cell(suggestion)} | Open |"
            )
        return "\n".join(lines)

    def _match_suggestion(
        self,
        failure: EvalResult,
        index: int,
        failures: list[EvalResult],
        suggestions: list[str],
    ) -> str:
        """Pick the suggestion for one improvement-log row."""
        if len(suggestions) >= len(failures):
            return suggestions[index]
        targeted = self.TYPE_SUGGESTIONS.get((failure.failure_type or "").lower())
        if targeted in suggestions:
            return targeted
        if index < len(suggestions):
            return suggestions[index]
        return "Triage required — no matching suggestion"

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []

        suggestions: list[str] = []
        # Most frequent failure type first: one fix there resolves most cases.
        for failure_type, _count in Counter(
            f.failure_type for f in failures if f.failure_type
        ).most_common():
            suggestion = self.TYPE_SUGGESTIONS.get(failure_type.lower())
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)

        recalls = [f.context_recall for f in failures if f.context_recall is not None]
        precisions = [
            f.context_precision for f in failures if f.context_precision is not None
        ]
        if recalls and sum(recalls) / len(recalls) < 0.6:
            suggestions.append(
                "Improve retrieval coverage with query expansion or hybrid "
                "BM25 + dense retrieval; verify with average Context Recall"
            )
        if precisions and sum(precisions) / len(precisions) < 0.6:
            suggestions.append(
                "Add a reranker so relevant chunks are ranked before noise; "
                "verify with average Context Precision"
            )

        fallbacks = [
            "Add every failed case to the regression benchmark and run "
            "run_regression() on each prompt, retrieval or model change",
            "Calibrate the word-overlap metrics against an LLM judge or human "
            "labels on the failed cases to separate real errors from paraphrases",
            "Review chunking so each policy rule stays in one chunk with its "
            "conditions and exceptions",
        ]
        for fallback in fallbacks:
            if len(suggestions) >= 3:
                break
            suggestions.append(fallback)
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
