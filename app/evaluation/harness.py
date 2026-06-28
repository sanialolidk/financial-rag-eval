"""Evaluation harness — scores every query and persists logs."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import EVAL_DATASET_PATH, settings
from app.evaluation.hallucination import detect_hallucinations
from app.evaluation.metrics import (
    score_chunk_recall,
    score_faithfulness,
    score_response_relevance,
)
from app.models import EvalQuery, EvalScores, QueryResult, RetrievedChunk
from app.observability.alerts import check_alerts
from app.observability.logger import QueryLogger


def load_eval_dataset(path: Path | None = None) -> list[EvalQuery]:
    path = path or EVAL_DATASET_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            query_id=item["query_id"],
            question=item["question"],
            expected_answer_contains=item.get("expected_answer_contains", []),
            expected_tickers=item.get("expected_tickers", []),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


def evaluate_query(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    eval_item: EvalQuery | None = None,
    skip_llm_judge: bool = False,
) -> tuple[EvalScores, int, int]:
    """Returns scores and total judge tokens (in, out)."""
    if skip_llm_judge or not answer:
        chunk_recall = score_chunk_recall(eval_item, chunks, answer)
        rel_avg = sum(rc.chunk_relevance_score for rc in chunks) / max(len(chunks), 1)
        scores = EvalScores(
            faithfulness=0.0 if not answer else 0.5,
            chunk_recall=chunk_recall,
            response_relevance=0.0 if not answer else 0.5,
            hallucination_detected=False,
            chunk_relevance_avg=rel_avg,
        )
        scores.alerts = check_alerts(scores)
        return scores, 0, 0

    faith, _, fi, fo = score_faithfulness(query, answer, chunks)
    rel, _, ri, ro = score_response_relevance(query, answer)
    chunk_recall = score_chunk_recall(eval_item, chunks, answer)
    hal_detected, hal_details, hi, ho = detect_hallucinations(answer, chunks)
    rel_avg = sum(rc.chunk_relevance_score for rc in chunks) / max(len(chunks), 1)

    scores = EvalScores(
        faithfulness=faith,
        chunk_recall=chunk_recall,
        response_relevance=rel,
        hallucination_detected=hal_detected,
        hallucination_details=hal_details,
        chunk_relevance_avg=rel_avg,
    )
    scores.alerts = check_alerts(scores)
    return scores, fi + ri + hi, fo + ro + ho


class EvalHarness:
    def __init__(self, logger: QueryLogger | None = None) -> None:
        self.logger = logger or QueryLogger()

    def run_eval_set(self, results: list[QueryResult]) -> dict:
        """Aggregate metrics across a batch of eval query results."""
        if not results:
            return {}

        faith = [r.scores.faithfulness for r in results if r.scores]
        recall = [r.scores.chunk_recall for r in results if r.scores]
        relevance = [r.scores.response_relevance for r in results if r.scores]
        retrieval_lat = [r.latency.retrieval_ms + r.latency.rerank_ms for r in results]
        total_lat = [r.latency.total_ms for r in results]
        hallucinations = sum(1 for r in results if r.scores and r.scores.hallucination_detected)
        alerts = sum(len(r.scores.alerts) for r in results if r.scores)

        summary = {
            "num_queries": len(results),
            "avg_faithfulness": round(sum(faith) / len(faith), 3) if faith else 0.0,
            "avg_chunk_recall": round(sum(recall) / len(recall), 3) if recall else 0.0,
            "avg_response_relevance": round(sum(relevance) / len(relevance), 3) if relevance else 0.0,
            "mean_retrieval_latency_ms": round(sum(retrieval_lat) / len(retrieval_lat), 1) if retrieval_lat else 0.0,
            "mean_total_latency_ms": round(sum(total_lat) / len(total_lat), 1) if total_lat else 0.0,
            "hallucination_count": hallucinations,
            "alert_count": alerts,
        }
        self.logger.log_eval_summary(summary)
        return summary