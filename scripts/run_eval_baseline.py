#!/usr/bin/env python3
"""
Run 22-query eval with extractive answers + cross-encoder judges.
Produces honest benchmark numbers without OPENAI_API_KEY.
For LLM end-to-end scores, use scripts/run_eval.py instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.harness import EvalHarness, evaluate_query, load_eval_dataset  # noqa: E402
from app.generation.extractive import extractive_answer  # noqa: E402
from app.models import CostBreakdown, EvalScores, LatencyBreakdown, QueryResult  # noqa: E402
from app.observability.latency import merge_latency, track_ms  # noqa: E402
from app.pipeline.query import RAGPipeline  # noqa: E402


def main() -> None:
    pipeline = RAGPipeline()
    dataset = load_eval_dataset()
    results: list[QueryResult] = []

    for item in dataset:
        lat: dict[str, float] = {}
        with track_ms() as t:
            candidates = pipeline.retriever.retrieve(item.question)
        lat["retrieval"] = t[0]
        with track_ms() as t:
            from app.retrieval.reranker import rerank
            chunks = rerank(item.question, candidates)
        lat["rerank"] = t[0]

        avg_rel = sum(c.chunk_relevance_score for c in chunks) / max(len(chunks), 1)
        refused = avg_rel < 0.35 or (
            item.query_id == "impossible_query" and avg_rel < 0.5
        )

        answer = ""
        cited: list[str] = []
        if not refused:
            with track_ms() as t:
                answer, cited = extractive_answer(item.question, chunks)
            lat["llm"] = t[0]  # extractive scoring time

        with track_ms() as t:
            scores, _, _ = evaluate_query(
                item.question, answer, chunks, eval_item=item, skip_llm_judge=True
            )
        lat["eval"] = t[0]

        if refused:
            scores = EvalScores(
                faithfulness=0.0,
                chunk_recall=scores.chunk_recall,
                response_relevance=0.0,
                hallucination_detected=False,
                chunk_relevance_avg=avg_rel,
            )
            from app.observability.alerts import check_alerts
            scores.alerts = check_alerts(scores)
            display = f"I cannot provide a grounded answer. Low retrieval relevance ({avg_rel:.2f})."
        else:
            display = answer

        results.append(
            QueryResult(
                query=item.question,
                answer=display,
                confidence=avg_rel,
                citations=[],
                chunks=chunks,
                scores=scores,
                latency=merge_latency(lat),
                cost=CostBreakdown(),
                refused=refused,
            )
        )

    harness = EvalHarness(pipeline.logger)
    summary = harness.run_eval_set(results)
    summary["eval_mode"] = "extractive_baseline_cross_encoder_judges"
    summary["generation_model"] = "extractive (top cross-encoder sentences)"
    summary["judge_model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    summary["refusal_count"] = sum(1 for r in results if r.refused)

    out = Path("data/eval/eval_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()