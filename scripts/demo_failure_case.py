#!/usr/bin/env python3
"""
Reproduce documented failure case: NVDA revenue with a prior-year hallucination.
Shows hallucination detector + faithfulness gate refusing the answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.harness import evaluate_query  # noqa: E402
from app.retrieval.bm25_index import BM25Index  # noqa: E402
from app.retrieval.chroma_store import ChromaStore  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.reranker import rerank  # noqa: E402

QUESTION = "What Data Center revenue did NVIDIA report in the most recent fiscal period?"
# Simulated LLM slip: cites a plausible but wrong prior-year figure not in retrieved chunks
# Wrong year + wrong figure: 10-Q reports $75.2B (fiscal 2026), not $10.32B (fiscal 2023)
HALLUCINATED_ANSWER = (
    "NVIDIA reported Data Center revenue of $10.32 billion in fiscal 2023, "
    "representing strong year-over-year growth in AI infrastructure demand."
)


def main() -> None:
    chroma = ChromaStore()
    bm25 = BM25Index()
    if not bm25.load():
        raise SystemExit("BM25 index missing")

    retriever = HybridRetriever(chroma, bm25)
    candidates = retriever.retrieve(QUESTION)
    chunks = rerank(QUESTION, candidates)

    scores, _, _ = evaluate_query(
        QUESTION, HALLUCINATED_ANSWER, chunks, skip_llm_judge=True
    )

    # Pipeline refusal logic mirrors app/pipeline/query.py gates
    refused = (
        scores.hallucination_detected
        or scores.faithfulness < 0.55
    )

    report = {
        "case": "nvda_datacenter_prior_year_hallucination",
        "question": QUESTION,
        "hallucinated_answer": HALLUCINATED_ANSWER,
        "retrieved_top_chunk": chunks[0].chunk.citation if chunks else None,
        "faithfulness": scores.faithfulness,
        "hallucination_detected": scores.hallucination_detected,
        "hallucination_details": scores.hallucination_details,
        "alerts": scores.alerts,
        "system_refused": refused,
        "explanation": (
            "Answer cites fiscal 2023 / $10.32B but retrieved NVDA 10-Q chunk states "
            "'Data Center revenue was $75.2 billion' (fiscal 2026). Numeric grounding "
            "detector flags ungrounded dollar amount and fiscal year; pipeline refuses."
        ),
    }

    out = Path("data/eval/failure_case_nvda.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()