#!/usr/bin/env python3
"""Measure top-1 chunk relevance before vs after cross-encoder reranking."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.harness import load_eval_dataset  # noqa: E402
from app.retrieval.bm25_index import BM25Index  # noqa: E402
from app.retrieval.chroma_store import ChromaStore  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.reranker import get_cross_encoder, rerank  # noqa: E402


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _top1_relevance(query: str, chunk_text: str) -> float:
    model = get_cross_encoder()
    score = float(model.predict([(query, chunk_text)], show_progress_bar=False)[0])
    return round(_sigmoid(score), 3)


def main() -> None:
    out = Path("data/eval/rerank_ablation.json")
    chroma = ChromaStore()
    bm25 = BM25Index()
    if not bm25.load() or chroma.count == 0:
        raise SystemExit("Indexes missing. Run ingest first.")

    retriever = HybridRetriever(chroma, bm25)
    dataset = load_eval_dataset()

    before_scores: list[float] = []
    after_scores: list[float] = []
    per_query: list[dict] = []

    for item in dataset:
        candidates = retriever.retrieve(item.question)
        if not candidates:
            continue
        pre_top = candidates[0]
        pre_rel = _top1_relevance(item.question, pre_top.chunk.text)

        post = rerank(item.question, list(candidates))
        post_top = post[0] if post else pre_top
        post_rel = post_top.chunk_relevance_score

        before_scores.append(pre_rel)
        after_scores.append(post_rel)
        per_query.append({
            "query_id": item.query_id,
            "before_top1_relevance": pre_rel,
            "after_top1_relevance": post_rel,
            "delta": round(post_rel - pre_rel, 3),
            "hybrid_top1_id": pre_top.chunk.chunk_id,
            "rerank_top1_id": post_top.chunk.chunk_id,
            "rank_changed": pre_top.chunk.chunk_id != post_top.chunk.chunk_id,
        })

    n = len(before_scores)
    avg_before = round(sum(before_scores) / n, 3)
    avg_after = round(sum(after_scores) / n, 3)
    rank_changes = sum(1 for q in per_query if q["rank_changed"])

    summary = {
        "num_queries": n,
        "avg_top1_relevance_before_rerank": avg_before,
        "avg_top1_relevance_after_rerank": avg_after,
        "improvement_absolute": round(avg_after - avg_before, 3),
        "improvement_percent": round((avg_after - avg_before) / max(avg_before, 0.001) * 100, 1),
        "top1_rank_changed_count": rank_changes,
        "top1_rank_changed_pct": round(rank_changes / n * 100, 1),
        "per_query": per_query,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Rerank Ablation ===")
    print(f"  avg top-1 relevance BEFORE rerank: {avg_before:.1%} ({avg_before})")
    print(f"  avg top-1 relevance AFTER rerank:  {avg_after:.1%} ({avg_after})")
    print(f"  improvement: +{(avg_after - avg_before):.3f} ({summary['improvement_percent']}%)")
    print(f"  top-1 chunk changed on {rank_changes}/{n} queries ({summary['top1_rank_changed_pct']}%)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()