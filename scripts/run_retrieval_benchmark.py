#!/usr/bin/env python3
"""Retrieval-only benchmark: chunk recall + latency (no OpenAI required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.harness import load_eval_dataset  # noqa: E402
from app.evaluation.metrics import score_chunk_recall  # noqa: E402
from app.observability.latency import merge_latency, track_ms  # noqa: E402
from app.retrieval.bm25_index import BM25Index  # noqa: E402
from app.retrieval.chroma_store import ChromaStore  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.reranker import rerank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/logs/retrieval_benchmark.json")
    args = parser.parse_args()

    chroma = ChromaStore()
    bm25 = BM25Index()
    if not bm25.load() or chroma.count == 0:
        raise SystemExit("Indexes missing. Run: python scripts/ingest_corpus.py")

    retriever = HybridRetriever(chroma, bm25)
    dataset = load_eval_dataset()

    recalls: list[float] = []
    retrieval_ms: list[float] = []
    rerank_ms: list[float] = []

    for item in dataset:
        lat: dict[str, float] = {}
        with track_ms() as t:
            candidates = retriever.retrieve(item.question)
        lat["retrieval"] = t[0]
        with track_ms() as t:
            top = rerank(item.question, candidates)
        lat["rerank"] = t[0]

        recall = score_chunk_recall(item, top, answer="")
        recalls.append(recall)
        retrieval_ms.append(lat["retrieval"] + lat["rerank"])

    summary = {
        "num_queries": len(dataset),
        "avg_chunk_recall": round(sum(recalls) / len(recalls), 3),
        "mean_retrieval_latency_ms": round(sum(retrieval_ms) / len(retrieval_ms), 1),
        "per_query": [
            {
                "query_id": item.query_id,
                "chunk_recall": recalls[i],
                "retrieval_ms": retrieval_ms[i],
            }
            for i, item in enumerate(dataset)
        ],
        "note": "Retrieval-only benchmark. Run scripts/run_eval.py with OPENAI_API_KEY for faithfulness scores.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Retrieval Benchmark ===")
    for k, v in summary.items():
        if k != "per_query":
            print(f"  {k}: {v}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()