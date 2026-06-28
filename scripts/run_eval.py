#!/usr/bin/env python3
"""Run the full eval dataset and print aggregate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.harness import EvalHarness, load_eval_dataset  # noqa: E402
from app.pipeline.query import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval harness on golden query set")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval queries")
    parser.add_argument("--output", type=str, default="data/logs/eval_summary.json", help="Write summary JSON to path")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    dataset = load_eval_dataset()
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Running {len(dataset)} eval queries...")
    results = []
    for i, item in enumerate(dataset, 1):
        print(f"  [{i}/{len(dataset)}] {item.query_id}: {item.question[:60]}...")
        results.append(pipeline.query(item.question, run_eval=True, eval_item=item))

    harness = EvalHarness(pipeline.logger)
    summary = harness.run_eval_set(results)

    print("\n=== Eval Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()