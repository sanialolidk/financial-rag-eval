#!/usr/bin/env python3
"""Download SEC filings and build Chroma + BM25 indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.ingest import ingest_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC filings into vector + BM25 stores")
    parser.add_argument("--no-download", action="store_true", help="Skip SEC download, use existing .txt files")
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset, e.g. AAPL TSLA")
    args = parser.parse_args()

    stats = ingest_corpus(
        tickers=args.tickers,
        download=not args.no_download,
        reset_store=True,
    )
    print("\n=== Ingestion stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()