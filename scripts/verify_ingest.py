#!/usr/bin/env python3
"""Smoke-test ingestion output: corpus files, Chroma, BM25."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CORPUS_DIR, CORPUS_COMPANIES  # noqa: E402
from app.ingestion.sec_fetcher import load_corpus_text_files  # noqa: E402
from app.retrieval.bm25_index import BM25Index  # noqa: E402
from app.retrieval.chroma_store import ChromaStore  # noqa: E402


def main() -> None:
    docs = load_corpus_text_files()
    txt_files = list(CORPUS_DIR.glob("*.txt"))
    print(f"Corpus .txt files: {len(txt_files)}")
    print(f"Parsed documents: {len(docs)}")

    tickers_found = {d["ticker"] for d in docs}
    missing = set(CORPUS_COMPANIES) - tickers_found
    if missing:
        print(f"WARNING missing tickers: {sorted(missing)}")
    else:
        print(f"All {len(CORPUS_COMPANIES)} tickers present")

    for d in docs[:3]:
        pages = len(d["pages"])
        chars = len(d["full_text"])
        print(f"  {d['ticker']} {d['filing_type']} {d['filing_date']}: {pages} pages, {chars:,} chars")

    chroma = ChromaStore()
    bm25 = BM25Index()
    bm25_ok = bm25.load()
    print(f"Chroma chunks: {chroma.count}")
    print(f"BM25 chunks: {bm25.size if bm25_ok else 'NOT LOADED'}")

    if chroma.count == 0 or not bm25_ok:
        raise SystemExit(1)
    print("Ingest verification OK")


if __name__ == "__main__":
    main()