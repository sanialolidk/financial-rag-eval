"""End-to-end ingestion: SEC filings → chunks → Chroma + BM25."""

from __future__ import annotations

from pathlib import Path

from app.config import CORPUS_DIR, ensure_data_dirs
from app.ingestion.chunker import chunk_all_documents, ingest_transcript_file
from app.ingestion.sec_fetcher import download_filings, load_corpus_text_files
from app.models import Chunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chroma_store import ChromaStore


def ingest_corpus(
    tickers: list[str] | None = None,
    download: bool = True,
    reset_store: bool = True,
) -> dict:
    ensure_data_dirs()

    if download:
        print("Downloading SEC filings (10-K, 10-Q)...")
        download_filings(tickers=tickers, limit_per_type=1)

    docs = load_corpus_text_files(CORPUS_DIR)
    if not docs:
        raise RuntimeError(
            f"No corpus files in {CORPUS_DIR}. Run with --download or place .txt filings there."
        )

    # Optional local earnings transcripts
    transcript_dir = CORPUS_DIR / "transcripts"
    extra_chunks: list[Chunk] = []
    if transcript_dir.exists():
        for path in transcript_dir.glob("*.txt"):
            # naming: AAPL_2024Q3_transcript.txt
            parts = path.stem.split("_")
            ticker = parts[0]
            call_date = parts[1] if len(parts) > 1 else "unknown"
            extra_chunks.extend(ingest_transcript_file(str(path), ticker, call_date))

    chunks = chunk_all_documents(docs) + extra_chunks
    if not chunks:
        raise RuntimeError("Chunking produced zero chunks — check corpus files.")

    chroma = ChromaStore()
    if reset_store:
        chroma.reset()
    chroma.add_chunks(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save()

    companies = sorted({c.metadata["ticker"] for c in chunks})
    filing_types = sorted({c.metadata["filing_type"] for c in chunks})

    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "companies": companies,
        "filing_types": filing_types,
        "chroma_count": chroma.count,
        "bm25_count": bm25.size,
    }
    print(f"Ingestion complete: {stats}")
    return stats