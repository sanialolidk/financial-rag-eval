"""Page-aware chunking with 512-token windows and rich metadata."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import tiktoken

from app.config import settings
from app.ingestion.sec_fetcher import _section_hint
from app.models import Chunk

_ENC = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return _ENC.encode(text)


def _detokenize(tokens: list[int]) -> str:
    return _ENC.decode(tokens)


def chunk_document(doc: dict[str, Any]) -> list[Chunk]:
    """
    Chunk a document page-by-page, then split long pages into ~512-token segments.
    Overlap preserves context across chunk boundaries.
    """
    chunks: list[Chunk] = []
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    step = max(size - overlap, 1)

    ticker = doc["ticker"]
    filing_type = doc["filing_type"]
    filing_date = doc["filing_date"]
    source_url = doc.get("source_url", "")

    for page_num, page_text in doc["pages"]:
        page_text = page_text.strip()
        if not page_text:
            continue
        tokens = _tokenize(page_text)
        if len(tokens) <= size:
            windows = [(0, len(tokens))]
        else:
            windows = []
            start = 0
            while start < len(tokens):
                end = min(start + size, len(tokens))
                windows.append((start, end))
                if end >= len(tokens):
                    break
                start += step

        for win_idx, (start, end) in enumerate(windows):
            text = _detokenize(tokens[start:end]).strip()
            if len(text) < 80:
                continue
            section = _section_hint(text)
            digest = hashlib.sha1(
                f"{ticker}|{filing_type}|{filing_date}|{page_num}|{win_idx}|{text[:200]}".encode()
            ).hexdigest()[:16]
            chunk_id = f"{ticker}_{filing_type}_{filing_date}_p{page_num}_w{win_idx}_{digest}"

            meta = {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "page": page_num,
                "window": win_idx,
                "section": section,
                "source_url": source_url,
                "token_count": end - start,
                "doc_path": doc.get("path", ""),
            }
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta))

    return chunks


def chunk_all_documents(docs: list[dict]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


def ingest_transcript_file(path: str, ticker: str, call_date: str) -> list[Chunk]:
    """Optional earnings-call transcript from a local .txt file."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    doc = {
        "ticker": ticker,
        "filing_type": "EARNINGS_TRANSCRIPT",
        "filing_date": call_date,
        "source_url": f"local://{path}",
        "path": path,
        "pages": [(1, text)],
        "full_text": text,
    }
    return chunk_document(doc)