"""Extractive answer baseline — grounded sentences from top chunks (no LLM)."""

from __future__ import annotations

import re

from app.models import RetrievedChunk
from app.retrieval.reranker import get_cross_encoder


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 40]


def extractive_answer(query: str, chunks: list[RetrievedChunk], max_sentences: int = 2) -> tuple[str, list[str]]:
    """Pick highest cross-encoder-scored sentences from retrieved chunks."""
    if not chunks:
        return "", []

    model = get_cross_encoder()
    candidates: list[tuple[float, str, str]] = []

    for rc in chunks[:3]:
        for sent in _sentences(rc.chunk.text):
            if len(sent) > 500:
                continue
            score = float(model.predict([(query, sent)], show_progress_bar=False)[0])
            candidates.append((score, sent, rc.chunk.chunk_id))

    if not candidates:
        top = chunks[0]
        preview = top.chunk.text[:280].strip()
        return f"{preview} [{top.chunk.chunk_id}]", [top.chunk.chunk_id]

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = candidates[:max_sentences]
    cited_ids = list(dict.fromkeys(c[2] for c in picked))
    body = " ".join(c[1] for c in picked)
    cite = " ".join(f"[{cid}]" for cid in cited_ids)
    return f"{body} {cite}", cited_ids