"""Latency tracking utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from app.models import LatencyBreakdown


@contextmanager
def track_ms() -> Generator[list[float], None, None]:
    bucket: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket[0] = (time.perf_counter() - start) * 1000.0


def merge_latency(parts: dict[str, float]) -> LatencyBreakdown:
    emb = parts.get("embedding", 0.0)
    ret = parts.get("retrieval", 0.0)
    rer = parts.get("rerank", 0.0)
    llm = parts.get("llm", 0.0)
    ev = parts.get("eval", 0.0)
    total = emb + ret + rer + llm + ev
    return LatencyBreakdown(
        embedding_ms=round(emb, 2),
        retrieval_ms=round(ret, 2),
        rerank_ms=round(rer, 2),
        llm_ms=round(llm, 2),
        eval_ms=round(ev, 2),
        total_ms=round(total, 2),
    )