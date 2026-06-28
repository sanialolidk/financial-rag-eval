"""Hybrid dense + BM25 retrieval with reciprocal rank fusion."""

from __future__ import annotations

from app.config import settings
from app.models import Chunk, RetrievedChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chroma_store import ChromaStore


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


class HybridRetriever:
    def __init__(self, chroma: ChromaStore, bm25: BM25Index) -> None:
        self.chroma = chroma
        self.bm25 = bm25

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.hybrid_top_k

        dense_hits = self.chroma.query(query, top_k=top_k)
        sparse_hits = self.bm25.search(query, top_k=top_k)

        fused: dict[str, RetrievedChunk] = {}

        for rank, (chunk, score) in enumerate(dense_hits):
            rc = fused.get(chunk.chunk_id) or RetrievedChunk(chunk=chunk)
            rc.dense_score = score
            rc.hybrid_score += settings.dense_weight * _rrf_score(rank)
            fused[chunk.chunk_id] = rc

        for rank, (chunk, score) in enumerate(sparse_hits):
            rc = fused.get(chunk.chunk_id) or RetrievedChunk(chunk=chunk)
            rc.bm25_score = score
            rc.hybrid_score += settings.bm25_weight * _rrf_score(rank)
            fused[chunk.chunk_id] = rc

        ranked = sorted(fused.values(), key=lambda r: r.hybrid_score, reverse=True)
        return ranked[:top_k]