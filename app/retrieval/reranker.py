"""Cross-encoder reranking: top-20 → top-5."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.models import RetrievedChunk


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.cross_encoder_model)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    model = get_cross_encoder()
    pairs = [(query, rc.chunk.text) for rc in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    for rc, score in zip(candidates, scores):
        rc.rerank_score = float(score)
        # Sigmoid-normalized chunk relevance for downstream confidence
        rc.chunk_relevance_score = _sigmoid(rc.rerank_score)

    ranked = sorted(candidates, key=lambda r: r.rerank_score, reverse=True)
    return ranked[:top_k]


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))