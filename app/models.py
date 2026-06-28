"""Shared data models for pipeline, API, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]

    @property
    def citation(self) -> str:
        ticker = self.metadata.get("ticker", "?")
        filing = self.metadata.get("filing_type", "?")
        period = self.metadata.get("filing_date", "?")
        section = self.metadata.get("section", "")
        page = self.metadata.get("page", "")
        parts = [f"{ticker} {filing} ({period})"]
        if section:
            parts.append(section)
        if page:
            parts.append(f"p.{page}")
        return " — ".join(parts)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    chunk_relevance_score: float = 0.0


@dataclass
class LatencyBreakdown:
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    eval_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class CostBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


@dataclass
class EvalScores:
    faithfulness: float
    chunk_recall: float
    response_relevance: float
    hallucination_detected: bool
    hallucination_details: str = ""
    chunk_relevance_avg: float = 0.0
    alerts: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    query: str
    answer: str
    confidence: float
    citations: list[dict[str, Any]]
    chunks: list[RetrievedChunk]
    scores: EvalScores | None
    latency: LatencyBreakdown
    cost: CostBreakdown
    refused: bool = False
    refusal_reason: str = ""
    timestamp: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "citations": self.citations,
            "chunks": [
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "citation": rc.chunk.citation,
                    "text_preview": rc.chunk.text[:400],
                    "rerank_score": rc.rerank_score,
                    "chunk_relevance_score": rc.chunk_relevance_score,
                }
                for rc in self.chunks
            ],
            "scores": None if self.scores is None else {
                "faithfulness": self.scores.faithfulness,
                "chunk_recall": self.scores.chunk_recall,
                "response_relevance": self.scores.response_relevance,
                "hallucination_detected": self.scores.hallucination_detected,
                "hallucination_details": self.scores.hallucination_details,
                "chunk_relevance_avg": self.scores.chunk_relevance_avg,
                "alerts": self.scores.alerts,
            },
            "latency": {
                "embedding_ms": self.latency.embedding_ms,
                "retrieval_ms": self.latency.retrieval_ms,
                "rerank_ms": self.latency.rerank_ms,
                "llm_ms": self.latency.llm_ms,
                "eval_ms": self.latency.eval_ms,
                "total_ms": self.latency.total_ms,
            },
            "cost": {
                "input_tokens": self.cost.input_tokens,
                "output_tokens": self.cost.output_tokens,
                "total_cost_usd": self.cost.total_cost_usd,
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvalQuery:
    query_id: str
    question: str
    expected_answer_contains: list[str]
    expected_tickers: list[str] = field(default_factory=list)
    notes: str = ""