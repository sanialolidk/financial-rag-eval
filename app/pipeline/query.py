"""End-to-end RAG pipeline with eval, latency, and cost tracking."""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.evaluation.harness import evaluate_query
from app.generation.answer import (
    build_citations,
    compute_confidence,
    generate_answer,
)
from app.generation.llm import cost_from_tokens
from app.models import EvalQuery, QueryResult
from app.observability.latency import merge_latency, track_ms
from app.observability.logger import QueryLogger
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import rerank


class RAGPipeline:
    def __init__(self) -> None:
        self.chroma = ChromaStore()
        self.bm25 = BM25Index()
        if not self.bm25.load():
            raise RuntimeError("BM25 index not found. Run: python scripts/ingest_corpus.py")
        if self.chroma.count == 0:
            raise RuntimeError("ChromaDB is empty. Run: python scripts/ingest_corpus.py")
        self.retriever = HybridRetriever(self.chroma, self.bm25)
        self.logger = QueryLogger()

    def query(
        self,
        question: str,
        run_eval: bool = True,
        eval_item: EvalQuery | None = None,
    ) -> QueryResult:
        lat_parts: dict[str, float] = {}
        token_parts: list[tuple[int, int]] = []

        with track_ms() as t:
            candidates = self.retriever.retrieve(question)
        lat_parts["retrieval"] = t[0]

        with track_ms() as t:
            top_chunks = rerank(question, candidates)
        lat_parts["rerank"] = t[0]

        with track_ms() as t:
            answer, cited_ids, llm_resp, refused, refusal_reason = generate_answer(
                question, top_chunks
            )
        lat_parts["llm"] = t[0]
        token_parts.append((llm_resp.input_tokens, llm_resp.output_tokens))

        scores = None
        if run_eval:
            with track_ms() as t:
                scores, j_in, j_out = evaluate_query(
                    question, answer, top_chunks, eval_item=eval_item
                )
            lat_parts["eval"] = t[0]
            token_parts.append((j_in, j_out))

            if scores and not refused:
                if scores.faithfulness < settings.min_faithfulness_to_answer:
                    refused = True
                    refusal_reason = (
                        f"Faithfulness {scores.faithfulness:.2f} below threshold "
                        f"{settings.min_faithfulness_to_answer}"
                    )
                    answer = ""
                if scores.hallucination_detected:
                    refused = True
                    refusal_reason = f"Hallucination detected: {scores.hallucination_details[:200]}"
                    answer = ""

        faith = scores.faithfulness if scores else 0.0
        hal = scores.hallucination_detected if scores else False
        confidence = compute_confidence(top_chunks, faith, hal)

        if not refused and confidence < settings.min_confidence_to_answer:
            refused = True
            refusal_reason = f"Confidence {confidence:.2f} below threshold"
            answer = ""

        citations = build_citations(top_chunks, cited_ids) if answer else []

        if refused and not answer:
            display_answer = (
                f"I cannot provide a grounded answer. Reason: {refusal_reason}"
            )
        else:
            display_answer = answer

        cost = cost_from_tokens(
            sum(p[0] for p in token_parts),
            sum(p[1] for p in token_parts),
        )

        result = QueryResult(
            query=question,
            answer=display_answer,
            confidence=confidence,
            citations=citations,
            chunks=top_chunks,
            scores=scores,
            latency=merge_latency(lat_parts),
            cost=cost,
            refused=refused,
            refusal_reason=refusal_reason,
        )
        self.logger.log_query(result)
        return result


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()