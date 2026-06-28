"""FastAPI backend for financial RAG + eval."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import CORPUS_COMPANIES, settings
from app.evaluation.harness import EvalHarness, load_eval_dataset
from app.observability.logger import QueryLogger
from app.pipeline.query import RAGPipeline, get_pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    run_eval: bool = True


class EvalRunRequest(BaseModel):
    limit: Optional[int] = Field(None, ge=1, le=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipeline = get_pipeline()
    except Exception as exc:
        app.state.pipeline = None
        app.state.pipeline_error = str(exc)
    yield


app = FastAPI(
    title="FinRAG Eval API",
    description="Financial document RAG with evaluation harness",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline: Optional[RAGPipeline] = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return {
            "status": "degraded",
            "error": getattr(app.state, "pipeline_error", "pipeline not loaded"),
        }
    return {
        "status": "ok",
        "chroma_chunks": pipeline.chroma.count,
        "bm25_chunks": pipeline.bm25.size,
        "companies": list(CORPUS_COMPANIES.keys()),
    }


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    pipeline: Optional[RAGPipeline] = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, detail=getattr(app.state, "pipeline_error", "Pipeline unavailable"))
    result = pipeline.query(req.question, run_eval=req.run_eval)
    return result.to_dict()


@app.post("/eval/run")
def run_eval(req: EvalRunRequest) -> dict[str, Any]:
    pipeline: Optional[RAGPipeline] = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, detail=getattr(app.state, "pipeline_error", "Pipeline unavailable"))

    dataset = load_eval_dataset()
    if req.limit:
        dataset = dataset[: req.limit]

    query_results = []
    for item in dataset:
        query_results.append(pipeline.query(item.question, run_eval=True, eval_item=item))

    harness = EvalHarness(pipeline.logger)
    summary = harness.run_eval_set(query_results)
    return {
        "summary": summary,
        "results": [r.to_dict() for r in query_results],
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    logger = QueryLogger()
    recent = logger.recent_queries(20)
    latest_eval = logger.latest_eval_summary()
    return {"recent_queries": recent, "latest_eval_summary": latest_eval}


@app.get("/config")
def config_public() -> dict[str, Any]:
    return {
        "hybrid_top_k": settings.hybrid_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "faithfulness_alert_threshold": settings.faithfulness_alert_threshold,
        "chunk_recall_alert_threshold": settings.chunk_recall_alert_threshold,
    }