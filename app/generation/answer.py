"""Grounded answer generation with mandatory citations and refusal path."""

from __future__ import annotations

import re

from app.generation.llm import LLMResponse, chat
from app.models import CostBreakdown, RetrievedChunk

SYSTEM_PROMPT = """You are a financial document analyst. Answer ONLY using the provided SEC filing excerpts.

Rules:
1. Every factual claim MUST cite a chunk ID in brackets, e.g. [AAPL_10-K_2024-09-28_p12_w0_abc123].
2. If the excerpts do not contain enough information, respond EXACTLY with: INSUFFICIENT_CONTEXT
3. Do NOT use outside knowledge. Do NOT guess numbers, dates, or guidance.
4. Prefer direct quotes for numeric facts (revenue, margins, guidance).
5. Keep answers concise (3-6 sentences)."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for rc in chunks:
        c = rc.chunk
        blocks.append(
            f"--- CHUNK {c.chunk_id} ---\n"
            f"Source: {c.citation}\n"
            f"Relevance: {rc.chunk_relevance_score:.3f}\n"
            f"{c.text}\n"
        )
    return "\n".join(blocks)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str], LLMResponse, bool, str]:
    """
    Returns: answer, cited_chunk_ids, llm_response, refused, refusal_reason
    """
    if not chunks:
        return "", [], LLMResponse("", 0, 0), True, "No relevant chunks retrieved."

    avg_rel = sum(rc.chunk_relevance_score for rc in chunks) / len(chunks)
    if avg_rel < 0.35:
        return "", [], LLMResponse("", 0, 0), True, "Retrieved chunks have low relevance scores."

    user = f"Question: {query}\n\nContext excerpts:\n{_format_context(chunks)}"
    llm_resp = chat(SYSTEM_PROMPT, user)

    if llm_resp.text.strip() == "INSUFFICIENT_CONTEXT":
        return "", [], llm_resp, True, "LLM determined context is insufficient."

    cited = re.findall(r"\[([A-Za-z0-9_]+)\]", llm_resp.text)
    valid_ids = {rc.chunk.chunk_id for rc in chunks}
    cited_valid = [c for c in cited if c in valid_ids]

    if not cited_valid:
        return llm_resp.text, [], llm_resp, True, "Answer lacks valid chunk citations."

    return llm_resp.text, cited_valid, llm_resp, False, ""


def build_citations(chunks: list[RetrievedChunk], cited_ids: list[str]) -> list[dict]:
    id_set = set(cited_ids)
    out = []
    for rc in chunks:
        if rc.chunk.chunk_id in id_set:
            out.append({
                "chunk_id": rc.chunk.chunk_id,
                "citation": rc.chunk.citation,
                "rerank_score": rc.rerank_score,
                "chunk_relevance_score": rc.chunk_relevance_score,
                "text_preview": rc.chunk.text[:500],
            })
    return out


def compute_confidence(
    chunks: list[RetrievedChunk],
    faithfulness: float,
    hallucination_detected: bool,
) -> float:
    if not chunks:
        return 0.0
    rel = sum(rc.chunk_relevance_score for rc in chunks) / len(chunks)
    base = 0.45 * rel + 0.45 * faithfulness + 0.10 * min(1.0, max(rc.rerank_score for rc in chunks) / 10.0)
    if hallucination_detected:
        base *= 0.5
    return round(max(0.0, min(1.0, base)), 3)