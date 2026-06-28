"""Hallucination detection: flag claims not grounded in retrieved chunks."""

from __future__ import annotations

import json
import re

from app.generation.llm import judge
from app.models import RetrievedChunk

HALLUCINATION_SYSTEM = """You detect hallucinations in RAG answers.
Given CONTEXT chunks and an ANSWER, list any factual claims in the answer that are NOT supported by the context.

Return JSON only:
{
  "hallucination_detected": true/false,
  "unsupported_claims": ["claim1", "claim2"],
  "details": "brief summary"
}

A claim is unsupported if it states a number, date, ratio, guidance, or company fact not present in context."""


def detect_hallucinations(answer: str, chunks: list[RetrievedChunk]) -> tuple[bool, str, int, int]:
    if not answer or not chunks:
        return False, "", 0, 0

    context = "\n\n".join(f"[{rc.chunk.chunk_id}]\n{rc.chunk.text[:1200]}" for rc in chunks)
    user = f"ANSWER:\n{answer}\n\nCONTEXT:\n{context}"
    resp = judge(HALLUCINATION_SYSTEM, user)

    try:
        m = re.search(r"\{.*\}", resp.text, re.S)
        if not m:
            return False, resp.text[:300], resp.input_tokens, resp.output_tokens
        data = json.loads(m.group())
        detected = bool(data.get("hallucination_detected", False))
        claims = data.get("unsupported_claims", [])
        details = str(data.get("details", ""))
        if claims:
            details = f"{details} | Unsupported: {claims}"
        return detected, details, resp.input_tokens, resp.output_tokens
    except Exception:
        return False, resp.text[:300], resp.input_tokens, resp.output_tokens