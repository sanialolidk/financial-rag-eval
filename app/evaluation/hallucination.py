"""Hallucination detection using RAGAS-style unsupported-claim listing."""

from __future__ import annotations

import json
import re

from app.evaluation.numeric_grounding import find_ungrounded_numerics
from app.evaluation.ragas_prompts import HALLUCINATION_SYSTEM
from app.evaluation.metrics import (
    _extract_statements_heuristic,
    _verify_statement_cross_encoder,
    _verify_statement_llm,
)
from app.generation.llm import judge
from app.models import RetrievedChunk


def _parse_json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    return json.loads(m.group())


def detect_hallucinations(
    answer: str,
    chunks: list[RetrievedChunk],
    use_llm: bool = True,
) -> tuple[bool, str, int, int]:
    if not answer or not chunks:
        return False, "", 0, 0

    # Financial-domain guardrail: dollar amounts and fiscal years must appear in context
    numeric_hits = find_ungrounded_numerics(answer, chunks)
    if numeric_hits:
        details = f"Ungrounded numerics: {numeric_hits}"
        return True, details, 0, 0

    context = "\n\n".join(f"[{rc.chunk.chunk_id}]\n{rc.chunk.text[:1200]}" for rc in chunks)

    if use_llm:
        try:
            user = f"ANSWER:\n{answer}\n\nCONTEXT:\n{context}"
            resp = judge(HALLUCINATION_SYSTEM, user)
            data = _parse_json_object(resp.text)
            detected = bool(data.get("hallucination_detected", False))
            claims = data.get("unsupported_claims", [])
            details = str(data.get("details", ""))
            if claims:
                details = f"{details} | Unsupported: {claims}"
            return detected, details, resp.input_tokens, resp.output_tokens
        except RuntimeError:
            use_llm = False

    # Fallback: reuse faithfulness claim verification
    statements = _extract_statements_heuristic(answer)
    unsupported: list[str] = []
    ctx_blob = "\n\n".join(rc.chunk.text[:1500] for rc in chunks)
    for stmt in statements:
        ok = _verify_statement_cross_encoder(stmt, chunks)
        if not ok:
            unsupported.append(stmt)

    detected = len(unsupported) > 0
    details = f"{len(unsupported)} unsupported claims (cross-encoder check)"
    if unsupported:
        details += f": {unsupported[:2]}"
    return detected, details, 0, 0