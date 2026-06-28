"""RAG evaluation metrics: faithfulness, chunk recall, response relevance."""

from __future__ import annotations

import json
import re

from app.generation.llm import judge
from app.models import EvalQuery, RetrievedChunk

FAITHFULNESS_SYSTEM = """You are an evaluation judge for RAG systems.
Score whether the ANSWER is fully supported by the CONTEXT chunks.
Return JSON only: {"score": 0.0-1.0, "reason": "brief explanation"}
- 1.0 = every claim in the answer appears in the context
- 0.0 = answer contradicts or invents facts not in context"""

RELEVANCE_SYSTEM = """You are an evaluation judge.
Score whether the ANSWER addresses the QUESTION.
Return JSON only: {"score": 0.0-1.0, "reason": "brief explanation"}
- 1.0 = directly and completely answers the question
- 0.0 = off-topic or does not answer"""


def _parse_judge_json(text: str, default: float = 0.5) -> tuple[float, str]:
    try:
        # Extract first JSON object
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return default, text[:200]
        data = json.loads(m.group())
        score = float(data.get("score", default))
        return max(0.0, min(1.0, score)), str(data.get("reason", ""))
    except Exception:
        return default, text[:200]


def score_faithfulness(query: str, answer: str, chunks: list[RetrievedChunk]) -> tuple[float, str, int, int]:
    context = "\n\n".join(f"[{rc.chunk.chunk_id}]\n{rc.chunk.text[:1500]}" for rc in chunks)
    user = f"QUESTION: {query}\n\nANSWER: {answer}\n\nCONTEXT:\n{context}"
    resp = judge(FAITHFULNESS_SYSTEM, user)
    score, reason = _parse_judge_json(resp.text)
    return score, reason, resp.input_tokens, resp.output_tokens


def score_response_relevance(query: str, answer: str) -> tuple[float, str, int, int]:
    user = f"QUESTION: {query}\n\nANSWER: {answer}"
    resp = judge(RELEVANCE_SYSTEM, user)
    score, reason = _parse_judge_json(resp.text)
    return score, reason, resp.input_tokens, resp.output_tokens


def score_chunk_recall(
    eval_item: EvalQuery | None,
    chunks: list[RetrievedChunk],
    answer: str,
) -> float:
    """
    Chunk recall: does retrieved context contain expected answer signals?
    Uses eval dataset keywords when available; otherwise checks citation overlap.
    """
    if not chunks:
        return 0.0

    combined = " ".join(rc.chunk.text.lower() for rc in chunks)
    answer_lower = answer.lower()

    if eval_item and eval_item.expected_answer_contains:
        hits = sum(1 for kw in eval_item.expected_answer_contains if kw.lower() in combined)
        return hits / len(eval_item.expected_answer_contains)

    # Fallback: fraction of answer content words found in chunks
    words = [w for w in re.findall(r"[a-z]{4,}", answer_lower) if w not in _STOPWORDS]
    if not words:
        return 0.5
    found = sum(1 for w in words if w in combined)
    return found / len(words)


_STOPWORDS = {
    "that", "this", "with", "from", "were", "been", "have", "their", "which",
    "would", "about", "into", "than", "them", "also", "other", "some", "such",
    "only", "over", "after", "most", "made", "between", "through", "during",
    "including", "according", "company", "fiscal", "year", "quarter", "million",
    "billion", "percent", "compared", "prior", "same", "period",
}