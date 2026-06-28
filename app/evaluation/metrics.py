"""RAG evaluation metrics: RAGAS-style faithfulness, chunk recall, response relevance."""

from __future__ import annotations

import json
import re

from app.evaluation.ragas_prompts import (
    ANSWER_RELEVANCE_SYSTEM,
    STATEMENT_GENERATOR_SYSTEM,
    STATEMENT_VERIFIER_SYSTEM,
)
from app.generation.llm import judge
from app.models import EvalQuery, RetrievedChunk
from app.retrieval.reranker import get_cross_encoder

_ENTAILMENT_THRESHOLD = 0.25  # cross-encoder logit threshold for local fallback


def _parse_json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    return json.loads(m.group())


def _extract_statements_llm(query: str, answer: str) -> tuple[list[str], int, int]:
    user = f"QUESTION:\n{query}\n\nANSWER:\n{answer}"
    resp = judge(STATEMENT_GENERATOR_SYSTEM, user)
    try:
        data = _parse_json_object(resp.text)
        statements = [s.strip() for s in data.get("statements", []) if s.strip()]
        return statements, resp.input_tokens, resp.output_tokens
    except Exception:
        return _extract_statements_heuristic(answer), resp.input_tokens, resp.output_tokens


def _extract_statements_heuristic(answer: str) -> list[str]:
    """Fallback when LLM unavailable: sentence-split factual lines."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    out = []
    for p in parts:
        p = re.sub(r"\[[^\]]+\]", "", p).strip()
        if len(p) < 12:
            continue
        if re.search(r"\d|percent|%|\$|billion|million|margin|revenue|income", p, re.I):
            out.append(p)
        elif len(out) < 3:
            out.append(p)
    return out[:8]


def _verify_statement_llm(statement: str, context: str) -> tuple[bool, int, int]:
    user = f"CONTEXT:\n{context}\n\nSTATEMENT:\n{statement}"
    resp = judge(STATEMENT_VERIFIER_SYSTEM, user)
    try:
        data = _parse_json_object(resp.text)
        return data.get("verdict", "no").lower() == "yes", resp.input_tokens, resp.output_tokens
    except Exception:
        return False, resp.input_tokens, resp.output_tokens


def _verify_statement_cross_encoder(statement: str, chunks: list[RetrievedChunk]) -> bool:
    """Local entailment proxy: max cross-encoder score across chunks."""
    if not statement or not chunks:
        return False
    model = get_cross_encoder()
    pairs = [(statement, rc.chunk.text[:2000]) for rc in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    return float(max(scores)) >= _ENTAILMENT_THRESHOLD


def score_faithfulness(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    use_llm: bool = True,
) -> tuple[float, str, int, int]:
    """
    RAGAS faithfulness: supported_claims / total_claims.
    Uses LLM claim decomposition + verification when use_llm=True and API key set.
    Falls back to heuristic claims + cross-encoder entailment otherwise.
    """
    if not answer or not chunks:
        return 0.0, "empty answer or context", 0, 0

    tokens_in, tokens_out = 0, 0
    if use_llm:
        try:
            statements, ti, to = _extract_statements_llm(query, answer)
            tokens_in += ti
            tokens_out += to
        except RuntimeError:
            statements = _extract_statements_heuristic(answer)
            use_llm = False
    else:
        statements = _extract_statements_heuristic(answer)

    if not statements:
        return 1.0, "no factual claims to verify (refusal or empty)", tokens_in, tokens_out

    context = "\n\n".join(rc.chunk.text[:1500] for rc in chunks)
    supported = 0
    reasons: list[str] = []

    for stmt in statements:
        if use_llm:
            try:
                ok, ti, to = _verify_statement_llm(stmt, context)
                tokens_in += ti
                tokens_out += to
            except RuntimeError:
                ok = _verify_statement_cross_encoder(stmt, chunks)
        else:
            ok = _verify_statement_cross_encoder(stmt, chunks)
        if ok:
            supported += 1
        else:
            reasons.append(stmt[:80])

    score = supported / len(statements)
    reason = f"{supported}/{len(statements)} claims supported"
    if reasons:
        reason += f"; unsupported: {reasons[:2]}"
    return round(score, 3), reason, tokens_in, tokens_out


def score_response_relevance(query: str, answer: str, use_llm: bool = True) -> tuple[float, str, int, int]:
    if not answer:
        return 0.0, "empty answer", 0, 0
    if not use_llm:
        # Heuristic: question keyword overlap
        q_words = set(re.findall(r"[a-z]{4,}", query.lower()))
        a_words = set(re.findall(r"[a-z]{4,}", answer.lower()))
        overlap = len(q_words & a_words) / max(len(q_words), 1)
        return round(min(1.0, overlap * 1.5), 3), "keyword overlap heuristic", 0, 0

    user = f"QUESTION:\n{query}\n\nANSWER:\n{answer}"
    try:
        resp = judge(ANSWER_RELEVANCE_SYSTEM, user)
        data = _parse_json_object(resp.text)
        score = float(data.get("score", 0.5))
        return max(0.0, min(1.0, score)), str(data.get("reason", "")), resp.input_tokens, resp.output_tokens
    except RuntimeError:
        return score_response_relevance(query, answer, use_llm=False)


def score_chunk_recall(
    eval_item: EvalQuery | None,
    chunks: list[RetrievedChunk],
    answer: str,
) -> float:
    if not chunks:
        return 0.0

    combined = " ".join(rc.chunk.text.lower() for rc in chunks)

    if eval_item and eval_item.expected_answer_contains:
        hits = sum(1 for kw in eval_item.expected_answer_contains if kw.lower() in combined)
        return round(hits / len(eval_item.expected_answer_contains), 3)

    answer_lower = answer.lower()
    words = [w for w in re.findall(r"[a-z]{4,}", answer_lower) if w not in _STOPWORDS]
    if not words:
        return 0.5
    found = sum(1 for w in words if w in combined)
    return round(found / len(words), 3)


_STOPWORDS = {
    "that", "this", "with", "from", "were", "been", "have", "their", "which",
    "would", "about", "into", "than", "them", "also", "other", "some", "such",
    "only", "over", "after", "most", "made", "between", "through", "during",
    "including", "according", "company", "fiscal", "year", "quarter", "million",
    "billion", "percent", "compared", "prior", "same", "period", "cannot",
    "provide", "grounded", "reason", "answer",
}