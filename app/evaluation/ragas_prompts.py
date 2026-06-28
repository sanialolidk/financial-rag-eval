"""RAGAS-aligned prompt templates (claim decomposition + verification)."""

# Step 1: decompose answer into atomic factual statements (RAGAS faithfulness stage 1)
STATEMENT_GENERATOR_SYSTEM = """You are extracting factual statements from an answer to evaluate faithfulness.
Given a QUESTION and an ANSWER, break the answer into distinct, self-contained factual statements.

Rules:
- Each statement must be a single verifiable fact (numbers, dates, entities, guidance, ratios).
- Do NOT include opinions, hedging, or citation brackets like [chunk_id].
- Do NOT duplicate statements.
- If the answer refuses or says it cannot answer, return an empty list.

Return JSON only:
{"statements": ["statement 1", "statement 2"]}"""

# Step 2: verify each statement against retrieved context (RAGAS faithfulness stage 2)
STATEMENT_VERIFIER_SYSTEM = """You verify whether a factual statement can be directly inferred from the given CONTEXT.
This is a strict entailment check used in RAG evaluation (RAGAS faithfulness).

Return JSON only:
{"verdict": "yes" or "no", "reason": "one sentence"}

Verdict "yes" ONLY if the context explicitly contains or clearly implies the statement.
Verdict "no" if the statement adds numbers, dates, entities, or claims not in the context."""

# Answer relevancy (RAGAS-style): does the answer address the question?
ANSWER_RELEVANCE_SYSTEM = """You evaluate answer relevancy for a RAG system.
Given a QUESTION and an ANSWER, score how well the answer addresses what was asked.

Return JSON only:
{"score": 0.0, "reason": "brief explanation"}

Scoring guide:
- 1.0: fully addresses the question with appropriate scope
- 0.5: partially addresses the question or includes irrelevant content
- 0.0: does not address the question or is a refusal unrelated to the ask"""

# Hallucination: list unsupported claims (inverse of faithfulness, explicit for alerts)
HALLUCINATION_SYSTEM = """You detect hallucinations in RAG answers using strict context grounding.

Given CONTEXT and ANSWER:
1. List every factual claim in the answer (numbers, dates, company facts, guidance).
2. Mark claims NOT supported by the context.

Return JSON only:
{
  "hallucination_detected": true or false,
  "unsupported_claims": ["claim text", ...],
  "details": "summary"
}

hallucination_detected is true if ANY unsupported factual claim exists."""