"""Per-query token and cost tracking."""

from __future__ import annotations

from app.generation.llm import cost_from_tokens
from app.models import CostBreakdown


def accumulate_cost(parts: list[tuple[int, int]]) -> CostBreakdown:
    inp = sum(p[0] for p in parts)
    out = sum(p[1] for p in parts)
    return cost_from_tokens(inp, out)