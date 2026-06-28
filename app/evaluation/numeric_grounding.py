"""Detect dollar amounts and fiscal years in answers that are absent from context."""

from __future__ import annotations

import re

from app.models import RetrievedChunk

_DOLLAR_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M|bn|mn)?",
    re.IGNORECASE,
)
_FISCAL_YEAR_RE = re.compile(r"\bfiscal\s+(?:year\s+)?(\d{4})\b", re.IGNORECASE)


def _normalize_amount(raw: str, unit: str) -> str:
    num = raw.replace(",", "")
    unit = (unit or "").lower()
    if unit in {"b", "bn", "billion"}:
        return f"{num}b"
    if unit in {"m", "mn", "million"}:
        return f"{num}m"
    return num


def _extract_amounts(text: str) -> list[str]:
    found: list[str] = []
    for m in _DOLLAR_RE.finditer(text):
        found.append(_normalize_amount(m.group(1), m.group(2) or ""))
    return found


def _extract_fiscal_years(text: str) -> list[str]:
    return _FISCAL_YEAR_RE.findall(text)


def find_ungrounded_numerics(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Return unsupported dollar amounts / fiscal years in answer vs retrieved chunks."""
    context = " ".join(rc.chunk.text for rc in chunks).lower()
    unsupported: list[str] = []

    for amt in _extract_amounts(answer):
        # Check raw number appears in context (e.g. 10.32, 75.2, 75,200)
        raw_num = amt.rstrip("bm")
        if raw_num not in context and raw_num.replace(".", "") not in context.replace(",", ""):
            # Also check if any close variant exists
            if not any(part in context for part in (raw_num, amt)):
                unsupported.append(f"${raw_num} ({amt})")

    answer_years = set(_extract_fiscal_years(answer))
    if answer_years:
        context_years = set(_extract_fiscal_years(context))
        for y in sorted(answer_years):
            if context_years and y not in context_years:
                unsupported.append(f"fiscal year {y}")

    return unsupported