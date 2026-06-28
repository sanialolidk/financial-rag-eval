"""OpenAI client wrapper with token/cost accounting."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from app.config import settings
from app.models import CostBreakdown


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=settings.openai_api_key)


def chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> LLMResponse:
    model = model or settings.openai_chat_model
    resp = _client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0].message.content or ""
    usage = resp.usage
    return LLMResponse(
        text=choice.strip(),
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def judge(system: str, user: str) -> LLMResponse:
    return chat(system, user, model=settings.openai_judge_model, temperature=0.0)


def cost_from_tokens(inp: int, out: int) -> CostBreakdown:
    in_cost = (inp / 1_000_000) * settings.input_cost_per_1m
    out_cost = (out / 1_000_000) * settings.output_cost_per_1m
    return CostBreakdown(
        input_tokens=inp,
        output_tokens=out,
        input_cost_usd=in_cost,
        output_cost_usd=out_cost,
        total_cost_usd=in_cost + out_cost,
    )