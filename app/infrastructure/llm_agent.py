"""Shared pydantic-ai model factory for DeepSeek.

Builds an OpenAIChatModel pointed at DeepSeek's OpenAI-compatible endpoint.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.config import settings


@lru_cache(maxsize=1)
def get_deepseek_model() -> OpenAIChatModel:
    """Cached DeepSeek model instance — share connection state across services."""
    return OpenAIChatModel(
        settings.DEEPSEEK_MODEL,
        provider=OpenAIProvider(
            base_url=(settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com").rstrip("/"),
            api_key=(settings.DEEPSEEK_API_KEY or "").strip(),
        ),
    )


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """DeepSeek pricing per 1M tokens."""
    in_cost = (input_tokens / 1_000_000.0) * settings.CHAT_DEEPSEEK_INPUT_PRICE_PER_1M
    out_cost = (output_tokens / 1_000_000.0) * settings.CHAT_DEEPSEEK_OUTPUT_PRICE_PER_1M
    return round(in_cost + out_cost, 8)
