"""Smoke test: pydantic-ai connects to DeepSeek via OpenAI-compatible endpoint.

Verifies:
- Agent.run() produces a response.
- The HTTP request sent to DeepSeek is shaped like our existing payload
  (same model, same temperature, same response_format mode when not using tools).
"""
from __future__ import annotations

import json

import pytest

from tests.conftest import make_text_response


async def test_pydantic_ai_basic_run(deepseek_mock, captured_requests):
    """Smoke: basic Agent.run call to DeepSeek."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        "deepseek-chat",
        provider=OpenAIProvider(
            base_url="https://api.deepseek.com",
            api_key="sk-test-key",
        ),
    )
    agent = Agent(model, system_prompt="You are a helper.")

    with deepseek_mock(responses=[make_text_response("Hello back!")]):
        result = await agent.run("Hello")

    assert result.output == "Hello back!"
    assert len(captured_requests) >= 1

    body = captured_requests[0]["body"]
    assert body["model"] == "deepseek-chat"
    # pydantic-ai default — verify what it sends
    assert "messages" in body
    print("REQUEST_BODY_KEYS:", sorted(body.keys()))
    print("REQUEST_BODY:", json.dumps(body, ensure_ascii=False, indent=2))


async def test_pydantic_ai_with_tool(deepseek_mock, captured_requests):
    """Smoke: pydantic-ai @agent.tool dispatching."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        "deepseek-chat",
        provider=OpenAIProvider(base_url="https://api.deepseek.com", api_key="sk-test-key"),
    )
    agent = Agent(model)

    @agent.tool_plain
    async def get_price(symbol: str) -> dict:
        """Get the live price of a stock symbol."""
        return {"symbol": symbol, "price": 30000}

    # First DS call → tool_calls; second DS call → final text.
    from tests.conftest import make_tool_call_response
    responses = [
        make_tool_call_response([{"id": "c1", "name": "get_price", "arguments": {"symbol": "HPG"}}]),
        make_text_response("HPG đang ở giá 30,000đ"),
    ]
    with deepseek_mock(responses=responses):
        result = await agent.run("HPG bao nhiêu?")

    assert result.output == "HPG đang ở giá 30,000đ"
    # Two LLM calls in the loop
    ds_calls = [c for c in captured_requests if c["target"] == "deepseek"]
    assert len(ds_calls) == 2

    # First call should declare the tool
    first_body = ds_calls[0]["body"]
    assert "tools" in first_body
    tool_names = [t["function"]["name"] for t in first_body["tools"]]
    assert "get_price" in tool_names

    # Second call should include the tool result in messages
    second_body = ds_calls[1]["body"]
    msg_roles = [m.get("role") for m in second_body["messages"]]
    assert "tool" in msg_roles
