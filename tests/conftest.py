"""Shared test fixtures.

Strategy: mock DeepSeek + backend HTTP, capture request bodies as baseline
JSON snapshots. After migration, run again and diff against baseline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import respx
import httpx

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-key")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-key")
os.environ.setdefault("JAVA_BACKEND_URL", "http://backend.test/api/internal")
os.environ.setdefault("CHAT_RAG_ENABLED", "false")
os.environ.setdefault("VOYAGE_API_KEY", "pa-test-key")
os.environ.setdefault("VOYAGE_EMBED_BASE_URL", "http://voyage.test/v1")
os.environ.setdefault("VOYAGE_RERANK_URL", "http://voyage.test/v1/rerank")
os.environ.setdefault("CHAT_QDRANT_URL", "")

BASELINES_DIR = Path(__file__).parent / "baselines"
BASELINES_DIR.mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def reset_http_client():
    """Force a fresh httpx client per test (respx mocks at transport level)."""
    from app.core import http_client as hc
    hc._client = None
    yield
    hc._client = None


@pytest.fixture
def captured_requests() -> list[dict[str, Any]]:
    """Collected request payloads (deepseek + backend) for snapshotting."""
    return []


@pytest.fixture
def deepseek_mock(captured_requests):
    """Mock the DeepSeek API. Returns a context manager that records request bodies.

    Usage:
        with deepseek_mock(responses=[...]) as rec:
            await service.something()
        # rec[0] is the first request body sent to DeepSeek
    """
    class _DSContext:
        def __init__(self, responses: list[dict[str, Any]]):
            self._responses = list(responses)
            self._mock = None

        def __enter__(self):
            self._mock = respx.mock(assert_all_called=False)
            self._mock.start()

            def _handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8"))
                captured_requests.append({"target": "deepseek", "body": body})
                if self._responses:
                    resp = self._responses.pop(0)
                else:
                    resp = _make_text_response("OK")
                return httpx.Response(200, json=resp)

            self._mock.post("https://api.deepseek.com/chat/completions").mock(
                side_effect=_handler
            )
            return captured_requests

        def __exit__(self, *exc):
            self._mock.stop()
            return False

    return _DSContext


@pytest.fixture
def backend_mock(captured_requests):
    """Mock the Spring Boot backend. Records all calls."""
    class _BEContext:
        def __init__(self, routes: dict[str, Any]):
            self._routes = routes
            self._mock = None

        def __enter__(self):
            self._mock = respx.mock(
                base_url="http://backend.test/api/internal",
                assert_all_called=False,
            )
            self._mock.start()

            def _make_handler(payload):
                def _handler(request: httpx.Request) -> httpx.Response:
                    try:
                        body = json.loads(request.content.decode("utf-8")) if request.content else None
                    except Exception:
                        body = None
                    captured_requests.append({
                        "target": "backend",
                        "method": request.method,
                        "url": str(request.url),
                        "body": body,
                    })
                    return httpx.Response(200, json=payload)
                return _handler

            for path, payload in self._routes.items():
                self._mock.route(path__regex=path).mock(side_effect=_make_handler(payload))
            return captured_requests

        def __exit__(self, *exc):
            self._mock.stop()
            return False

    return _BEContext


def _make_text_response(content: str, tool_calls: list | None = None) -> dict:
    """Build an OpenAI-compatible response."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def make_json_response(payload: dict) -> dict:
    """Build a DeepSeek response whose content is a JSON string of payload."""
    return _make_text_response(json.dumps(payload, ensure_ascii=False))


def make_tool_call_response(tool_calls: list[dict]) -> dict:
    """Build a DeepSeek response with tool_calls.

    tool_calls: list of {"id": "...", "name": "...", "arguments": {...}}
    """
    formatted = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
            },
        }
        for tc in tool_calls
    ]
    return _make_text_response("", tool_calls=formatted)


def make_text_response(content: str) -> dict:
    """Build a DeepSeek response with plain text content."""
    return _make_text_response(content)


def save_baseline(name: str, data: Any) -> None:
    path = BASELINES_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def load_baseline(name: str) -> Any:
    path = BASELINES_DIR / f"{name}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_for_diff(body: dict) -> dict:
    """Strip volatile fields before comparing."""
    body = dict(body)
    # Common volatile bits — none for now since we use temperature=0.
    return body
