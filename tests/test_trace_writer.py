"""Verify trace writer dumps a JSON file per request when enabled."""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from app.models.chat import ChatOrchestrateRequest
from tests.conftest import make_text_response


@pytest.fixture
def trace_dir(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="finflow_trace_")
    from app.core.config import settings
    monkeypatch.setattr(settings, "CHAT_TRACE_ENABLED", True)
    monkeypatch.setattr(settings, "CHAT_TRACE_DIR", tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


async def test_trace_written_on_success(trace_dir, deepseek_mock, captured_requests):
    """Successful request → 1 JSON file with full message log + response."""
    from app.services.chat.orchestrator import ChatOrchestrator

    orchestrator = ChatOrchestrator()
    request = ChatOrchestrateRequest(
        thread_id="thread-trace",
        user_id="u-trace",
        user_message="Xin chào",
        last_messages=[],
    )

    with deepseek_mock(responses=[make_text_response("Chào bạn!")]):
        await orchestrator.orchestrate(request)

    files = sorted(os.listdir(trace_dir))
    assert len(files) == 1, f"Expected 1 trace file, got {files}"
    assert files[0].endswith(".json")
    assert "thread-trace" in files[0]
    assert "u-trace" in files[0]

    payload = json.loads(open(os.path.join(trace_dir, files[0]), encoding="utf-8").read())
    assert payload["request"]["user_message"] == "Xin chào"
    assert payload["response"]["assistant_message"]
    assert payload["agent_messages"] is not None
    # Verify message log shape — pydantic-ai dumps a list of model requests/responses.
    assert isinstance(payload["agent_messages"], list)
    assert len(payload["agent_messages"]) >= 2  # at least the request + response


async def test_trace_skipped_when_disabled(monkeypatch, deepseek_mock):
    """Default (trace disabled) → no files written."""
    from app.core.config import settings
    tmp = tempfile.mkdtemp(prefix="finflow_trace_off_")
    monkeypatch.setattr(settings, "CHAT_TRACE_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_DIR", tmp)

    from app.services.chat.orchestrator import ChatOrchestrator
    try:
        orchestrator = ChatOrchestrator()
        request = ChatOrchestrateRequest(
            thread_id="t", user_id="u",
            user_message="Hello", last_messages=[],
        )
        with deepseek_mock(responses=[make_text_response("Hi")]):
            await orchestrator.orchestrate(request)
        assert os.listdir(tmp) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
