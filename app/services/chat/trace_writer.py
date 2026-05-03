"""Per-request chat trace writer.

Dumps the full pydantic-ai message log (system prompt, user, tool calls,
tool returns, final answer) plus the inbound request and outbound
response into one JSON file per chat turn.

Filename format: {YYYYMMDD-HHMMSS}_{thread_id}_{user_id}_{run_id}.json

Disabled by default. Toggle via CHAT_TRACE_ENABLED=true in .env.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.chat import ChatOrchestrateRequest, ChatOrchestrateResponse

logger = logging.getLogger(__name__)


class ChatTraceWriter:
    """Stateless writer — reads config at call-time so env changes
    (and tests that reload settings) pick up immediately."""

    @property
    def enabled(self) -> bool:
        return bool(settings.CHAT_TRACE_ENABLED)

    @property
    def _dir(self) -> Path:
        path = Path(settings.CHAT_TRACE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(
        self,
        *,
        request: ChatOrchestrateRequest,
        response: ChatOrchestrateResponse,
        all_messages_json: bytes | None,
        latency_ms: int,
        error: Exception | None = None,
        rag_traces: list[dict] | None = None,
    ) -> Path | None:
        """Write one trace file per orchestration turn. Best-effort.

        Returns the file path on success, None if disabled or write failed.
        """
        if not self.enabled:
            return None

        try:
            timestamp = datetime.now()
            run_id = response.model[:6] if response.model else "unknown"
            filename = (
                f"{timestamp.strftime('%Y%m%d-%H%M%S-%f')[:-3]}"
                f"_{_safe(request.thread_id)}"
                f"_{_safe(request.user_id)}"
                f"_{run_id}.json"
            )
            path = self._dir / filename

            messages_decoded: Any = None
            if all_messages_json:
                try:
                    messages_decoded = json.loads(all_messages_json.decode("utf-8"))
                except Exception:
                    messages_decoded = "<failed to decode all_messages_json>"

            payload = {
                "ts": timestamp.isoformat(timespec="milliseconds"),
                "latency_ms": latency_ms,
                "error": _serialize_error(error) if error else None,
                "rag_traces": rag_traces or [],
                "request": {
                    "thread_id": request.thread_id,
                    "user_id": request.user_id,
                    "user_message": request.user_message,
                    "context_summary": request.context_summary or "",
                    "history_messages": [
                        {"role": m.role, "content": m.content}
                        for m in (request.last_messages or [])
                    ],
                },
                "agent_messages": messages_decoded,
                "response": {
                    "assistant_message": response.assistant_message,
                    "needs_clarification": response.needs_clarification,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "cost_usd": response.cost_usd,
                    "tool_calls": response.tool_calls,
                    "tool_results": response.tool_results,
                    "citations": [c.model_dump() for c in response.citations],
                    "context_update": response.context_update,
                },
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("[CHAT][TRACE] %s (%d bytes)", path.name, path.stat().st_size)
            return path
        except Exception as exc:
            logger.warning("[CHAT][TRACE] failed to write trace: %s", exc)
            return None


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in (name or "anon"))[:32]


def _serialize_error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}
