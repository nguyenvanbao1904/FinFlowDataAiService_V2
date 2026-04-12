"""Request-scoped tracing for chat orchestration pipeline.

Provides structured, per-request trace logging with step-level timing,
token counts, and metadata. Prevents log interleaving during concurrent requests.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """A single step in the request trace."""
    name: str
    duration_ms: int = 0
    tokens: int = 0
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    """Collects timing and metadata for a single orchestration request.

    Usage:
        trace = RequestTrace()
        with trace.step("planner") as s:
            result = await planner.plan(...)
            s.tokens = usage["total_tokens"]
            s.metadata["intent"] = plan.intent_type
        ...
        trace.log_summary()
    """
    request_id: str = field(default_factory=lambda: uuid4().hex[:12])
    steps: list[TraceStep] = field(default_factory=list)
    _start_time: float = field(default_factory=time.perf_counter, repr=False)

    class _StepContext:
        """Context manager for timing a trace step."""
        def __init__(self, trace: RequestTrace, name: str) -> None:
            self._trace = trace
            self._step = TraceStep(name=name)
            self._t0 = 0.0

        def __enter__(self) -> TraceStep:
            self._t0 = time.perf_counter()
            return self._step

        def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
            self._step.duration_ms = int((time.perf_counter() - self._t0) * 1000)
            if exc_type is not None:
                self._step.status = f"error:{exc_type.__name__}"
            self._trace.steps.append(self._step)

    def step(self, name: str) -> _StepContext:
        """Create a traced step context manager."""
        return self._StepContext(self, name)

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_ms": self.total_ms,
            "total_tokens": self.total_tokens,
            "steps": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "tokens": s.tokens,
                    "status": s.status,
                    **({"meta": s.metadata} if s.metadata else {}),
                }
                for s in self.steps
            ],
        }

    def log_summary(self) -> None:
        """Emit a structured log line with the full trace."""
        summary = self.to_dict()
        steps_str = " → ".join(
            f"{s.name}({s.duration_ms}ms)"
            for s in self.steps
        )
        logger.info(
            "[CHAT][TRACE][%s] %dms %dtok | %s",
            self.request_id,
            self.total_ms,
            self.total_tokens,
            steps_str,
        )
        # Structured log for log aggregation systems.
        logger.debug(
            "[CHAT][TRACE_DETAIL][%s] %s",
            self.request_id,
            summary,
        )
