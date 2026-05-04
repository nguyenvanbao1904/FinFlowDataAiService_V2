"""ReAct chat orchestrator powered by pydantic-ai.

The agent loop, tool routing, message-building and validation are handled
by pydantic-ai. This file only:
- Builds AppDeps per request
- Runs the agent
- Maps pydantic-ai's message log into the wire response shape consumed by
  Spring Boot (assistant_message, tool_calls, tool_results, citations,
  context_update).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits

from app.core.config import settings
from app.infrastructure.llm_agent import estimate_cost, get_deepseek_model
from app.infrastructure.market_data_client import MarketDataToolClient
from app.infrastructure.rag_client import RagRetrievalService
from app.models.chat import (
    ChatCitation,
    ChatOrchestrateRequest,
    ChatOrchestrateResponse,
    ThreadSummaryRequest,
    ThreadSummaryResponse,
)
from app.services.chat.agent_tools import AppDeps, build_chat_agent, _is_cfo_context
from app.services.chat.trace_writer import ChatTraceWriter
from app.services.chat.utils.json_io import parse_llm_json

logger = logging.getLogger(__name__)

_RAG_TOOL_NAME = "search_annual_reports"
_HISTORY_LIMIT = 8
_CONTEXT_SUMMARY_MAX = 2000

# Hard caps per chat turn — protects against runaway tool loops and
# adversarial cost burn. Tuned for: ReAct loop ≤ 8 LLM calls, ≤ 12 tool
# calls (1 compute_fair_value can fan out to 5 backend reads internally).
_USAGE_LIMITS = UsageLimits(request_limit=8, tool_calls_limit=12)


class ChatOrchestrator:
    """Thin wrapper around a pydantic-ai Agent."""

    def __init__(self) -> None:
        self._agent = build_chat_agent()
        self._market_client = MarketDataToolClient()
        self._rag_service = RagRetrievalService()
        self._trace = ChatTraceWriter()

    async def orchestrate(self, request: ChatOrchestrateRequest) -> ChatOrchestrateResponse:
        deps = AppDeps(
            user_id=request.user_id,
            market_client=self._market_client,
            rag_service=self._rag_service,
            cfo_context=_is_cfo_context(request.user_message),
        )

        trace = self._trace.enabled
        if trace:
            history_len = len(request.last_messages or [])
            logger.info(
                "[TRACE] ▶ thread=%s user=%s | msg=%r | history=%d",
                request.thread_id[:12], request.user_id[:8],
                request.user_message[:80], history_len,
            )

        t0 = time.perf_counter()
        try:
            result = await self._agent.run(
                request.user_message,
                message_history=_build_history(request),
                deps=deps,
                usage_limits=_USAGE_LIMITS,
            )
        except Exception as exc:
            logger.exception("Agent run failed")
            err_response = _error_response(exc, latency_ms=_elapsed_ms(t0))
            if self._trace.enabled:
                self._trace.write(
                    request=request, response=err_response,
                    all_messages_json=None,
                    latency_ms=_elapsed_ms(t0), error=exc,
                )
            return err_response

        latency_ms = _elapsed_ms(t0)
        tool_calls, tool_results, rag_chunks = _extract_tool_io(result)

        if trace:
            _log_tool_io(tool_calls, tool_results)
        usage = result.usage()
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        # Sanitization happens inside the agent via @agent.output_validator.
        message = (result.output or "").strip() or "Xin lỗi, tôi chưa thể trả lời lúc này."
        ticker, year = _extract_context(tool_calls)
        context_update: dict[str, Any] = {}
        if ticker:
            context_update["last_ticker"] = ticker
        if year:
            context_update["last_year"] = year

        # Prefer the actual model name reported by the API (e.g. "deepseek-v4-flash")
        # over the config alias ("deepseek-chat") for accurate logging/billing.
        model_id = _actual_model_name(result) or (
            self._agent.model.model_name if self._agent.model else settings.DEEPSEEK_MODEL
        )
        provider = model_id.split("-")[0]

        response = ChatOrchestrateResponse(
            assistant_message=message,
            needs_clarification=_needs_clarification(message),
            clarification_question=message if _needs_clarification(message) else None,
            provider=provider,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=estimate_cost(input_tokens, output_tokens),
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            tool_results=tool_results,
            citations=[ChatCitation(**c) for c in _pick_citations(rag_chunks)],
            context_update=context_update,
        )

        if self._trace.enabled:
            logger.info(
                "[TRACE] ◀ %dms | tokens=%d+%d | tools=%d | answer=%r",
                latency_ms, response.input_tokens, response.output_tokens,
                len(tool_calls), (response.assistant_message or "")[:120],
            )
            try:
                messages_json = result.all_messages_json()
            except Exception:
                messages_json = None
            self._trace.write(
                request=request, response=response,
                all_messages_json=messages_json,
                latency_ms=latency_ms,
                rag_traces=self._rag_service.pop_retrieve_traces(),
            )

        return response

    async def summarize_thread(self, request: ThreadSummaryRequest) -> ThreadSummaryResponse:
        return await summarize_thread(request, self._agent.model.model_name if self._agent.model else settings.DEEPSEEK_MODEL)


# ── Pure helpers (module-level so they're easy to test) ─────────────


def _build_history(request: ChatOrchestrateRequest) -> list[ModelRequest | ModelResponse]:
    history: list[ModelRequest | ModelResponse] = []
    if request.context_summary and request.context_summary.strip():
        summary = request.context_summary.strip()[:_CONTEXT_SUMMARY_MAX]
        history.append(ModelRequest(parts=[
            UserPromptPart(content=f"Context từ cuộc hội thoại trước: {summary}")
        ]))
    current = request.user_message.strip()
    for msg in request.last_messages[-_HISTORY_LIMIT:]:
        # Backend includes the current message in last_messages; skip it to
        # avoid sending the same user turn twice (pydantic-ai appends it again).
        if msg.role == "user" and msg.content.strip() == current:
            continue
        if msg.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=msg.content)]))
        elif msg.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=msg.content)]))
    return history


def _extract_tool_io(result: Any) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk the message log → (tool_calls, tool_results, rag_chunks)."""
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    rag_chunks: list[dict] = []

    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                args = part.args
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                tool_calls.append({"name": part.tool_name, "arguments": args or {}})
            elif isinstance(part, ToolReturnPart):
                parsed = parse_llm_json(part.content) if isinstance(part.content, str) else part.content
                tool_results.append({
                    "name": part.tool_name,
                    "ok": True,
                    "data": parsed,
                    "error_code": None,
                    "error_message": None,
                    "source_refs": [],
                })
                if part.tool_name == _RAG_TOOL_NAME and isinstance(parsed, list):
                    rag_chunks.extend(c for c in parsed if isinstance(c, dict))
            elif isinstance(part, RetryPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                tool_results.append({
                    "name": part.tool_name or "unknown",
                    "ok": False,
                    "data": None,
                    "error_code": "TOOL_ERROR",
                    "error_message": content[:300],
                    "source_refs": [],
                })

    return tool_calls, tool_results, rag_chunks


def _extract_context(tool_calls: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    tickers: set[str] = set()
    years: list[int] = []
    for call in tool_calls:
        args = call.get("arguments") or {}
        symbol = args.get("symbol") or args.get("ticker")
        if isinstance(symbol, str) and symbol.strip():
            tickers.add(symbol.strip().upper())
        target_year = args.get("targetYear")
        if isinstance(target_year, int):
            years.append(target_year)
    return (
        ",".join(sorted(tickers)) if tickers else None,
        max(years) if years else None,
    )


def _needs_clarification(message: str) -> bool:
    """Detect when the agent is BLOCKED waiting for user input.

    A long answer that happens to end with a polite "Bạn có muốn xem thêm...?"
    is NOT a clarification — the agent already delivered value. Clarification
    means the agent could not proceed without more info.
    """
    s = message.strip()
    # Long answers (>240 chars) are deliveries, not clarifications, even if
    # they end with a follow-up question.
    if len(s) > 240:
        return False
    lower = s.lower()
    blocking_patterns = (
        "vui lòng cho tôi biết",
        "mã cổ phiếu nào",
        "bạn đang hỏi về mã",
        "bạn muốn hỏi về",
        "cho tôi biết mã",
        "chưa rõ mã cổ phiếu",
    )
    if any(p in lower for p in blocking_patterns):
        return True
    # Short message ending with a question and no concrete content yet.
    if s.endswith("?") and len(s) < 160:
        return True
    return False


def _pick_citations(rag_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": r.get("chunk_id"),
            "source_title": r.get("source_title"),
            "page_number": r.get("page_number"),
            "score": r.get("score"),
        }
        for r in rag_chunks[:5]
        if isinstance(r, dict)
    ]


def _actual_model_name(result: Any) -> str | None:
    """Extract the real model name from the last ModelResponse in the message log."""
    name = None
    for msg in result.all_messages():
        model_name = getattr(msg, "model_name", None)
        if model_name:
            name = model_name
    return name


def _log_tool_io(tool_calls: list[dict], tool_results: list[dict]) -> None:
    """Print each tool call + result summary to stdout when trace is enabled."""
    for i, call in enumerate(tool_calls):
        args_preview = json.dumps(call.get("arguments") or {}, ensure_ascii=False)[:120]
        logger.info("[TRACE]   tool[%d] → %s(%s)", i + 1, call["name"], args_preview)
    for i, res in enumerate(tool_results):
        if res.get("ok"):
            data_preview = json.dumps(res.get("data"), ensure_ascii=False, default=str)[:160]
            logger.info("[TRACE]   result[%d] ← %s OK: %s", i + 1, res["name"], data_preview)
        else:
            logger.info(
                "[TRACE]   result[%d] ← %s ERROR: %s",
                i + 1, res.get("name", "?"), res.get("error_message", "")[:120],
            )


def _error_response(exc: Exception, latency_ms: int) -> ChatOrchestrateResponse:
    return ChatOrchestrateResponse(
        assistant_message=f"Xin lỗi, đã xảy ra lỗi: {type(exc).__name__}. Vui lòng thử lại.",
        provider="deepseek",
        model=settings.DEEPSEEK_MODEL,
        latency_ms=latency_ms,
    )


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


# ── Thread summary ────────────────────────────────────────────────────


async def summarize_thread(request: ThreadSummaryRequest, model_id: str) -> ThreadSummaryResponse:
    """Summarize chat thread context — uses a one-shot agent."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    recent = [
        {"role": m.role, "content": m.content, "created_at": m.created_at}
        for m in request.recent_messages
    ]
    user_prompt = (
        f"Summary cũ: {request.existing_summary or '{}'}\n"
        f"Recent messages: {json.dumps(recent, ensure_ascii=False)}"
    )

    agent = Agent(
        get_deepseek_model(),
        system_prompt=(
            "Bạn là module context summary cho chat đầu tư. "
            "Output ONLY JSON với keys: current_ticker,current_period,"
            "user_goal,facts_confirmed,open_questions,decisions. "
            "Nếu thiếu ticker/year thì để null. "
            "Danh sách fields dạng list phải ngắn gọn."
        ),
        model_settings=OpenAIChatModelSettings(
            temperature=0.0,
            max_tokens=400,
            extra_body={"response_format": {"type": "json_object"}},
        ),
        output_type=str,
    )

    t0 = time.perf_counter()
    result = await agent.run(user_prompt)
    latency_ms = _elapsed_ms(t0)

    parsed = parse_llm_json(result.output)
    summary_json = parsed if isinstance(parsed, dict) else {}

    usage = result.usage()
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    current_ticker = summary_json.get("current_ticker")
    current_period = summary_json.get("current_period")

    return ThreadSummaryResponse(
        context_summary=json.dumps(summary_json, ensure_ascii=False, separators=(",", ":")),
        current_ticker=(
            str(current_ticker).upper()
            if isinstance(current_ticker, str) and current_ticker.strip()
            else None
        ),
        current_period=current_period if isinstance(current_period, int) else None,
        provider=model_id.split("-")[0],
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=estimate_cost(input_tokens, output_tokens),
        latency_ms=latency_ms,
    )
