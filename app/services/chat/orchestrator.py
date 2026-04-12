"""ReAct chat orchestrator — single-loop agent with native tool calling.

Replaces the 4-stage pipeline (Planner → Executor → NumericFactsExtractor → Synthesizer)
with a single ReAct agent loop where the LLM directly calls tools and
generates the final response.

Architecture:
    User → [System Prompt + History + User Message]
         → LLM decides: call tools? or answer?
         → If tools: execute → feed results back → LLM decides again
         → If answer: return to user

Benefits:
    - 1 LLM call path (vs 2 before), iterating until done
    - LLM sees raw tool data — no lossy NumericFactsExtractor middleman
    - No schema sync problem — tool_registry.py is single source of truth
    - LLM self-heals: if data is missing, it calls more tools or estimates
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from datetime import timedelta, timezone
from typing import Any

import httpx

from app.core.config import settings
from app.core.http_client import get_http_client
from app.domain.entities.chat_orchestrator import (
    ChatCitation,
    ChatOrchestrateRequest,
    ChatOrchestrateResponse,
    ThreadSummaryRequest,
    ThreadSummaryResponse,
)
from app.services.chat.llm_client import LLMClient, LLMToolResponse, ToolCallInfo, _accumulate_usage
from app.services.chat.prompts.agent_prompt import AGENT_SYSTEM_PROMPT
from app.services.chat.tool_registry import (
    LOCAL_TOOL_NAME,
    MARKET_DATA_TOOLS,
    PERSONAL_FINANCE_TOOL_NAMES,
    RAG_TOOL_NAME,
    TOOL_DEFINITIONS,
)
from app.services.chat.tracing import RequestTrace
from app.services.chat.utils.vietnamese_text import sanitize_user_facing_message
from app.services.chat.valuation_engine import compute_fair_value
from app.services.market_data_tool_client import MarketDataToolClient
from app.services.rag_retrieval_service import RagRetrievalService

logger = logging.getLogger(__name__)

# Safety valve — prevent infinite tool-calling loops.
MAX_AGENT_ITERATIONS = 6

# Max characters per tool result to prevent context overflow.
MAX_TOOL_RESULT_CHARS = 6000


class ChatOrchestrator:
    """ReAct agent orchestrator — single LLM loop with tool calling."""

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._tool_client = MarketDataToolClient()
        self._rag_service = RagRetrievalService()
        self._debug_log = bool(settings.CHAT_DEBUG_LOG_PROMPTS)

        # Filter out RAG tool if RAG is disabled.
        if bool(settings.CHAT_RAG_ENABLED):
            self._tool_defs = TOOL_DEFINITIONS
        else:
            self._tool_defs = [
                t for t in TOOL_DEFINITIONS
                if t["function"]["name"] != RAG_TOOL_NAME
            ]

    async def orchestrate(self, request: ChatOrchestrateRequest) -> ChatOrchestrateResponse:
        """Run the ReAct agent loop."""
        trace = RequestTrace()

        # Build initial messages.
        messages = self._build_messages(request)

        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        rag_chunks: list[dict[str, Any]] = []
        accumulated_usage: dict[str, int] = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        }
        final_content = ""

        # ── ReAct loop ────────────────────────────────────────────────
        for iteration in range(MAX_AGENT_ITERATIONS):
            with trace.step(f"agent_turn_{iteration}") as step:
                response: LLMToolResponse = await self._llm.call_with_tools(
                    messages=messages,
                    tools=self._tool_defs,
                    stage=f"agent_turn_{iteration}",
                )
                _accumulate_usage(accumulated_usage, response.usage)
                step.tokens = response.usage.get("total_tokens", 0)

            if response.tool_calls:
                # ── LLM wants to call tools ───────────────────────────
                # Append assistant message with tool_calls to conversation.
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(
                                    tc.arguments, ensure_ascii=False,
                                ),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })

                # Execute all tool calls in parallel.
                with trace.step(f"tools_{iteration}") as step:
                    results = await self._execute_tool_calls(response.tool_calls)
                    step.metadata = {
                        "tools": [tc.name for tc in response.tool_calls],
                        "ok": sum(1 for r in results if r.get("ok")),
                    }

                # Feed results back into the conversation.
                for tc, result in zip(response.tool_calls, results):
                    all_tool_calls.append({
                        "name": tc.name, "arguments": tc.arguments,
                    })
                    all_tool_results.append(result)

                    # Track RAG chunks for citations.
                    if tc.name == RAG_TOOL_NAME and result.get("ok"):
                        for chunk in (result.get("data") or []):
                            if isinstance(chunk, dict):
                                rag_chunks.append(chunk)

                    # Serialize tool result for LLM context.
                    result_data = result.get("data")
                    if not result.get("ok"):
                        result_data = {
                            "error": result.get("error_message", "unknown error"),
                        }
                    result_content = json.dumps(
                        result_data, ensure_ascii=False, default=str,
                    )
                    if len(result_content) > MAX_TOOL_RESULT_CHARS:
                        result_content = (
                            result_content[:MAX_TOOL_RESULT_CHARS]
                            + "...[truncated]"
                        )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })
            else:
                # ── LLM returned final answer ─────────────────────────
                final_content = response.content or ""
                break
        else:
            # Max iterations exhausted.
            if not final_content:
                final_content = (
                    "Xin lỗi, tôi cần thêm thời gian để phân tích. "
                    "Bạn có thể hỏi lại được không?"
                )

        # ── Build response ────────────────────────────────────────────
        assistant_message = sanitize_user_facing_message(final_content.strip())
        if not assistant_message:
            assistant_message = "Xin lỗi, tôi chưa thể trả lời lúc này."

        ticker, year = self._extract_context(all_tool_calls)
        citations = self._pick_citations(rag_chunks)
        needs_clarification = self._detect_clarification(assistant_message)

        cost_usd = self._llm.estimate_cost(
            accumulated_usage["input_tokens"],
            accumulated_usage["output_tokens"],
        )

        context_update: dict[str, Any] = {}
        if ticker:
            context_update["last_ticker"] = ticker
        if year:
            context_update["last_year"] = year

        trace.log_summary()

        return ChatOrchestrateResponse(
            assistant_message=assistant_message,
            needs_clarification=needs_clarification,
            clarification_question=(
                assistant_message if needs_clarification else None
            ),
            provider="deepseek",
            model=self._llm.model,
            input_tokens=accumulated_usage["input_tokens"],
            output_tokens=accumulated_usage["output_tokens"],
            total_tokens=accumulated_usage["total_tokens"],
            cost_usd=cost_usd,
            latency_ms=trace.total_ms,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            citations=[ChatCitation(**c) for c in citations],
            context_update=context_update,
        )

    # ── Message building ──────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        request: ChatOrchestrateRequest,
    ) -> list[dict[str, Any]]:
        """Build the initial message array for the agent."""
        system_content = AGENT_SYSTEM_PROMPT

        vn_tz = timezone(timedelta(hours=7))
        now_vn = datetime.datetime.now(vn_tz)
        system_content += (
            f"\n\n--- THÔNG TIN NGỮ CẢNH HỆ THỐNG ---\n"
            f"Thời gian hiện tại: {now_vn.strftime('%Y-%m-%dT%H:%M:%S.%f%z')} "
            f"({now_vn.strftime('%A, %d/%m/%Y')})\n"
            f"USER_ID của người dùng hiện tại: {request.user_id}\n"
            f"LUÔN TUÂN THỦ NGÀY GIỜ NÀY. Nếu user hỏi 'hôm qua', hãy lùi 1 ngày so với ngày hiện tại này."
        )
        if request.context_summary and request.context_summary.strip():
            system_content += (
                f"\n\nContext từ cuộc hội thoại trước: "
                f"{request.context_summary}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        # Add recent chat history (only user/assistant turns).
        for msg in request.last_messages[-8:]:
            if msg.role in ("user", "assistant"):
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Add current user message.
        messages.append({
            "role": "user",
            "content": request.user_message,
        })

        return messages

    # ── Tool execution ────────────────────────────────────────────────

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCallInfo],
    ) -> list[dict[str, Any]]:
        """Execute tool calls in parallel, routing to the appropriate service."""

        async def _unknown_tool(name: str) -> dict[str, Any]:
            return {
                "name": name, "ok": False, "data": None,
                "error_code": "UNKNOWN_TOOL",
                "error_message": f"Tool '{name}' is not registered",
                "source_refs": [],
            }

        tasks: list[Any] = []
        for tc in tool_calls:
            if tc.name == RAG_TOOL_NAME:
                tasks.append(self._execute_rag_tool(tc))
            elif tc.name == LOCAL_TOOL_NAME:
                tasks.append(self._execute_local_tool(tc))
            elif tc.name in PERSONAL_FINANCE_TOOL_NAMES:
                tasks.append(self._execute_personal_finance_tool(tc))
            elif tc.name in MARKET_DATA_TOOLS:
                tasks.append(
                    self._tool_client.execute_tool_call(tc.name, tc.arguments)
                )
            else:
                tasks.append(_unknown_tool(tc.name))

        return list(await asyncio.gather(*tasks))

    async def _execute_rag_tool(
        self,
        tc: ToolCallInfo,
    ) -> dict[str, Any]:
        """Execute RAG retrieval as a tool call."""
        ticker = (tc.arguments.get("ticker") or "").strip().upper()
        query = (tc.arguments.get("query") or "").strip()

        if not ticker or not query:
            return {
                "name": RAG_TOOL_NAME, "ok": False, "data": None,
                "error_code": "INVALID_ARGS",
                "error_message": "ticker and query are required",
                "source_refs": [],
            }

        try:
            chunks = await self._rag_service.retrieve(
                query=query, ticker=ticker, years=None,
            )
            # Truncate chunks for token efficiency.
            compact = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "source_title": c.get("source_title"),
                    "page_number": c.get("page_number"),
                    "text": (c.get("text") or "")[:1200],
                }
                for c in (chunks or [])[:6]
            ]
            return {
                "name": RAG_TOOL_NAME, "ok": True, "data": compact,
                "error_code": None, "error_message": None,
                "source_refs": [],
            }
        except Exception as exc:
            logger.exception("RAG retrieval failed: %s", exc)
            return {
                "name": RAG_TOOL_NAME, "ok": False, "data": None,
                "error_code": "RAG_ERROR",
                "error_message": str(exc)[:200],
                "source_refs": [],
            }

    async def _execute_local_tool(
        self,
        tc: ToolCallInfo,
    ) -> dict[str, Any]:
        """Execute compute_fair_value: self-fetch data + deterministic computation."""
        try:
            if tc.name != LOCAL_TOOL_NAME:
                return {
                    "name": tc.name, "ok": False, "data": None,
                    "error_code": "UNKNOWN_LOCAL",
                    "error_message": f"Unknown local tool: {tc.name}",
                    "source_refs": [],
                }

            symbol = (tc.arguments.get("symbol") or "").strip().upper()
            if not symbol:
                return {
                    "name": tc.name, "ok": False, "data": None,
                    "error_code": "INVALID_ARGS",
                    "error_message": "symbol is required",
                    "source_refs": [],
                }

            target_year = tc.arguments.get("target_year")

            # Step 1: Fetch all data from backend in parallel.
            inputs = await self._fetch_valuation_inputs(symbol, target_year)
            if "error" in inputs and len(inputs) == 1:
                return {
                    "name": tc.name, "ok": False, "data": inputs,
                    "error_code": "DATA_ERROR",
                    "error_message": inputs["error"],
                    "source_refs": [],
                }

            # Step 2: Deterministic computation (pure Python, no I/O).
            result_data = compute_fair_value(inputs)

            is_error = "error" in result_data and len(result_data) == 1
            return {
                "name": tc.name,
                "ok": not is_error,
                "data": result_data,
                "error_code": "COMPUTE_ERROR" if is_error else None,
                "error_message": result_data.get("error") if is_error else None,
                "source_refs": [],
            }
        except Exception as exc:
            logger.exception("Local tool '%s' failed: %s", tc.name, exc)
            return {
                "name": tc.name, "ok": False, "data": None,
                "error_code": "LOCAL_TOOL_ERROR",
                "error_message": str(exc)[:200],
                "source_refs": [],
            }
    async def _execute_personal_finance_tool(
        self,
        tc: ToolCallInfo,
    ) -> dict[str, Any]:
        """Execute personal finance tools: call Spring Boot internal API."""
        user_id = (tc.arguments.get("user_id") or "").strip()
        if not user_id:
            return {
                "name": tc.name, "ok": False, "data": None,
                "error_code": "INVALID_ARGS",
                "error_message": "user_id is required",
                "source_refs": [],
            }

        try:
            base_url = (settings.JAVA_BACKEND_URL or "http://localhost:8080/api/internal").rstrip("/")
            internal_key = (settings.INTERNAL_API_KEY or "").strip()

            headers: dict[str, str] = {}
            if internal_key:
                headers["X-Internal-Api-Key"] = internal_key

            timeout = httpx.Timeout(max(5, int(settings.CHAT_TOOL_TIMEOUT_SECONDS)))
            client = get_http_client()

            if tc.name == "get_personal_finance_report":
                response = await client.get(
                    f"{base_url}/transaction/finance-report",
                    params={"userId": user_id},
                    headers=headers,
                    timeout=timeout,
                )
            elif tc.name == "get_user_transaction_context":
                response = await client.get(
                    f"{base_url}/transaction/user-context",
                    params={"userId": user_id},
                    headers=headers,
                    timeout=timeout,
                )
            elif tc.name == "add_transaction":
                body = {
                    "amount": tc.arguments.get("amount"),
                    "type": tc.arguments.get("type"),
                    "categoryId": tc.arguments.get("categoryId"),
                    "accountId": tc.arguments.get("accountId"),
                    "note": tc.arguments.get("note", ""),
                    "transactionDate": tc.arguments.get("transactionDate"),
                }
                response = await client.post(
                    f"{base_url}/transaction/add-transaction",
                    params={"userId": user_id},
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
            elif tc.name == "get_user_budgets":
                response = await client.get(
                    f"{base_url}/budget/budgets",
                    params={"userId": user_id},
                    headers=headers,
                    timeout=timeout,
                )
            elif tc.name == "add_budget":
                body: dict[str, Any] = {
                    "categoryId": tc.arguments.get("categoryId"),
                    "targetAmount": tc.arguments.get("targetAmount"),
                    "startDate": tc.arguments.get("startDate"),
                    "endDate": tc.arguments.get("endDate"),
                }
                if "isRecurring" in tc.arguments and tc.arguments["isRecurring"] is not None:
                    body["isRecurring"] = tc.arguments.get("isRecurring")
                rsd = tc.arguments.get("recurringStartDate")
                if isinstance(rsd, str) and rsd.strip():
                    body["recurringStartDate"] = rsd.strip()
                response = await client.post(
                    f"{base_url}/budget/create-budget",
                    params={"userId": user_id},
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
            else:
                return {
                    "name": tc.name, "ok": False, "data": None,
                    "error_code": "UNKNOWN_PF_TOOL",
                    "error_message": f"Unknown personal finance tool: {tc.name}",
                    "source_refs": [],
                }

            if response.status_code < 200 or response.status_code >= 300:
                return {
                    "name": tc.name, "ok": False, "data": None,
                    "error_code": f"HTTP_{response.status_code}",
                    "error_message": response.text[:300],
                    "source_refs": [],
                }

            data = response.json()
            return {
                "name": tc.name, "ok": True, "data": data,
                "error_code": None, "error_message": None,
                "source_refs": [],
            }
        except Exception as exc:
            logger.exception("Personal finance tool '%s' failed: %s", tc.name, exc)
            return {
                "name": tc.name, "ok": False, "data": None,
                "error_code": "PERSONAL_FINANCE_ERROR",
                "error_message": str(exc)[:200],
                "source_refs": [],
            }

    async def _fetch_valuation_inputs(
        self,
        symbol: str,
        target_year: int | None = None,
    ) -> dict[str, Any]:
        """Fetch all data needed for valuation from backend in parallel.

        Returns a dict ready to be passed to compute_fair_value().
        """
        today = datetime.date.today()
        five_years_ago = today.replace(year=today.year - 5)
        yr = target_year or today.year

        # Fetch 5 data sources in parallel.
        metrics_r, financial_r, daily_val_r, live_r, forecast_r = (
            await asyncio.gather(
                self._tool_client.execute_tool_call(
                    "get_company_metrics", {"symbol": symbol},
                ),
                self._tool_client.execute_tool_call(
                    "get_company_financial_series",
                    {"symbol": symbol, "annualLimit": 5},
                ),
                self._tool_client.execute_tool_call(
                    "get_company_daily_valuations",
                    {
                        "symbol": symbol,
                        "startDate": five_years_ago.isoformat(),
                        "endDate": today.isoformat(),
                    },
                ),
                self._tool_client.execute_tool_call(
                    "get_company_live_valuation_snapshot", {"symbol": symbol},
                ),
                self._tool_client.execute_tool_call(
                    "get_company_forecast",
                    {"symbol": symbol, "targetYear": yr},
                ),
            )
        )

        # Check for critical failures.
        errors: list[str] = []
        for label, result in [
            ("metrics", metrics_r),
            ("financial", financial_r),
            ("live", live_r),
        ]:
            if not result.get("ok"):
                errors.append(
                    f"{label}: {result.get('error_message', 'unknown')}"
                )
        if errors:
            return {"error": f"Không lấy được dữ liệu: {'; '.join(errors)}"}

        # ── Extract fields ──
        overview = (metrics_r.get("data") or {}).get("overview") or {}

        # Profit history from financial series.
        fin_data = financial_r.get("data") or {}
        raw_entries = fin_data.get("nonBank") or fin_data.get("bank") or []
        profit_history = [
            {
                "year": item["year"],
                "profit_after_tax": item.get("profitAfterTax", 0),
            }
            for item in raw_entries
            if isinstance(item, dict) and "year" in item
        ]

        # Daily valuations summary.
        daily_summary = (
            (daily_val_r.get("data") or {}).get("summary") or {}
        )

        # Live price.
        live_data = live_r.get("data") or {}

        # Forecast (non-critical — may not have forecast for future year).
        forecast_data = forecast_r.get("data") or {}

        return {
            "eps": overview.get("eps", 0),
            "bvps": overview.get("bvps", 0),
            "roe": overview.get("roe", 0),
            "live_price": live_data.get("livePriceVnd", 0),
            "profit_history": profit_history,
            "industry_icb_code": overview.get("industryIcbCode"),
            "industry_label": overview.get("industryLabel"),
            "median_pe": daily_summary.get("pe_median"),
            "median_pb": daily_summary.get("pb_median"),
            "forecast_profit": forecast_data.get("profit_pred"),
            "cplh": overview.get("cplh", 0),
            "symbol": symbol,
            "target_year": yr,
            "company_name": overview.get("companyName", symbol),
        }

    # ── Context extraction ────────────────────────────────────────────

    @staticmethod
    def _extract_context(
        tool_calls: list[dict[str, Any]],
    ) -> tuple[str | None, int | None]:
        """Extract ticker and year from the tool calls made during the session."""
        tickers: set[str] = set()
        years: list[int] = []

        for call in tool_calls:
            args = call.get("arguments", {})
            symbol = args.get("symbol") or args.get("ticker")
            if isinstance(symbol, str) and symbol.strip():
                tickers.add(symbol.strip().upper())
            target_year = args.get("targetYear")
            if isinstance(target_year, int):
                years.append(target_year)

        ticker = next(iter(tickers)) if len(tickers) == 1 else None
        year = max(years) if years else None
        return ticker, year

    @staticmethod
    def _detect_clarification(message: str) -> bool:
        """Heuristic: detect if the response is asking the user for clarification."""
        stripped = message.strip()
        if stripped.endswith("?"):
            return True
        patterns = [
            "bạn muốn", "bạn có thể cho", "vui lòng cho tôi biết",
            "mã cổ phiếu nào", "bạn đang hỏi về",
        ]
        lower = stripped.lower()
        return any(p in lower for p in patterns)

    @staticmethod
    def _pick_citations(
        rag_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pick top citations from RAG chunks."""
        if not rag_chunks:
            return []
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

    # ── Thread summary (unchanged from old architecture) ──────────────

    async def summarize_thread(
        self,
        request: ThreadSummaryRequest,
    ) -> ThreadSummaryResponse:
        """Summarize chat thread context."""
        trace = RequestTrace()

        recent_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
            }
            for msg in request.recent_messages
        ]

        user_prompt = (
            f"Summary cũ: {request.existing_summary or '{}'}\n"
            f"Recent messages: {json.dumps(recent_messages, ensure_ascii=False)}"
        )

        with trace.step("thread_summary") as s:
            summary_json, usage = await self._llm.call_json(
                system_prompt=(
                    "Bạn là module context summary cho chat đầu tư. "
                    "Output ONLY JSON với keys: current_ticker,current_period,"
                    "user_goal,facts_confirmed,open_questions,decisions. "
                    "Nếu thiếu ticker/year thì để null. "
                    "Danh sách fields dạng list phải ngắn gọn."
                ),
                user_prompt=user_prompt,
                max_output_tokens=400,
                stage="thread_summary",
            )
            s.tokens = usage.get("total_tokens", 0)

        current_ticker = summary_json.get("current_ticker")
        current_period = summary_json.get("current_period")
        context_summary = json.dumps(
            summary_json, ensure_ascii=False, separators=(",", ":"),
        )

        trace.log_summary()

        return ThreadSummaryResponse(
            context_summary=context_summary,
            current_ticker=(
                str(current_ticker).upper()
                if isinstance(current_ticker, str) and current_ticker.strip()
                else None
            ),
            current_period=(
                int(current_period) if isinstance(current_period, int) else None
            ),
            provider="deepseek",
            model=self._llm.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=self._llm.estimate_cost(
                usage["input_tokens"], usage["output_tokens"],
            ),
            latency_ms=trace.total_ms,
        )


