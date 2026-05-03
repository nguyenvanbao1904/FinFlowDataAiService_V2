"""Chat agent tools — pydantic-ai @agent.tool wrappers.

Tool schemas (name, description, params) are auto-generated from Python
function signatures + docstrings. The agent loop, validation, and retry
are handled by pydantic-ai.

The orchestrator builds AppDeps per request and calls agent.run(); this
module owns the tool catalogue.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModelSettings

from app.core.config import settings
from app.core.http_client import get_http_client
from app.infrastructure.llm_agent import get_deepseek_model
from app.infrastructure.market_data_client import MarketDataToolClient
from app.infrastructure.rag_client import RagRetrievalService
from app.services.chat.prompts.agent_prompt import AGENT_SYSTEM_PROMPT
from app.services.chat.utils.vietnamese_text import sanitize_user_facing_message
from app.services.chat.valuation_engine import compute_fair_value as _compute_fair_value
from app.services.chat.valuation_inputs import fetch_valuation_inputs

logger = logging.getLogger(__name__)

_VN_TZ = datetime.timezone(datetime.timedelta(hours=7))


@dataclass
class AppDeps:
    """Per-request dependencies injected into every tool via RunContext."""
    user_id: str
    market_client: MarketDataToolClient
    rag_service: RagRetrievalService


def build_chat_agent() -> Agent[AppDeps, str]:
    """Build the chat agent with all tools registered."""
    agent = Agent[AppDeps, str](
        get_deepseek_model(),
        deps_type=AppDeps,
        output_type=str,
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=4096),
    )

    @agent.output_validator
    def _sanitize_output(_ctx: RunContext[AppDeps], output: str) -> str:
        """Strip markdown + technical jargon from the LLM's final answer.

        Runs as part of the agent loop; pydantic-ai treats the cleaned
        string as the canonical output (no need to clean again downstream).
        """
        cleaned = sanitize_user_facing_message((output or "").strip())
        return cleaned or "Xin lỗi, tôi chưa thể trả lời lúc này."

    @agent.system_prompt
    def _system_prompt(ctx: RunContext[AppDeps]) -> str:
        now = datetime.datetime.now(_VN_TZ)
        return (
            AGENT_SYSTEM_PROMPT
            + "\n\n--- THÔNG TIN NGỮ CẢNH HỆ THỐNG ---\n"
            + f"Thời gian hiện tại: {now.strftime('%Y-%m-%dT%H:%M:%S.%f%z')} "
            + f"({now.strftime('%A, %d/%m/%Y')})\n"
            + f"USER_ID của người dùng hiện tại: {ctx.deps.user_id}\n"
            + "LUÔN TUÂN THỦ NGÀY GIỜ NÀY. Nếu user hỏi 'hôm qua', "
            + "hãy lùi 1 ngày so với ngày hiện tại này."
        )

    _register_market_tools(agent)
    _register_personal_finance_tools(agent)
    _register_compute_tool(agent)
    if bool(settings.CHAT_RAG_ENABLED):
        _register_rag_tool(agent)
    return agent


# ── Helpers ───────────────────────────────────────────────────────────


async def _call_market(
    ctx: RunContext[AppDeps], tool_name: str, **arguments: Any,
) -> dict:
    args = {k: v for k, v in arguments.items() if v is not None}
    result = await ctx.deps.market_client.execute_tool_call(tool_name, args)
    if not result.get("ok"):
        error_msg = result.get("error_message") or "tool failed"
        # Distinguish recoverable errors (let the LLM retry with adjusted args)
        # from upstream failures (let the agent loop bubble up).
        code = result.get("error_code") or ""
        if code in ("INVALID_TOOL_ARGS", "HTTP_404"):
            raise ModelRetry(
                f"Tool {tool_name} không thực hiện được: {error_msg}. "
                f"Hãy kiểm tra lại tham số (ví dụ symbol có hợp lệ không) "
                f"hoặc dùng suggest_companies để tìm mã đúng."
            )
        raise RuntimeError(f"{tool_name}: {error_msg}")
    return result.get("data") or {}


async def _pf_request(
    ctx: RunContext[AppDeps],
    method: str,
    path: str,
    body: dict | None = None,
) -> dict:
    base_url = settings.JAVA_BACKEND_URL.rstrip("/")
    headers: dict[str, str] = {}
    internal_key = (settings.INTERNAL_API_KEY or "").strip()
    if internal_key:
        headers["X-Internal-Api-Key"] = internal_key
    timeout = httpx.Timeout(max(5, int(settings.CHAT_TOOL_TIMEOUT_SECONDS)))
    client = get_http_client()

    response = await client.request(
        method, f"{base_url}{path}",
        params={"userId": ctx.deps.user_id},
        headers=headers,
        json=body,
        timeout=timeout,
    )
    if response.status_code == 400:
        # Validation error → let the LLM read the message and retry with fixed args.
        raise ModelRetry(
            f"Backend từ chối yêu cầu (400): {response.text[:300]}. "
            f"Hãy kiểm tra lại các tham số và thử lại."
        )
    if response.status_code == 404:
        raise ModelRetry(
            "Không tìm thấy dữ liệu (404). Có thể tham số sai hoặc dữ liệu chưa tồn tại."
        )
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"HTTP_{response.status_code}: {response.text[:300]}")
    return response.json()


# ── Market data tools ─────────────────────────────────────────────────


def _register_market_tools(agent: Agent[AppDeps, str]) -> None:
    @agent.tool
    async def get_company_financial_series(
        ctx: RunContext[AppDeps],
        symbol: str,
        annualLimit: int | None = None,
        quarterlyLimit: int | None = None,
    ) -> dict:
        """Doanh thu, lợi nhuận, ROE, ROA, biên lãi theo năm/quý từ DB báo cáo tài chính. Backend ĐÃ TÍNH SẴN tăng trưởng YoY cho từng chỉ tiêu: yoyGrowth (LNST), yoyNetRevenue (DT thuần, non-bank), yoyCustomerLoan (cho vay KH, bank), yoyTotalOperatingIncome (TOI, bank), yoyNpl (nợ xấu, bank), yoyInventories (hàng tồn, non-bank). KHÔNG cần tự tính YoY — dùng trực tiếp giá trị đã có. Dùng khi cần phân tích sức khỏe tài chính, tăng trưởng, hiệu quả hoạt động. Mặc định annualLimit=3 nếu không chỉ định."""
        return await _call_market(ctx, "get_company_financial_series",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_daily_valuations(
        ctx: RunContext[AppDeps],
        symbol: str,
        startDate: str,
        endDate: str,
        summary: bool | None = None,
    ) -> dict:
        """Bảng tóm tắt PE/PB/PS hàng ngày (trung vị, trung bình, min, max). ĐÂY LÀ NGUỒN CHÍNH cho so sánh định giá lịch sử. Mặc định dùng 5 năm. Dữ liệu thực tế có thể 3-5 năm tùy mã. Backend tự tính toán thống kê, trả về summary gọn."""
        return await _call_market(ctx, "get_company_daily_valuations",
                                  symbol=symbol, startDate=startDate, endDate=endDate, summary=summary)

    @agent.tool
    async def get_company_live_valuation_snapshot(ctx: RunContext[AppDeps], symbol: str) -> dict:
        """Giá realtime + PE/PB/PS hiện tại + median lịch sử + nhãn đánh giá rẻ/đắt."""
        return await _call_market(ctx, "get_company_live_valuation_snapshot", symbol=symbol)

    @agent.tool
    async def get_company_forecast(
        ctx: RunContext[AppDeps], symbol: str, targetYear: int | None = None,
    ) -> dict:
        """Dự báo doanh thu & lợi nhuận cho năm mục tiêu, kèm top yếu tố ảnh hưởng. Dùng khi cần dự báo tương lai hoặc tính giá trị hợp lý."""
        return await _call_market(ctx, "get_company_forecast", symbol=symbol, targetYear=targetYear)

    @agent.tool
    async def get_company_metrics(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Overview gọn: EPS, BVPS, cổ phiếu lưu hành (CPLH), ROE, median PE/PB. CẦN THIẾT khi tính giá hợp lý (cần EPS, BVPS, CPLH). Đã loại bỏ dữ liệu thừa."""
        return await _call_market(ctx, "get_company_metrics",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_dividends(
        ctx: RunContext[AppDeps], symbol: str, annualLimit: int | None = None,
    ) -> dict:
        """Lịch sử cổ tức của công ty."""
        return await _call_market(ctx, "get_company_dividends", symbol=symbol, annualLimit=annualLimit)

    @agent.tool
    async def get_company_valuations(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None,
        startDate: str | None = None, endDate: str | None = None,
        showQuarterly: bool | None = None,
    ) -> dict:
        """PE/PB/PS theo quý từ DB (có thể tới 20+ năm từ FireAnt). Ưu tiên dùng get_company_daily_valuations nếu cần thống kê trung vị/trung bình."""
        return await _call_market(ctx, "get_company_valuations",
                                  symbol=symbol, annualLimit=annualLimit,
                                  startDate=startDate, endDate=endDate, showQuarterly=showQuarterly)

    @agent.tool
    async def get_company_cash_flows(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Dòng tiền hoạt động, đầu tư, tài chính theo năm/quý. Dùng khi hỏi về chất lượng lợi nhuận, CFO/FCF, lợi nhuận có chuyển thành tiền hay không, dòng tiền âm/dương, hoặc rủi ro dòng tiền."""
        return await _call_market(ctx, "get_company_cash_flows",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_analysis(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Phân tích chi tiết công ty (overview, tài chính, cổ đông). Ưu tiên dùng get_company_metrics nếu chỉ cần EPS/BVPS/CPLH."""
        return await _call_market(ctx, "get_company_analysis",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_market_data(
        ctx: RunContext[AppDeps], symbol: str,
        include: str | None = None,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Dữ liệu thị trường tổng quát (overview, financials, valuation, dividends)."""
        return await _call_market(ctx, "get_company_market_data",
                                  symbol=symbol, include=include,
                                  annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_industry_nodes(ctx: RunContext[AppDeps]) -> dict:
        """Danh sách toàn bộ ngành nghề trên thị trường."""
        return await _call_market(ctx, "get_industry_nodes")

    @agent.tool
    async def suggest_companies(
        ctx: RunContext[AppDeps], q: str, limit: int | None = None,
    ) -> dict:
        """Tìm kiếm/gợi ý công ty theo tên hoặc mã."""
        return await _call_market(ctx, "suggest_companies", q=q, limit=limit)

    @agent.tool
    async def get_company_industries(
        ctx: RunContext[AppDeps], symbols: list[str],
    ) -> dict:
        """Lấy ngành nghề của các mã cổ phiếu."""
        return await _call_market(ctx, "get_company_industries", symbols=symbols)


# ── Personal finance tools ────────────────────────────────────────────


def _register_personal_finance_tools(agent: Agent[AppDeps, str]) -> None:
    """user_id is taken from ctx.deps — never exposed to the LLM schema."""

    @agent.tool
    async def get_personal_finance_report(ctx: RunContext[AppDeps]) -> dict:
        """Lấy báo cáo tài chính cá nhân của người dùng: thu nhập, chi tiêu, tỷ lệ tiết kiệm, top danh mục chi tiêu, biến động theo tháng (4 tháng gần nhất). Dùng khi người dùng hỏi về tình hình thu chi, ngân sách, chi tiêu cá nhân, hoặc muốn tạo báo cáo tài chính cá nhân. Tool TỰ TÍNH TOÁN mọi số liệu. KHÔNG dùng tool này cho phân tích cổ phiếu hay công ty."""
        return await _pf_request(ctx, "GET", "/transaction/finance-report")

    @agent.tool
    async def get_user_transaction_context(ctx: RunContext[AppDeps]) -> dict:
        """Lấy danh sách danh mục giao dịch (categories) và tài khoản (accounts) của người dùng. BẮT BUỘC gọi tool này TRƯỚC khi gọi add_transaction để lấy đúng categoryId và accountId. Kết quả gồm: categories (id, name, type: INCOME/EXPENSE/SAVING), accounts (id, name, type, balance)."""
        return await _pf_request(ctx, "GET", "/transaction/user-context")

    @agent.tool
    async def add_transaction(
        ctx: RunContext[AppDeps],
        amount: float,
        type: Literal["INCOME", "EXPENSE", "SAVING"],
        categoryId: str,
        accountId: str,
        transactionDate: str,
        note: str = "",
    ) -> dict:
        """Thêm giao dịch mới cho người dùng. CHỈ gọi tool này SAU KHI đã xác nhận với người dùng. Trước khi gọi, BẮT BUỘC gọi get_user_transaction_context để lấy đúng categoryId và accountId. CHÚ Ý: transactionDate phải ở format ISO8601 với timezone, VD: 2026-04-10T19:00:00.000+07:00."""
        return await _pf_request(ctx, "POST", "/transaction/add-transaction", body={
            "amount": amount, "type": type,
            "categoryId": categoryId, "accountId": accountId,
            "note": note, "transactionDate": transactionDate,
        })

    @agent.tool
    async def get_user_budgets(ctx: RunContext[AppDeps]) -> dict:
        """Lấy danh sách ngân sách của người dùng: danh mục, hạn mức, đã chi, khoảng thời gian, lặp lại hay không. Dùng khi user hỏi về ngân sách hiện có hoặc trước khi tạo ngân sách mới để tránh trùng."""
        return await _pf_request(ctx, "GET", "/budget/budgets")

    @agent.tool
    async def get_wealth_account_types(ctx: RunContext[AppDeps]) -> dict:
        """Lấy danh sách loại tài khoản tài sản có thể tạo (VD: Tiền mặt, Tài khoản ngân hàng, Ví điện tử, Bất động sản, ...). Mỗi loại có id, displayName, isTransactionEligible (có dùng để ghi giao dịch không), isDebt (có phải nợ không). BẮT BUỘC gọi tool này TRƯỚC khi gọi create_wealth_account để lấy đúng accountTypeId."""
        return await _pf_request(ctx, "GET", "/wealth/account-types")

    @agent.tool
    async def create_wealth_account(
        ctx: RunContext[AppDeps],
        name: str,
        accountTypeId: str,
        balance: float,
        includeInNetWorth: bool = True,
    ) -> dict:
        """Tạo tài khoản tài sản mới cho người dùng (ví, tài khoản ngân hàng, bất động sản, ...). CHỈ gọi SAU KHI user đã xác nhận. TRƯỚC ĐÓ BẮT BUỘC gọi get_wealth_account_types để lấy đúng accountTypeId. includeInNetWorth=True nghĩa là tính vào tổng tài sản ròng, False thì không tính (dùng cho tài khoản theo dõi riêng). balance là số dư / giá trị ban đầu."""
        return await _pf_request(ctx, "POST", "/wealth/create-account", body={
            "name": name,
            "accountTypeId": accountTypeId,
            "balance": balance,
            "includeInNetWorth": includeInNetWorth,
        })

    @agent.tool
    async def add_budget(
        ctx: RunContext[AppDeps],
        categoryId: str,
        targetAmount: float,
        startDate: str,
        endDate: str,
        isRecurring: bool | None = None,
        recurringStartDate: str | None = None,
    ) -> dict:
        """Tạo ngân sách chi tiêu theo danh mục. CHỈ gọi SAU KHI user đã xác nhận. Trước đó BẮT BUỘC gọi get_user_transaction_context để chọn categoryId loại EXPENSE. Ngày dùng format YYYY-MM-DD (theo lịch, không cần giờ). isRecurring: true nếu user muốn ngân sách lặp lại theo kỳ (mặc định true nếu user nói 'hàng tháng')."""
        body: dict[str, Any] = {
            "categoryId": categoryId, "targetAmount": targetAmount,
            "startDate": startDate, "endDate": endDate,
        }
        if isRecurring is not None:
            body["isRecurring"] = isRecurring
        if isinstance(recurringStartDate, str) and recurringStartDate.strip():
            body["recurringStartDate"] = recurringStartDate.strip()
        return await _pf_request(ctx, "POST", "/budget/create-budget", body=body)

    @agent.tool
    async def update_budget(
        ctx: RunContext[AppDeps],
        budgetId: str,
        categoryId: str,
        targetAmount: float,
        startDate: str,
        endDate: str,
        isRecurring: bool | None = None,
        recurringStartDate: str | None = None,
    ) -> dict:
        """Cập nhật ngân sách đã có. CHỈ gọi SAU KHI user xác nhận thông tin mới. budgetId lấy từ get_user_budgets. Ngày dùng format YYYY-MM-DD."""
        body: dict[str, Any] = {
            "categoryId": categoryId, "targetAmount": targetAmount,
            "startDate": startDate, "endDate": endDate,
        }
        if isRecurring is not None:
            body["isRecurring"] = isRecurring
        if isinstance(recurringStartDate, str) and recurringStartDate.strip():
            body["recurringStartDate"] = recurringStartDate.strip()
        return await _pf_request(ctx, "PUT", f"/budget/budgets/{budgetId}", body=body)

    @agent.tool
    async def delete_budget(
        ctx: RunContext[AppDeps],
        budgetId: str,
    ) -> dict:
        """Xóa ngân sách. CHỈ gọi SAU KHI user xác nhận muốn xóa. budgetId lấy từ get_user_budgets."""
        return await _pf_request(ctx, "DELETE", f"/budget/budgets/{budgetId}")

    @agent.tool
    async def get_portfolio_analysis(
        ctx: RunContext[AppDeps],
        portfolioId: str | None = None,
    ) -> dict:
        """Lấy phân tích danh mục đầu tư của người dùng. Trả về: holdings (symbol, averagePrice, closePrice, unrealizedPnLPct, weightPct), totalMarketValue, totalCostBasis, unrealizedPnL, currentPE, currentPB, cashBalance, historyQuarters (PE/PB/ROE/ROA theo quý), allPortfolios (nếu có nhiều danh mục). Dùng khi user hỏi về danh mục, lãi lỗ, đánh giá sức khỏe danh mục, hoặc gợi ý đầu tư với tiền dư. Kết hợp với compute_fair_value (cho top 3 holdings theo weightPct) và get_personal_finance_report (cho tiền dư tháng) để phân tích toàn diện."""
        base_url = settings.JAVA_BACKEND_URL.rstrip("/")
        headers: dict[str, str] = {}
        internal_key = (settings.INTERNAL_API_KEY or "").strip()
        if internal_key:
            headers["X-Internal-Api-Key"] = internal_key
        params: dict[str, str] = {"userId": ctx.deps.user_id}
        if portfolioId:
            params["portfolioId"] = portfolioId
        response = await get_http_client().get(
            f"{base_url}/api/internal/portfolio/analysis",
            params=params,
            headers=headers,
            timeout=httpx.Timeout(max(5, int(settings.CHAT_TOOL_TIMEOUT_SECONDS))),
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"portfolio/analysis HTTP_{response.status_code}: {response.text[:300]}")
        return response.json()


# ── Compute fair value ─────────────────────────────────────────────────


def _register_compute_tool(agent: Agent[AppDeps, str]) -> None:
    @agent.tool
    async def compute_fair_value(
        ctx: RunContext[AppDeps],
        symbol: str,
        target_year: int | None = None,
    ) -> dict:
        """Tính giá trị hợp lý cổ phiếu. Tool TỰ LẤY toàn bộ dữ liệu cần thiết và tính toán. Chỉ cần truyền mã cổ phiếu. KHÔNG cần gọi tool nào khác trước khi gọi tool này. Nếu user nói 'tầm nhìn 2030' hoặc chỉ rõ năm → truyền target_year. Kết quả gồm: P/E target, P/B target, giá hợp lý, verdict, forecast_series (lộ trình LNST/DT tới năm mục tiêu), forecast_top_factors (yếu tố ảnh hưởng dự phóng từ mô hình ML), growth_source (forecast hoặc historical)."""
        inputs = await fetch_valuation_inputs(ctx.deps.market_client, symbol, target_year)
        if "error" in inputs and len(inputs) == 1:
            # Data fetch failure → ask the LLM to verify the symbol or fall back
            # to a different tool (e.g. suggest_companies).
            raise ModelRetry(
                f"Không đủ dữ liệu để định giá {symbol}: {inputs['error']}. "
                f"Hãy thử suggest_companies('{symbol}') để xác minh mã."
            )
        result = _compute_fair_value(inputs)
        if "error" in result and len(result) == 1:
            raise ModelRetry(
                f"Tính giá hợp lý {symbol} thất bại: {result['error']}. "
                f"Có thể thiếu EPS/BVPS — thử get_company_metrics trước rồi quyết định."
            )
        return result


# ── RAG ────────────────────────────────────────────────────────────────


def _register_rag_tool(agent: Agent[AppDeps, str]) -> None:
    @agent.tool
    async def search_annual_reports(
        ctx: RunContext[AppDeps],
        ticker: str,
        query: str,
    ) -> list[dict]:
        """Tìm kiếm thông tin định tính từ báo cáo thường niên (~700 công ty, 5 năm 2019-2024). Dùng khi cần: chiến lược kinh doanh, rủi ro, quản trị, kế hoạch mở rộng, triển vọng ngành, giải thích nguyên nhân biến động tài chính, hoặc bổ sung bối cảnh cho phân tích định giá. NÊN gọi sau compute_fair_value để bổ sung góc nhìn chiến lược cho định giá."""
        ticker_clean = (ticker or "").strip().upper()
        query_clean = (query or "").strip()
        if not ticker_clean or not query_clean:
            raise ModelRetry(
                "search_annual_reports cần cả ticker (mã cổ phiếu) và query (câu hỏi). "
                "Hãy gọi lại với đầy đủ 2 tham số."
            )
        chunks = await ctx.deps.rag_service.retrieve(
            query=query_clean, ticker=ticker_clean, years=None,
        )
        return [
            {
                "chunk_id": c.get("chunk_id"),
                "source_title": c.get("source_title"),
                "page_number": c.get("page_number"),
                "text": (c.get("text") or "")[:1200],
            }
            for c in (chunks or [])[:6]
        ]
