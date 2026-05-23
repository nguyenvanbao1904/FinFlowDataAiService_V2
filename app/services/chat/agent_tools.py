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
from app.services.chat.prompts.agent_prompt import AGENT_SYSTEM_PROMPT, CFO_STRESS_ADDENDUM as _CFO_STRESS_ADDENDUM
from app.services.chat.utils.vietnamese_text import sanitize_user_facing_message
from app.services.chat.valuation_engine import compute_fair_value as _compute_fair_value
from app.services.chat.valuation_inputs import fetch_valuation_inputs

logger = logging.getLogger(__name__)

_VN_TZ = datetime.timezone(datetime.timedelta(hours=7))


_CFO_TRIGGER_KEYWORDS = (
    "cfo ảo", "stress tài chính", "quỹ dự phòng", "survival runway",
    "tốc độ đầu tư", "tỷ lệ đầu tư", "monthly invest", "phân tích tài chính cá nhân",
    "dòng tiền thặng dư", "dòng tiền", "thu nhập thặng dư", "tiền dư hàng tháng",
)


def _is_cfo_context(user_message: str) -> bool:
    lower = user_message.lower()
    return any(kw in lower for kw in _CFO_TRIGGER_KEYWORDS)


@dataclass
class AppDeps:
    """Per-request dependencies injected into every tool via RunContext."""
    user_id: str
    market_client: MarketDataToolClient
    rag_service: RagRetrievalService
    cfo_context: bool = False


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
        """Strip markdown + technical jargon from the LLM's final answer."""
        cleaned = sanitize_user_facing_message((output or "").strip())
        return cleaned or "Xin lỗi, tôi chưa thể trả lời lúc này."

    @agent.system_prompt
    def _system_prompt(ctx: RunContext[AppDeps]) -> str:
        now = datetime.datetime.now(_VN_TZ)
        base = AGENT_SYSTEM_PROMPT
        if ctx.deps.cfo_context:
            base += _CFO_STRESS_ADDENDUM
        return (
            base
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
        """Doanh thu, lợi nhuận, ROE, ROA, biên lãi theo năm/quý.
        Dùng khi phân tích sức khỏe tài chính, tăng trưởng, hiệu quả hoạt động, so sánh các năm.
        Mặc định annualLimit=3; truyền annualLimit=5 khi cần xu hướng dài hạn.
        Backend đã tính sẵn tăng trưởng YoY: yoyGrowth (LNST), yoyNetRevenue (doanh thu thuần, non-bank),
        yoyCustomerLoan (cho vay KH, bank), yoyTotalOperatingIncome (TOI, bank),
        yoyNpl (nợ xấu, bank), yoyInventories (hàng tồn, non-bank).
        KHÔNG tự tính YoY — dùng trực tiếp các trường đã có."""
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
        """Bảng thống kê PE/PB/PS hàng ngày (trung vị, trung bình, min, max) theo khoảng thời gian.
        Dùng để SO SÁNH định giá LỊCH SỬ: hiện tại vs 5 năm qua — KHÔNG phải để tính giá hợp lý.
        Gọi đồng thời với get_company_live_valuation_snapshot và compute_fair_value
        khi user hỏi "đắt không", "rẻ không", "có nên mua không" — cần cả 3 góc nhìn."""
        return await _call_market(ctx, "get_company_daily_valuations",
                                  symbol=symbol, startDate=startDate, endDate=endDate, summary=summary)

    @agent.tool
    async def get_company_live_valuation_snapshot(ctx: RunContext[AppDeps], symbol: str) -> dict:
        """Giá realtime + PE/PB/PS hiện tại + median lịch sử + nhãn đánh giá rẻ/đắt (so với quá khứ).
        Tool này CHỈ so sánh hiện tại vs lịch sử — KHÔNG tính giá hợp lý (fair value).
        Khi user hỏi "đắt không", "rẻ không", "có nên mua": gọi ĐỒNG THỜI tool này VÀ compute_fair_value.
        Đừng kết luận đắt/rẻ chỉ dựa vào tool này — thiếu góc nhìn forward-looking từ forecast."""
        return await _call_market(ctx, "get_company_live_valuation_snapshot", symbol=symbol)

    @agent.tool
    async def get_company_forecast(
        ctx: RunContext[AppDeps], symbol: str, targetYear: int | None = None,
    ) -> dict:
        """Dự báo doanh thu & lợi nhuận cho năm mục tiêu, kèm top yếu tố ảnh hưởng.
        Dùng khi cần dự báo tương lai hoặc tính giá trị hợp lý theo năm cụ thể.
        Thường được gọi nội bộ bởi compute_fair_value — chỉ gọi riêng khi cần xem raw forecast."""
        return await _call_market(ctx, "get_company_forecast", symbol=symbol, targetYear=targetYear)

    @agent.tool
    async def get_company_metrics(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Overview gọn: EPS, BVPS, cổ phiếu lưu hành (CPLH), ROE, median PE/PB.
        Dùng khi cần EPS/BVPS/CPLH để tính giá hợp lý thủ công, hoặc thông tin tổng quan công ty.
        Ưu tiên compute_fair_value nếu chỉ cần kết quả định giá cuối cùng."""
        return await _call_market(ctx, "get_company_metrics",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_dividends(
        ctx: RunContext[AppDeps], symbol: str, annualLimit: int | None = None,
    ) -> dict:
        """Lịch sử cổ tức của công ty theo năm.
        Dùng khi user hỏi về cổ tức, tỷ suất cổ tức, chính sách trả cổ tức."""
        return await _call_market(ctx, "get_company_dividends", symbol=symbol, annualLimit=annualLimit)

    @agent.tool
    async def get_company_valuations(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None,
        startDate: str | None = None, endDate: str | None = None,
        showQuarterly: bool | None = None,
    ) -> dict:
        """PE/PB/PS theo quý từ DB (có thể tới 20+ năm).
        Ưu tiên dùng get_company_daily_valuations nếu cần thống kê trung vị/trung bình."""
        return await _call_market(ctx, "get_company_valuations",
                                  symbol=symbol, annualLimit=annualLimit,
                                  startDate=startDate, endDate=endDate, showQuarterly=showQuarterly)

    @agent.tool
    async def get_company_cash_flows(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Dòng tiền hoạt động, đầu tư, tài chính theo năm/quý.
        Dùng khi hỏi về chất lượng lợi nhuận, CFO/FCF, lợi nhuận có chuyển thành tiền không,
        dòng tiền âm/dương, hoặc rủi ro dòng tiền."""
        return await _call_market(ctx, "get_company_cash_flows",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_analysis(
        ctx: RunContext[AppDeps], symbol: str,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Phân tích chi tiết công ty: overview, tài chính, cổ đông lớn.
        Ưu tiên get_company_metrics nếu chỉ cần EPS/BVPS/CPLH."""
        return await _call_market(ctx, "get_company_analysis",
                                  symbol=symbol, annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_company_market_data(
        ctx: RunContext[AppDeps], symbol: str,
        include: str | None = None,
        annualLimit: int | None = None, quarterlyLimit: int | None = None,
    ) -> dict:
        """Dữ liệu thị trường tổng quát: overview, financials, valuation, dividends."""
        return await _call_market(ctx, "get_company_market_data",
                                  symbol=symbol, include=include,
                                  annualLimit=annualLimit, quarterlyLimit=quarterlyLimit)

    @agent.tool
    async def get_industry_nodes(ctx: RunContext[AppDeps]) -> dict:
        """Danh sách toàn bộ ngành nghề trên thị trường chứng khoán VN."""
        return await _call_market(ctx, "get_industry_nodes")

    @agent.tool
    async def suggest_companies(
        ctx: RunContext[AppDeps], q: str, limit: int | None = None,
    ) -> dict:
        """Tìm kiếm / gợi ý công ty theo tên hoặc mã cổ phiếu.
        Gọi khi user nhắc tên công ty mà chưa rõ mã (VD: "Vietcombank", "Vingroup"),
        hoặc khi tool khác báo lỗi mã không hợp lệ."""
        return await _call_market(ctx, "suggest_companies", q=q, limit=limit)

    @agent.tool
    async def get_company_industries(
        ctx: RunContext[AppDeps], symbols: list[str],
    ) -> dict:
        """Lấy ngành nghề của danh sách mã cổ phiếu.
        Dùng khi cần phân loại ngành để phân tích phân bổ hoặc so sánh cùng ngành."""
        return await _call_market(ctx, "get_company_industries", symbols=symbols)


# ── Personal finance tools ────────────────────────────────────────────


def _register_personal_finance_tools(agent: Agent[AppDeps, str]) -> None:
    """user_id is taken from ctx.deps — never exposed to the LLM schema."""

    @agent.tool
    async def get_personal_finance_report(ctx: RunContext[AppDeps]) -> dict:
        """Báo cáo tài chính cá nhân: thu nhập, chi tiêu, tỷ lệ tiết kiệm, top danh mục chi tiêu,
        biến động 4 tháng gần nhất. Tool TỰ TÍNH TOÁN mọi số liệu.
        Dùng khi user hỏi về thu chi, ngân sách, báo cáo tài chính cá nhân.
        KHÔNG dùng cho phân tích cổ phiếu hay công ty.
        Trình bày kết quả theo 4 phần:
          1. Tổng quan kỳ: thu nhập, chi tiêu, dòng tiền ròng, tỷ lệ tiết kiệm
          2. Xu hướng theo tháng: so sánh tháng hiện tại vs tháng trước
          3. Top danh mục chi tiêu lớn nhất và biến động
          4. Nhận xét sức khỏe tài chính và gợi ý cụ thể
        KHÔNG tự tính toán lại — chỉ diễn giải và nhận xét số liệu từ kết quả tool."""
        return await _pf_request(ctx, "GET", "/transaction/finance-report")

    @agent.tool
    async def get_user_transaction_context(ctx: RunContext[AppDeps]) -> dict:
        """Lấy danh sách danh mục giao dịch (categories) và tài khoản (accounts) của user.
        BẮT BUỘC gọi tool này TRƯỚC khi gọi add_transaction hoặc add_budget
        để lấy đúng categoryId và accountId.
        Kết quả: categories (id, name, type: INCOME/EXPENSE/SAVING), accounts (id, name, type, balance).
        Cũng dùng khi user hỏi xem có tài khoản nào (trường accounts)."""
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
        """Thêm giao dịch mới cho user. LUỒNG BẮT BUỘC:
        BƯỚC 1 — Gọi get_user_transaction_context để lấy categories và accounts.
          CHỈ gọi nếu conversation history CHƯA có kết quả tool này. Nếu đã có → BỎ QUA, dùng lại.
        BƯỚC 2 — Tự suy luận TẤT CẢ các trường (không hỏi user từng trường):
          - type: "ăn sáng/điện/mua sắm" → EXPENSE; "lương/thưởng" → INCOME.
          - amount: quy đổi "50k"→50000, "2tr"→2000000, "1.5 triệu"→1500000.
          - categoryId: chọn category phù hợp nhất từ danh sách.
          - accountId: chọn account eligible đầu tiên nếu user không chỉ định.
          - transactionDate: mặc định hôm nay, format ISO8601+timezone VD: 2026-04-10T19:00:00.000+07:00.
          - note: từ nội dung user mô tả.
        BƯỚC 3 — HIỂN THỊ XÁC NHẬN đầy đủ 6 trường, CHƯA GỌI tool này.
          CHỈ hiển thị xác nhận nếu conversation history CHƯA có màn hình xác nhận cho giao dịch này.
          Nếu đã hiển thị và user vừa reply "ok/đúng/nhập đi/ừ/luôn đi" → NHẢY THẲNG BƯỚC 4.
          Template:
            "Bạn muốn nhập giao dịch sau phải không?
            - Loại: [Chi tiêu/Thu nhập/Tiết kiệm]
            - Số tiền: [X,XXX] VND
            - Danh mục: [tên]
            - Tài khoản: [tên]
            - Ngày: [ngày]
            - Ghi chú: [note]
            Xác nhận để tôi nhập luôn nhé!"
        BƯỚC 4 — GỌI tool này ngay sau khi user xác nhận. Phản hồi thành công KHÔNG dùng emoji.
        Chỉ hỏi thêm khi thực sự thiếu thông tin không thể suy ra (VD: không biết số tiền)."""
        return await _pf_request(ctx, "POST", "/transaction/add-transaction", body={
            "amount": amount, "type": type,
            "categoryId": categoryId, "accountId": accountId,
            "note": note, "transactionDate": transactionDate,
        })

    @agent.tool
    async def get_user_budgets(ctx: RunContext[AppDeps]) -> dict:
        """Lấy danh sách ngân sách của user: danh mục, hạn mức, đã chi, khoảng thời gian, lặp lại.
        Dùng khi user hỏi về ngân sách hiện có, hoặc trước khi tạo/sửa/xóa ngân sách."""
        return await _pf_request(ctx, "GET", "/budget/budgets")

    @agent.tool
    async def get_wealth_account_types(ctx: RunContext[AppDeps]) -> dict:
        """Lấy danh sách loại tài khoản tài sản có thể tạo (Tiền mặt, Ngân hàng, Ví điện tử, BĐS...).
        Mỗi loại có: id, displayName, isTransactionEligible, isDebt.
        BẮT BUỘC gọi tool này NGAY LẬP TỨC khi user muốn tạo tài khoản — kể cả khi chưa đủ thông tin.
        KHÔNG tự bịa loại tài khoản trước khi có kết quả tool."""
        return await _pf_request(ctx, "GET", "/wealth/account-types")

    @agent.tool
    async def create_wealth_account(
        ctx: RunContext[AppDeps],
        name: str,
        accountTypeId: str,
        balance: float,
        includeInNetWorth: bool = True,
    ) -> dict:
        """Tạo tài khoản tài sản mới (ví, ngân hàng, bất động sản...). LUỒNG BẮT BUỘC:
        BƯỚC 1 — Gọi get_wealth_account_types để lấy danh sách loại.
          CHỈ gọi nếu conversation history CHƯA có kết quả tool này. Nếu đã có → BỎ QUA, dùng lại.
        BƯỚC 2 — Tự suy luận: tên tài khoản, loại phù hợp nhất (ĐỀ XUẤT 1 loại, không liệt kê nhiều),
          số dư ban đầu (mặc định 0), includeInNetWorth (mặc định true).
        BƯỚC 3 — HIỂN THỊ XÁC NHẬN 4 trường, CHƯA GỌI tool này.
          CHỈ hiển thị xác nhận nếu conversation history CHƯA có màn hình xác nhận cho tài khoản này.
          Nếu đã hiển thị và user vừa reply "ok/đúng/tạo đi/ừ/luôn đi" → NHẢY THẲNG BƯỚC 4.
          Template:
            "Bạn muốn tạo tài khoản sau phải không?
            - Tên: [tên]
            - Loại: [tên loại]
            - Số dư ban đầu: [số tiền]
            - Tính vào tổng tài sản: Có/Không
            Xác nhận để tôi tạo luôn nhé!"
        BƯỚC 4 — GỌI tool này ngay sau khi user xác nhận. Phản hồi thành công KHÔNG dùng emoji.
        includeInNetWorth=false chỉ khi user nói "không tính vào tổng tài sản"."""
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
        """Tạo ngân sách chi tiêu theo danh mục. LUỒNG BẮT BUỘC:
        BƯỚC 1 — Gọi get_user_transaction_context (chỉ dùng category type=EXPENSE) và get_user_budgets
          (để tránh trùng kỳ).
          CHỈ gọi nếu conversation history CHƯA có kết quả các tool này. Nếu đã có → BỎ QUA, dùng lại.
        BƯỚC 2 — Tự suy luận TẤT CẢ trường:
          - Quy đổi số tiền (50k→50000).
          - startDate/endDate format YYYY-MM-DD; nếu "tháng này/tháng 4": ngày đầu+cuối tháng đúng năm.
          - isRecurring: true nếu user nói "hàng tháng/lặp lại", mặc định false.
          - endDate không được trước hôm nay.
        BƯỚC 3 — HIỂN THỊ XÁC NHẬN đầy đủ trường, CHƯA GỌI tool này.
          CHỈ hiển thị xác nhận nếu conversation history CHƯA có màn hình xác nhận cho ngân sách này.
          Nếu đã hiển thị và user vừa reply "ok/đúng/tạo đi/ừ/luôn đi" → NHẢY THẲNG BƯỚC 4.
          Template:
            "Bạn muốn đặt ngân sách sau phải không?
            - Danh mục: [tên]
            - Hạn mức: [X,XXX] VND
            - Từ ngày: [startDate]
            - Đến ngày: [endDate]
            - Lặp hàng tháng: Có/Không
            Xác nhận để tôi tạo luôn nhé!"
        BƯỚC 4 — GỌI tool này ngay sau khi user xác nhận. Phản hồi thành công KHÔNG dùng emoji."""
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
        """Cập nhật ngân sách đã có. LUỒNG BẮT BUỘC:
        BƯỚC 1 — Gọi get_user_budgets để xác định budgetId cần sửa.
          Gọi get_user_transaction_context nếu user muốn đổi danh mục.
          CHỈ gọi nếu conversation history CHƯA có kết quả các tool này. Nếu đã có → BỎ QUA, dùng lại.
        BƯỚC 2 — Giữ nguyên các trường user không nhắc. Tự suy luận các trường cần thay đổi.
        BƯỚC 3 — HIỂN THỊ XÁC NHẬN đầy đủ trường (kể cả trường giữ nguyên), CHƯA GỌI tool này.
          CHỈ hiển thị nếu chưa có màn hình xác nhận trong conversation history.
          Nếu đã hiển thị và user vừa reply "ok/đúng/cập nhật đi/ừ" → NHẢY THẲNG BƯỚC 4.
          Template:
            "Bạn muốn cập nhật ngân sách sau phải không?
            - Danh mục: [tên]
            - Hạn mức mới: [X,XXX] VND
            - Từ ngày: [startDate]
            - Đến ngày: [endDate]
            - Lặp hàng tháng: Có/Không
            Xác nhận để tôi cập nhật luôn nhé!"
        BƯỚC 4 — GỌI tool này ngay sau khi user xác nhận. Phản hồi thành công KHÔNG dùng emoji.
        budgetId lấy từ get_user_budgets. Ngày format YYYY-MM-DD."""
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
        """Xóa ngân sách. LUỒNG BẮT BUỘC:
        BƯỚC 1 — Gọi get_user_budgets để xác định budgetId.
          CHỈ gọi nếu conversation history CHƯA có kết quả tool này. Nếu đã có → BỎ QUA, dùng lại.
        BƯỚC 2 — HIỂN THỊ XÁC NHẬN: tên danh mục + khoảng thời gian, hỏi user xác nhận xóa.
          CHỈ hiển thị nếu chưa có màn hình xác nhận. Nếu đã hiển thị và user reply "ok/xóa đi/ừ" → BƯỚC 3.
        BƯỚC 3 — GỌI tool này ngay sau khi user xác nhận. Phản hồi thành công KHÔNG dùng emoji."""
        return await _pf_request(ctx, "DELETE", f"/budget/budgets/{budgetId}")

    @agent.tool
    async def get_portfolio_analysis(
        ctx: RunContext[AppDeps],
        portfolioId: str | None = None,
    ) -> dict:
        """Phân tích danh mục đầu tư của user.
        Trả về: holdings (symbol, averagePrice, closePrice, unrealizedPnL, unrealizedPnLPct, weightPct,
        marketValue), totalMarketValue, totalCostBasis, unrealizedPnL, cashBalance,
        currentPE, currentPB, currentPS, historyQuarters (PE/PB/ROE/ROA theo quý),
        allPortfolios (nếu có nhiều danh mục).

        Dùng khi user hỏi về danh mục, lãi lỗ, sức khỏe danh mục, hay gợi ý đầu tư với tiền dư.
        Nếu allPortfolios.length > 1: hỏi user chọn danh mục nào trước.

        LUỒNG CHUẨN khi phân tích danh mục — gọi ĐỒNG THỜI:
          1. get_portfolio_analysis() — lấy holdings và sức khỏe danh mục
          2. get_personal_finance_report() — lấy tiền dư tháng (net_cash_flow = income - expense)
          3. compute_fair_value() cho TỐI ĐA 3 mã có weightPct lớn nhất trong holdings

        Trình bày kết quả theo 4 phần:
        ## Tổng quan danh mục
        - Tổng giá trị thị trường, giá vốn, lãi/lỗ tổng (số tiền + %)
        - Tiền mặt trong danh mục

        ## Holdings
        Bảng Markdown: Mã | Tỷ trọng | Giá vốn | Giá TT | Lãi/Lỗ% | Biên an toàn
        - "Biên an toàn" = (fair_value_pe - averagePrice) / averagePrice × 100
          → Tính theo GIÁ VỐN (averagePrice), KHÔNG phải giá thị trường.
          → Dương: fair value > giá vốn → luận điểm còn nguyên.
          → Âm: fair value < giá vốn → đang nắm trên giá trị ước tính.
          → Nếu chưa có fair value → để "—".
        - Sắp xếp theo weightPct giảm dần, làm tròn giá đến hàng trăm đồng.

        ## Đánh giá từng mã top 3
        Với mỗi mã đã compute fair value:
        - Biên an toàn: "Bạn mua [giá vốn], fair value ước tính [X], biên an toàn [+/-Y%]"
        - 1-2 câu sức khỏe tài chính + luận điểm còn đúng không
        - Nếu lãi/lỗ% < -20% VÀ biên an toàn âm: cảnh báo rõ cần xem lại luận điểm

        ## Sức khỏe danh mục
        - P/E & P/B bình quân (currentPE/currentPB) vs thị trường VN (~12-15x / ~1.5-2x)
        - Cảnh báo tập trung: 1 mã >40% hoặc 1 ngành >60%
        - Cảnh báo đa dạng hóa: ≤3 mã

        ## Gợi ý với tiền dư
        - Nếu net_cash_flow > 0: ưu tiên mã biên an toàn dương lớn nhất VÀ weightPct < 40%.
          Tính số cp = floor(net_cash_flow / closePrice), làm tròn xuống bội số 100.
          Nêu: "Với [X triệu] tiền dư, bạn có thể mua thêm ~[N] cp [MÃ] ở giá [giá TT],
          nâng tỷ trọng từ [X%] lên [Y%]"
        - Nếu net_cash_flow ≤ 0: không gợi ý mua thêm.

        Lưu ý: nếu priceType="INSUFFICIENT" → nêu thiếu giá thị trường, P/E P/B mang tính tham khảo.
        KHÔNG khuyến nghị mua/bán — phân tích dữ liệu, để user tự quyết."""
        base_url = settings.JAVA_BACKEND_URL.rstrip("/")
        headers: dict[str, str] = {}
        internal_key = (settings.INTERNAL_API_KEY or "").strip()
        if internal_key:
            headers["X-Internal-Api-Key"] = internal_key
        params: dict[str, str] = {"userId": ctx.deps.user_id}
        if portfolioId:
            params["portfolioId"] = portfolioId
        response = await get_http_client().get(
            f"{base_url}/portfolio/analysis",
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
        """Tính giá trị hợp lý cổ phiếu bằng phương pháp định giá phù hợp với đặc thù doanh nghiệp.
        Tool TỰ LẤY toàn bộ dữ liệu cần thiết và tính toán — KHÔNG cần gọi tool nào khác trước.
        Chỉ cần truyền mã cổ phiếu. Nếu user nói "tầm nhìn 2030" hoặc chỉ rõ năm → truyền target_year.
        So sánh 2 mã: gọi 2 lần ĐỒNG THỜI.

        GỌI TOOL NÀY KHI user hỏi bất kỳ câu nào liên quan đến:
        - "đắt không", "rẻ không", "có nên mua", "giá hợp lý", "định giá", "fair value"
        - "có tăng không", "mua ở giá này được không", "nên giữ hay bán"
        LUÔN gọi đồng thời get_company_live_valuation_snapshot để bổ sung so sánh lịch sử.

        Kết quả gồm: valuation_model, valuation_formula, model_reason, model_confidence, key_assumptions,
        summary, verdict, growth_source, forecast_series (lộ trình LNST/DT tới năm mục tiêu),
        forecast_top_factors.

        Khi trả lời, hãy linh hoạt như CFO, không cần ép đúng một template. Nên có các ý chính:
        - Giá hiện tại, giá hợp lý, upside/downside và kết luận.
        - Tên công ty và ngành phải lấy từ company_name/industry_label/industry_key trong kết quả tool.
          Không tự suy diễn ngành. Nếu ngành không rõ, nói "ngành chưa rõ trong dữ liệu".
        - Phương pháp định giá bằng tiếng Việt tự nhiên và công thức valuation_formula nếu hữu ích.
        - Vì sao phương pháp đó phù hợp với doanh nghiệp.
        - Một vài giả định chính: tăng trưởng dự phóng, tỷ lệ cổ tức tiền mặt/lợi nhuận sau thuế,
          tỷ suất cổ tức tiền mặt trên giá cổ phiếu, ROE chuẩn hóa, hoặc P/E/P/B lịch sử nếu liên quan.
        - Rủi ro lớn nhất làm định giá sai.

        Không lộ tên model nội bộ tiếng Anh như normalized_earnings, bank_pe_pb_blend,
        earnings_exit, ddm_dividend_discount. Không dùng từ "payout" hoặc "dividend yield"
        nếu có thể nói tiếng Việt.
        Nếu có forecast_top_factors: chỉ nêu 2-3 yếu tố chính và hướng tác động, không trích feature_value.
        Khi user hỏi định giá/phân tích/so sánh cổ phiếu cụ thể, NÊN gọi search_annual_reports sau định giá
        để bổ sung chất lượng mô hình kinh doanh, nguồn lợi nhuận chính, công ty liên doanh/liên kết,
        cổ tức, chiến lược và rủi ro. Query nên nêu thẳng các ý này, ví dụ:
        "nguồn lợi nhuận chính, lợi nhuận từ công ty liên doanh liên kết, cổ tức, mô hình kinh doanh,
        chiến lược và rủi ro"; với doanh nghiệp có liên doanh lớn có thể thêm tên đối tác nếu biết.
        Nếu so sánh 2-3 mã, gọi RAG cho từng mã được so sánh. Nếu nhiều hơn 3 mã,
        chỉ gọi RAG cho tối đa 3 mã trọng yếu hoặc mã user nhấn mạnh để tránh quá chậm.
        Không dùng kết quả RAG để tạo mục "Nguồn"; chỉ dùng để làm nhận xét kinh doanh thông minh hơn."""
        inputs = await fetch_valuation_inputs(ctx.deps.market_client, symbol, target_year)
        if "error" in inputs and len(inputs) == 1:
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
        """Tìm kiếm thông tin định tính từ báo cáo thường niên (~700 công ty, 5 năm 2019-2024).
        Dùng khi cần: chiến lược kinh doanh, rủi ro, quản trị, kế hoạch mở rộng, triển vọng ngành,
        giải thích nguyên nhân biến động tài chính.
        Gọi khi user hỏi định giá/phân tích/so sánh cổ phiếu cụ thể để bổ sung mô hình kinh doanh,
        nguồn lợi nhuận chính, liên doanh/liên kết, chiến lược và rủi ro định tính.
        Không dùng tool này chỉ để gắn "nguồn" cho một câu trả lời định giá định lượng.
        Hiện chưa hỗ trợ mở PDF đúng trang từ citation, nên không quảng cáo khả năng bấm nguồn như NotebookLM.
        Kết quả: tối đa 6 đoạn văn bản từ báo cáo, mỗi đoạn tối đa 1200 ký tự."""
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
