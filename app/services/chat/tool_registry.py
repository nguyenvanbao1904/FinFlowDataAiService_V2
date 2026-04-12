"""Central tool registry — single source of truth for all tool metadata.

Eliminates the schema-sync problem by defining tool names, descriptions,
and JSON schemas in ONE place. Used by:
- ChatOrchestrator  → pass to DeepSeek API as `tools` parameter
- MarketDataToolClient → unchanged (routes by tool name)

No other file should hard-code tool schemas or descriptions.
"""
from __future__ import annotations

from typing import Any


def _tool(
    name: str,
    desc: str,
    params: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible tool definition."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": params,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": schema,
        },
    }


# ── Tool definitions (OpenAI-compatible) ──────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _tool(
        "get_company_financial_series",
        "Doanh thu, lợi nhuận, ROE, ROA, biên lãi theo năm/quý từ DB báo cáo tài chính. "
        "Dùng khi cần phân tích sức khỏe tài chính, tăng trưởng, hiệu quả hoạt động. "
        "Mặc định annualLimit=3 nếu không chỉ định.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu, VD: HPG"},
            "annualLimit": {"type": "integer", "minimum": 1, "description": "Số năm gần nhất"},
            "quarterlyLimit": {"type": "integer", "minimum": 1, "description": "Số quý gần nhất"},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_daily_valuations",
        "Bảng tóm tắt PE/PB/PS hàng ngày (trung vị, trung bình, min, max) từ DB 5-10 năm. "
        "ĐÂY LÀ NGUỒN CHÍNH cho so sánh định giá lịch sử. Mặc định dùng 5 năm. "
        "Backend tự tính toán thống kê, trả về summary gọn.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "startDate": {"type": "string", "minLength": 8, "description": "Ngày bắt đầu YYYY-MM-DD"},
            "endDate": {"type": "string", "minLength": 8, "description": "Ngày kết thúc YYYY-MM-DD"},
            "summary": {"type": "boolean"},
        },
        required=["symbol", "startDate", "endDate"],
    ),
    _tool(
        "get_company_live_valuation_snapshot",
        "Giá realtime + PE/PB/PS hiện tại + median lịch sử + nhãn đánh giá rẻ/đắt.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_forecast",
        "Dự báo doanh thu & lợi nhuận cho năm mục tiêu, kèm top yếu tố ảnh hưởng. "
        "Dùng khi cần dự báo tương lai hoặc tính giá trị hợp lý.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "targetYear": {"type": "integer", "minimum": 2000, "maximum": 2100, "description": "Năm dự báo"},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_metrics",
        "Overview gọn: EPS, BVPS, cổ phiếu lưu hành (CPLH), ROE, median PE/PB. "
        "CẦN THIẾT khi tính giá hợp lý (cần EPS, BVPS, CPLH). Đã loại bỏ dữ liệu thừa.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "annualLimit": {"type": "integer", "minimum": 1},
            "quarterlyLimit": {"type": "integer", "minimum": 1},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_dividends",
        "Lịch sử cổ tức của công ty.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "annualLimit": {"type": "integer", "minimum": 1},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_valuations",
        "PE/PB/PS theo quý từ DB. Chỉ có dữ liệu ~2 năm gần đây. "
        "Ưu tiên dùng get_company_daily_valuations thay thế.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "annualLimit": {"type": "integer", "minimum": 1},
            "startDate": {"type": "string"},
            "endDate": {"type": "string"},
            "showQuarterly": {"type": "boolean"},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_analysis",
        "Phân tích chi tiết công ty (overview, tài chính, cổ đông). "
        "Ưu tiên dùng get_company_metrics nếu chỉ cần EPS/BVPS/CPLH.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "annualLimit": {"type": "integer", "minimum": 1},
            "quarterlyLimit": {"type": "integer", "minimum": 1},
        },
        required=["symbol"],
    ),
    _tool(
        "get_company_market_data",
        "Dữ liệu thị trường tổng quát (overview, financials, valuation, dividends).",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu"},
            "include": {"type": "string", "description": "all|company|shareholders|dividends|financialIndicators|bankBalanceSheets|nonBankBalanceSheets|bankIncomeStatements|nonBankIncomeStatements"},
            "annualLimit": {"type": "integer", "minimum": 1},
            "quarterlyLimit": {"type": "integer", "minimum": 1},
        },
        required=["symbol"],
    ),
    _tool(
        "get_industry_nodes",
        "Danh sách toàn bộ ngành nghề trên thị trường.",
        {},
    ),
    _tool(
        "suggest_companies",
        "Tìm kiếm/gợi ý công ty theo tên hoặc mã.",
        {
            "q": {"type": "string", "minLength": 1, "description": "Từ khóa tìm kiếm"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        required=["q"],
    ),
    _tool(
        "get_company_industries",
        "Lấy ngành nghề của các mã cổ phiếu.",
        {
            "symbols": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        required=["symbols"],
    ),
    _tool(
        "search_annual_reports",
        "Tìm kiếm thông tin từ báo cáo thường niên (~50 công ty, 5 năm gần nhất). "
        "Dùng khi cần thông tin định tính: chiến lược kinh doanh, rủi ro, quản trị, kế hoạch mở rộng, triển vọng ngành.",
        {
            "ticker": {"type": "string", "description": "Mã cổ phiếu"},
            "query": {"type": "string", "description": "Câu hỏi hoặc từ khóa tìm kiếm"},
        },
        required=["ticker", "query"],
    ),
    _tool(
        "compute_fair_value",
        "Tính giá trị hợp lý cổ phiếu. Tool TỰ LẤY toàn bộ dữ liệu cần thiết và tính toán. "
        "Chỉ cần truyền mã cổ phiếu. KHÔNG cần gọi tool nào khác trước khi gọi tool này. "
        "Kết quả gồm: P/E target, P/B target, giá hợp lý, so sánh với giá hiện tại, verdict.",
        {
            "symbol": {"type": "string", "description": "Mã cổ phiếu, VD: FRT, HPG, VCB"},
            "target_year": {"type": "integer", "description": "Năm dự báo (mặc định năm sau)"},
        },
        required=["symbol"],
    ),
    _tool(
        "get_personal_finance_report",
        "Lấy báo cáo tài chính cá nhân của người dùng: thu nhập, chi tiêu, tỷ lệ tiết kiệm, "
        "top danh mục chi tiêu, biến động theo tháng (4 tháng gần nhất). "
        "Dùng khi người dùng hỏi về tình hình thu chi, ngân sách, chi tiêu cá nhân, "
        "hoặc muốn tạo báo cáo tài chính cá nhân. "
        "Tool TỰ TÍNH TOÁN mọi số liệu — chỉ cần truyền user_id. "
        "KHÔNG dùng tool này cho phân tích cổ phiếu hay công ty.",
        {
            "user_id": {"type": "string", "description": "ID người dùng (tự động lấy từ context)"},
        },
        required=["user_id"],
    ),
    _tool(
        "get_user_transaction_context",
        "Lấy danh sách danh mục giao dịch (categories) và tài khoản (accounts) của người dùng. "
        "BẮT BUỘC gọi tool này TRƯỚC khi gọi add_transaction để lấy đúng categoryId và accountId. "
        "Kết quả gồm: categories (id, name, type: INCOME/EXPENSE/SAVING), accounts (id, name, type, balance).",
        {
            "user_id": {"type": "string", "description": "ID người dùng (tự động lấy từ context)"},
        },
        required=["user_id"],
    ),
    _tool(
        "add_transaction",
        "Thêm giao dịch mới cho người dùng. CHỈ gọi tool này SAU KHI đã xác nhận với người dùng. "
        "Trước khi gọi, BẮT BUỘC gọi get_user_transaction_context để lấy đúng categoryId và accountId. "
        "CHÚ Ý: transactionDate phải ở format ISO8601 với timezone, VD: 2026-04-10T19:00:00.000+07:00.",
        {
            "user_id": {"type": "string", "description": "ID người dùng (tự động lấy từ context)"},
            "amount": {"type": "number", "description": "Số tiền giao dịch (VND)"},
            "type": {"type": "string", "enum": ["INCOME", "EXPENSE", "SAVING"], "description": "Loại: INCOME (thu nhập), EXPENSE (chi tiêu), SAVING (tiết kiệm)"},
            "categoryId": {"type": "string", "description": "UUID của danh mục (lấy từ get_user_transaction_context)"},
            "accountId": {"type": "string", "description": "UUID của tài khoản (lấy từ get_user_transaction_context)"},
            "note": {"type": "string", "description": "Ghi chú / mô tả giao dịch"},
            "transactionDate": {"type": "string", "description": "Ngày giao dịch ISO8601 VD: 2026-04-10T19:00:00.000+07:00"},
        },
        required=["user_id", "amount", "type", "categoryId", "accountId", "transactionDate"],
    ),
    _tool(
        "get_user_budgets",
        "Lấy danh sách ngân sách của người dùng: danh mục, hạn mức, đã chi, khoảng thời gian, lặp lại hay không. "
        "Dùng khi user hỏi về ngân sách hiện có hoặc trước khi tạo ngân sách mới để tránh trùng. "
        "Chỉ cần user_id.",
        {
            "user_id": {"type": "string", "description": "ID người dùng (tự động lấy từ context)"},
        },
        required=["user_id"],
    ),
    _tool(
        "add_budget",
        "Tạo ngân sách chi tiêu theo danh mục. CHỈ gọi SAU KHI user đã xác nhận. "
        "Trước đó BẮT BUỘC gọi get_user_transaction_context để chọn categoryId loại EXPENSE. "
        "Ngày dùng format YYYY-MM-DD (theo lịch, không cần giờ). "
        "isRecurring: true nếu user muốn ngân sách lặp lại theo kỳ (mặc định true nếu user nói 'hàng tháng').",
        {
            "user_id": {"type": "string", "description": "ID người dùng (tự động lấy từ context)"},
            "categoryId": {"type": "string", "description": "UUID danh mục chi tiêu (EXPENSE)"},
            "targetAmount": {"type": "number", "description": "Hạn mức VND (số dương)"},
            "startDate": {"type": "string", "description": "Ngày bắt đầu YYYY-MM-DD"},
            "endDate": {"type": "string", "description": "Ngày kết thúc YYYY-MM-DD (phải >= hôm nay theo quy tắc hệ thống)"},
            "isRecurring": {"type": "boolean", "description": "Có lặp lại theo kỳ hay không (tuỳ chọn)"},
            "recurringStartDate": {"type": "string", "description": "Ngày bắt đầu chu kỳ lặp YYYY-MM-DD (tuỳ chọn; nếu isRecurring=true có thể bỏ qua = startDate)"},
        },
        required=["user_id", "categoryId", "targetAmount", "startDate", "endDate"],
    ),
]

# ── Convenience accessors ─────────────────────────────────────────────

# Tool names routed to MarketDataToolClient (all except RAG, local tools, and personal-finance tools).
RAG_TOOL_NAME: str = "search_annual_reports"
LOCAL_TOOL_NAME: str = "compute_fair_value"
PERSONAL_FINANCE_TOOL_NAMES: frozenset[str] = frozenset({
    "get_personal_finance_report",
    "get_user_transaction_context",
    "add_transaction",
    "get_user_budgets",
    "add_budget",
})

_NON_MARKET_TOOLS: frozenset[str] = frozenset(
    {RAG_TOOL_NAME, LOCAL_TOOL_NAME} | PERSONAL_FINANCE_TOOL_NAMES
)

MARKET_DATA_TOOLS: frozenset[str] = frozenset(
    t["function"]["name"]
    for t in TOOL_DEFINITIONS
    if t["function"]["name"] not in _NON_MARKET_TOOLS
)


def get_all_tool_names() -> frozenset[str]:
    """All valid tool names."""
    return frozenset(t["function"]["name"] for t in TOOL_DEFINITIONS)


