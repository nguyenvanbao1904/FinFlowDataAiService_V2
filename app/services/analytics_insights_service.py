from __future__ import annotations

import json
import logging
from typing import Any

from app.domain.entities.analytics_insights import (
    AnalyticsInsightItem,
    AnalyticsInsightsRequest,
    AnalyticsInsightsResponse,
)
from app.services.chat.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalyticsInsightsService:
    def __init__(self) -> None:
        self.llm_client = LLMClient()

    async def generate(self, request: AnalyticsInsightsRequest) -> AnalyticsInsightsResponse:
        if not self._has_activity_in_last_two_months(request):
            logger.warning("Analytics insights skip LLM: no activity in last 2 calendar months (monthlySeries)")
            return self._insufficient_recent_data_response()

        try:
            logger.warning("Analytics insights LLM call start: tier=%s", request.insightTier)
            raw = await self._call_llm(request)
            response = self._normalize(raw, request)
            response.cached = False
            logger.warning("Analytics insights LLM call success: insights=%d", len(response.insights))
            return response
        except Exception as exc:
            logger.warning(
                "Analytics insights unexpected error: type=%s detail=%s",
                type(exc).__name__,
                exc,
            )
            return self._fallback_response(
                request,
                cached=False,
                warn=f"LLM insight delivery error: {type(exc).__name__}: {exc}",
            )

    async def _call_llm(self, request: AnalyticsInsightsRequest) -> dict[str, Any]:
        prompt = self._build_prompt(request)
        system_prompt = (
            "You are a concise Vietnamese personal finance insights generator. "
            "Return ONLY a raw JSON object. No markdown, no code fences, no prose. "
            "All numerical stats are pre-calculated. Do NOT perform any math — "
            "just quote the provided statistics and write human-readable insights."
        )
        
        parsed, usage = await self.llm_client.call_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=400,
            stage="analytics",
            enable_thinking=False,  # Simple NLP task: fast, uses response_format=json_object
        )
        logger.warning("Analytics LLM parsed keys=%s", list(parsed.keys()) if parsed else "EMPTY")
        return parsed



    def _build_prompt(self, request: AnalyticsInsightsRequest) -> str:
        if request.insightTier == "SPARSE":
            return self._build_prompt_sparse(request)
        return self._build_prompt_full(request)

    def _build_prompt_full(self, request: AnalyticsInsightsRequest) -> str:
        categories = [
            {"name": c.name, "amount": c.amount, "sharePct": c.sharePct}
            for c in request.previousMonthTopExpenseCategories[:5]
        ]
        category_delta = [
            {
                "name": c.name,
                "previousAmount": c.previousAmount,
                "baselineAvgAmount": c.baselineAvgAmount,
                "deltaPct": c.deltaPct,
            }
            for c in request.previousMonthCategoryDelta[:5]
        ]
        savings_rate_series = [
            {"month": s.month, "savingsRatePct": s.savingsRatePct}
            for s in request.savingsRateSeries[:4]
        ]
        current_day = request.currentDayOfMonth if request.currentDayOfMonth is not None else 0
        monthly_series = [
            {
                "month": m.month,
                "income": m.income,
                "expense": m.expense,
                "net": m.net,
                "topExpenseCategories": [
                    {"name": c.name, "amount": c.amount, "sharePct": c.sharePct}
                    for c in m.topExpenseCategories[:3]
                ],
            }
            for m in request.monthlySeries[:4]
        ]
        return f"""
STRICT RULES:
- Generate exactly 2 insights: one WARNING and one TIP.
- Title: short, descriptive, action-oriented (e.g. "Cảnh báo chi ăn uống", "Cải thiện tỷ lệ tiết kiệm"). Do NOT include dates or time ranges in titles.
- Message: short (<= 180 chars), practical, non-judgmental. You may reference months (e.g. "tháng 3") in the message body when relevant.
- Do not invent data outside provided context.
- Use Vietnamese.
- insightTier=FULL: interpret trends using MONTHLY_SERIES. Compare CURRENT_MONTH_LABEL vs PREVIOUS_MONTH_LABEL when relevant.
- CRITICAL: If CURRENT_DAY_OF_MONTH < 5, DO NOT conclude trend for CURRENT_MONTH_LABEL.
- UNIT CHECK: month-level claim → MONTHLY_SERIES or PREVIOUS_MONTH_TOP_EXPENSE_CATEGORIES; lookback-level claim → TOTAL_*_LOOKBACK.
- PRIORITY: (1) lifestyle creep via PREVIOUS_MONTH_CATEGORY_DELTA, (2) savings discipline via SAVINGS_RATE_SERIES, (3) practical next step.
- Do not claim yearly or all-time trends unless data supports it.

RETURN SCHEMA:
{{
  "insights": [
    {{
      "id": string,
      "type": "WARNING"|"TIP",
      "title": string,
      "message": string,
      "confidence": number
    }}
  ],
  "warnings": string[]
}}

CONTEXT:
locale={request.locale}
timezone={request.timezone}
currency={request.currency}
periodLabel={request.periodLabel}
insightTier=FULL
asOfDate={request.asOfDate}
currentDayOfMonth={current_day}
isBeginningOfMonth={request.isBeginningOfMonth}
currentMonthLabel={request.currentMonthLabel}
previousMonthLabel={request.previousMonthLabel}
lookbackLabel={request.lookbackLabel}
recentTransactionCount={request.recentTransactionCount}
totalIncomeLookback={self._lookback_income(request)}
totalExpenseLookback={self._lookback_expense(request)}
netCashflowLookback={self._lookback_net_cashflow(request)}
avgIncomePrev2Months={request.avgIncomePrev2Months}
avgExpensePrev2Months={request.avgExpensePrev2Months}
savingsRateSeries={json.dumps(savings_rate_series, ensure_ascii=False)}
previousMonthCategoryDelta={json.dumps(category_delta, ensure_ascii=False)}
previousMonthTopExpenseCategories={json.dumps(categories, ensure_ascii=False)}
monthlySeries={json.dumps(monthly_series, ensure_ascii=False)}
""".strip()

    def _build_prompt_sparse(self, request: AnalyticsInsightsRequest) -> str:
        categories = [
            {"name": c.name, "amount": c.amount, "sharePct": c.sharePct}
            for c in request.previousMonthTopExpenseCategories[:5]
        ]
        current_day = request.currentDayOfMonth if request.currentDayOfMonth is not None else 0
        monthly_series = [
            {
                "month": m.month,
                "income": m.income,
                "expense": m.expense,
                "net": m.net,
                "topExpenseCategories": [
                    {"name": c.name, "amount": c.amount, "sharePct": c.sharePct}
                    for c in m.topExpenseCategories[:3]
                ],
            }
            for m in request.monthlySeries[:4]
        ]
        n = request.recentTransactionCount if request.recentTransactionCount is not None else 0
        return f"""
STRICT RULES:
- Generate exactly 2 insights: one WARNING and one TIP.
- Title: short, descriptive, action-oriented. Do NOT include dates or time ranges in titles.
- Message: short (<= 180 chars), practical, non-judgmental.
- Do not invent data outside provided context.
- Use Vietnamese.
- insightTier=SPARSE: data is limited. Do NOT claim strong trends.
- If CURRENT_DAY_OF_MONTH < 5, avoid concluding trend for CURRENT_MONTH_LABEL.
- Give practical, general guidance (ghi đều, phân loại, theo dõi tuần) and acknowledge limited data briefly.

RETURN SCHEMA:
{{
  "insights": [
    {{
      "id": string,
      "type": "WARNING"|"TIP",
      "title": string,
      "message": string,
      "confidence": number (keep <= 0.75)
    }}
  ],
  "warnings": string[]
}}

CONTEXT:
locale={request.locale}
timezone={request.timezone}
currency={request.currency}
periodLabel={request.periodLabel}
insightTier=SPARSE
asOfDate={request.asOfDate}
currentDayOfMonth={current_day}
currentMonthLabel={request.currentMonthLabel}
previousMonthLabel={request.previousMonthLabel}
lookbackLabel={request.lookbackLabel}
recentTransactionCount={n}
totalIncomeLookback={self._lookback_income(request)}
totalExpenseLookback={self._lookback_expense(request)}
netCashflowLookback={self._lookback_net_cashflow(request)}
previousMonthTopExpenseCategories={json.dumps(categories, ensure_ascii=False)}
monthlySeries={json.dumps(monthly_series, ensure_ascii=False)}
""".strip()

    def _normalize(self, output: dict[str, Any], request: AnalyticsInsightsRequest) -> AnalyticsInsightsResponse:
        raw_items = output.get("insights")
        items: list[AnalyticsInsightItem] = []
        real_count = 0
        if isinstance(raw_items, list):
            for idx, item in enumerate(raw_items[:5]):
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("type", "")).upper()
                if kind not in {"WARNING", "TIP"}:
                    continue
                title = str(item.get("title", "")).strip()[:60]
                message = str(item.get("message", "")).strip()[:180]
                if not title or not message:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = min(max(confidence, 0.0), 1.0)
                if request.insightTier == "SPARSE":
                    confidence = min(confidence, 0.75)
                items.append(
                    AnalyticsInsightItem(
                        id=str(item.get("id") or f"insight-{idx}"),
                        type=kind,  # type: ignore[arg-type]
                        title=title,
                        message=message,
                        confidence=confidence,
                    )
                )
                real_count += 1

        logger.warning("Analytics normalize: real_count=%d from LLM output", real_count)
        warning_present = any(i.type == "WARNING" for i in items)
        tip_present = any(i.type == "TIP" for i in items)
        if not warning_present:
            items.insert(
                0,
                AnalyticsInsightItem(
                    id="fallback-warning",
                    type="WARNING",
                    title="Theo dõi chi tiêu",
                    message="Duy trì ghi nhận giao dịch đều để phát hiện sớm biến động chi tiêu.",
                    confidence=0.65 if request.insightTier == "SPARSE" else 0.7,
                ),
            )
        if not tip_present:
            items.append(
                AnalyticsInsightItem(
                    id="fallback-tip",
                    type="TIP",
                    title="Mẹo tài chính",
                    message="Đặt mức trần cho 1–2 danh mục chi lớn để kiểm soát ngân sách.",
                    confidence=0.65 if request.insightTier == "SPARSE" else 0.7,
                )
            )

        by_type: dict[str, AnalyticsInsightItem] = {}
        for item in items:
            by_type[item.type] = item

        result = []
        if "WARNING" in by_type:
            result.append(by_type["WARNING"])
        if "TIP" in by_type:
            result.append(by_type["TIP"])

        raw_warnings = output.get("warnings")
        warnings = [str(w) for w in raw_warnings[:5]] if isinstance(raw_warnings, list) else []
        return AnalyticsInsightsResponse(insights=result, warnings=warnings, cached=False)



    @staticmethod
    def _has_activity_in_last_two_months(request: AnalyticsInsightsRequest) -> bool:
        series = request.monthlySeries
        if len(series) < 2:
            return False
        tail = series[-2:]
        return any((m.income > 0 or m.expense > 0) for m in tail)

    @staticmethod
    def _insufficient_recent_data_response() -> AnalyticsInsightsResponse:
        return AnalyticsInsightsResponse(
            insights=[
                AnalyticsInsightItem(
                    id="insufficient-recent-data",
                    type="TIP",
                    title="AI đang học cách bạn sử dụng",
                    message=(
                        "Chưa có giao dịch trong 2 tháng gần nhất. "
                        "Ghi thêm giao dịch để AI có dữ liệu phân tích thói quen chi tiêu của bạn."
                    ),
                    confidence=1.0,
                )
            ],
            warnings=[],
            cached=False,
        )

    def _fallback_response(
        self,
        request: AnalyticsInsightsRequest,
        cached: bool,
        warn: str | None = None,
    ) -> AnalyticsInsightsResponse:
        scope = request.lookbackLabel or (f"dữ liệu đến {request.asOfDate}" if request.asOfDate else "kỳ hiện tại")
        income = self._lookback_income(request)
        expense = self._lookback_expense(request)
        net_cashflow = self._lookback_net_cashflow(request)
        ratio = (expense / income) if income > 0 else 0.0
        ratio_pct = int(round(ratio * 100))
        n = request.recentTransactionCount if request.recentTransactionCount is not None else 0

        if request.insightTier == "SPARSE":
            warning_msg = (
                f"Trong {scope}, dữ liệu còn ít ({n} giao dịch) — insight mang tính gợi ý chung."
            )
            tip_msg = (
                f"Gợi ý cho {scope}: ghi đều và phân loại để sau này so sánh "
                f"{request.currentMonthLabel or 'tháng này'} với {request.previousMonthLabel or 'tháng trước'} rõ hơn."
            )
        else:
            warning_msg = (
                f"Trong {scope}, bạn chưa ghi nhận thu nhập nhưng đã có chi tiêu. Hãy kiểm tra để tránh thâm hụt."
                if income <= 0 and expense > 0
                else f"Trong {scope}, tỷ lệ chi tiêu/thu nhập khoảng {ratio_pct}%. Theo dõi các khoản chi lớn."
            )
            tip_msg = (
                f"Trong {scope}, bạn đang dương {net_cashflow:,.0f} {request.currency} — cân nhắc trích một phần vào quỹ dự phòng."
                if net_cashflow > 0
                else f"Trong {scope}, hãy đặt hạn mức cho các danh mục chi chính để cân bằng dòng tiền."
            )

        warnings = [warn] if warn else []
        return AnalyticsInsightsResponse(
            insights=[
                AnalyticsInsightItem(
                    id="fallback-warning",
                    type="WARNING",
                    title="Cảnh báo chi tiêu",
                    message=warning_msg[:180],
                    confidence=0.65,
                ),
                AnalyticsInsightItem(
                    id="fallback-tip",
                    type="TIP",
                    title="Mẹo tài chính",
                    message=tip_msg[:180],
                    confidence=0.65,
                ),
            ],
            warnings=warnings,
            cached=cached,
        )

    @staticmethod
    def _truncate_for_log(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}...(truncated {len(value) - max_chars} chars)"

    @staticmethod
    def _lookback_income(request: AnalyticsInsightsRequest) -> float:
        return request.totalIncomeLookback or 0.0

    @staticmethod
    def _lookback_expense(request: AnalyticsInsightsRequest) -> float:
        return request.totalExpenseLookback or 0.0

    @staticmethod
    def _lookback_net_cashflow(request: AnalyticsInsightsRequest) -> float:
        if request.netCashflowLookback is not None:
            return request.netCashflowLookback
        return AnalyticsInsightsService._lookback_income(request) - AnalyticsInsightsService._lookback_expense(request)
