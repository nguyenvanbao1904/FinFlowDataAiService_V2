from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings

from app.infrastructure.llm_agent import get_deepseek_model
from app.models.home import HomeInsightRequest, HomeInsightResponse
from app.services.chat.utils.json_io import parse_llm_json

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a concise Vietnamese personal finance home insight writer. "
    "Return ONLY a raw JSON object. No markdown, no code fences, no prose. "
    "Use only the provided pre-calculated facts. Do not invent numbers."
)

_AMBIGUOUS_MONEY_PATTERNS = (
    "tien mat nhan roi",
    "tien nhan roi",
    "khoan tien nhan roi",
)


class HomeInsightService:
    def __init__(self) -> None:
        self._agent = Agent(
            get_deepseek_model(),
            system_prompt=_SYSTEM_PROMPT,
            model_settings=OpenAIChatModelSettings(
                temperature=0.0,
                max_tokens=180,
                extra_body={"response_format": {"type": "json_object"}},
            ),
            output_type=str,
        )

    async def generate(self, request: HomeInsightRequest) -> HomeInsightResponse:
        try:
            logger.info("Home insight LLM call start")
            result = await self._agent.run(self._build_prompt(request))
            parsed = parse_llm_json(result.output)
            if not isinstance(parsed, dict):
                parsed = {}
            response = self._normalize(parsed, request)
            logger.info("Home insight LLM call success")
            return response
        except Exception as exc:
            logger.warning("Home insight fallback: type=%s detail=%s", type(exc).__name__, exc)
            return self._fallback_response(request, warn=f"LLM error: {type(exc).__name__}")

    def _build_prompt(self, request: HomeInsightRequest) -> str:
        monthly_cashflow = request.totalIncome - request.totalExpense
        budget_ratio = (
            (request.budgetSpentTotal / request.budgetTargetTotal) * 100
            if request.budgetTargetTotal > 0
            else 0
        )
        return f"""
STRICT RULES:
- Return exactly this JSON schema: {{"title": "Gợi ý hôm nay", "message": string, "warnings": string[]}}
- message must be Vietnamese, friendly, 1 sentence, <= 22 words, <= 120 characters.
- Mention the single most useful signal and one next action.
- If mentioning money, name its source explicitly: "thanh khoản", "dòng tiền tháng", or "tiền mặt trong danh mục".
- Do NOT use vague phrases like "tiền nhàn rỗi", "tiền mặt nhàn rỗi", or "khoản tiền nhàn rỗi".
- Do not sound like advertising.
- Do not include markdown or bullets.

CONTEXT:
locale={request.locale}
timezone={request.timezone}
currency={request.currency}
netWorth={request.netWorth}
liquidAssets={request.liquidAssets}
debtTotal={request.debtTotal}
investmentAssets={request.investmentAssets}
totalBalance={request.totalBalance}
totalIncome={request.totalIncome}
totalExpense={request.totalExpense}
monthlyCashflow={monthly_cashflow}
budgetTargetTotal={request.budgetTargetTotal}
budgetSpentTotal={request.budgetSpentTotal}
budgetUsagePct={budget_ratio}
portfolioCount={request.portfolioCount}
portfolioCashTotal={request.portfolioCashTotal}
primaryPortfolioName={request.primaryPortfolioName or ""}
investmentTotalValue={request.investmentTotalValue}
""".strip()

    def _normalize(self, raw: dict[str, Any], request: HomeInsightRequest) -> HomeInsightResponse:
        title = self._clean_text(raw.get("title")) or "Gợi ý hôm nay"
        message = self._clean_text(raw.get("message"))
        warnings = raw.get("warnings") if isinstance(raw.get("warnings"), list) else []
        clean_warnings = [self._clean_text(item) for item in warnings]
        clean_warnings = [item for item in clean_warnings if item]

        if not message:
            return self._fallback_response(request, warn="empty_message")
        if self._has_ambiguous_money_reference(message):
            return self._fallback_response(request, warn="ambiguous_money_reference")

        return HomeInsightResponse(
            title=title,
            message=self._shorten(message, 120),
            warnings=clean_warnings,
            cached=False,
        )

    def _fallback_response(self, request: HomeInsightRequest, warn: str | None = None) -> HomeInsightResponse:
        monthly_cashflow = request.totalIncome - request.totalExpense
        if request.portfolioCashTotal > 0:
            message = (
                f"Tiền mặt trong danh mục còn {self._compact_vnd(request.portfolioCashTotal)}; "
                "hãy lên kế hoạch giải ngân."
            )
        elif monthly_cashflow > 0:
            message = (
                f"Dòng tiền tháng dương {self._compact_vnd(monthly_cashflow)}; "
                "hãy phân bổ vào quỹ dự phòng hoặc đầu tư."
            )
        else:
            message = (
                f"Thanh khoản đang ở {self._compact_vnd(request.liquidAssets)}; "
                "hãy giữ đủ quỹ dự phòng."
            )
        warnings = [warn] if warn else []
        return HomeInsightResponse(
            title="Gợi ý hôm nay",
            message=self._shorten(message, 120),
            warnings=warnings,
            cached=False,
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    @staticmethod
    def _shorten(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        shortened = text[:max_length]
        if " " in shortened:
            shortened = shortened.rsplit(" ", 1)[0]
        return shortened.rstrip(" .,;:") + "."

    @staticmethod
    def _compact_vnd(value: float) -> str:
        abs_value = abs(value)
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f} tỷ"
        if abs_value >= 1_000_000:
            return f"{round(value / 1_000_000):.0f} triệu"
        return f"{value:,.0f} đ".replace(",", ".")

    @staticmethod
    def _has_ambiguous_money_reference(text: str) -> bool:
        without_marks = unicodedata.normalize("NFKD", text.lower())
        normalized = "".join(ch for ch in without_marks if not unicodedata.combining(ch)).replace("đ", "d")
        return any(pattern in normalized for pattern in _AMBIGUOUS_MONEY_PATTERNS)
