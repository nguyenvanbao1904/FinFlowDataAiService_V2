from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings

from app.infrastructure.llm_agent import get_deepseek_model
from app.models.portfolio import (
    PortfolioInsightsRequest,
    PortfolioInsightsResponse,
    PortfolioInsightItem,
)
from app.services.chat.utils.json_io import parse_llm_json

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a concise Vietnamese investment portfolio advisor. "
    "Return ONLY a raw JSON array. No markdown, no code fences, no prose. "
    "Analyze the portfolio data and provide exactly 3 insights in order: "
    "1. nhan_xet (overall assessment), 2. canh_bao (main risk), 3. loi_khuyen (actionable advice)."
)


class PortfolioInsightsService:
    def __init__(self) -> None:
        self._agent = Agent(
            get_deepseek_model(),
            system_prompt=_SYSTEM_PROMPT,
            model_settings=OpenAIChatModelSettings(
                temperature=0.0,
                max_tokens=300,
                extra_body={"response_format": {"type": "json_object"}},
            ),
            output_type=str,
        )

    async def generate(self, request: PortfolioInsightsRequest) -> PortfolioInsightsResponse:
        if not request.assets or request.totalMarketValue <= 0:
            logger.info("Portfolio insights skip: no assets or zero market value")
            return PortfolioInsightsResponse(insights=[])

        try:
            logger.info("Portfolio insights LLM call start: portfolio=%s assets=%d",
                       request.portfolioName, len(request.assets))
            raw = await self._call_llm(request)
            insights = self._parse_insights(raw)
            logger.info("Portfolio insights LLM call success: insights=%d", len(insights))
            return PortfolioInsightsResponse(insights=insights)
        except Exception as exc:
            logger.warning(
                "Portfolio insights error: type=%s detail=%s",
                type(exc).__name__,
                exc,
            )
            return PortfolioInsightsResponse(insights=[])

    async def _call_llm(self, request: PortfolioInsightsRequest) -> dict:
        result = await self._agent.run(self._build_prompt(request))
        parsed = parse_llm_json(result.output)
        if not isinstance(parsed, dict):
            parsed = {}
        return parsed

    def _build_prompt(self, request: PortfolioInsightsRequest) -> str:
        # Build asset summary
        asset_lines = []
        for asset in request.assets:
            pnl_sign = "+" if asset.unrealizedPnL >= 0 else ""
            pct_sign = "+" if asset.unrealizedPnLPct >= 0 else ""
            weight = (asset.marketValue / request.totalMarketValue * 100) if request.totalMarketValue > 0 else 0
            industry = f" ({asset.industryName})" if asset.industryName else ""
            asset_lines.append(
                f"- {asset.symbol}{industry}: {weight:.1f}% danh mục, "
                f"lãi/lỗ {pnl_sign}{asset.unrealizedPnL:,.0f} ({pct_sign}{asset.unrealizedPnLPct:.1f}%)"
            )

        assets_text = "\n".join(asset_lines)

        # Personal finance context
        fsi_note = ""
        if request.monthlyExpenses and request.liquidAssets:
            runway = request.liquidAssets / request.monthlyExpenses
            if runway < 3:
                fsi_note += f" Quỹ dự phòng còn {runway:.1f} tháng (dưới mức an toàn 3 tháng)."
        if request.monthlyInvestRatio and request.monthlyInvestRatio > 0.80:
            fsi_note += f" ~{int(round(request.monthlyInvestRatio * 100))}% thu nhập thặng dư đang đổ vào đầu tư."

        pnl_sign = "+" if request.unrealizedPnL >= 0 else ""
        pct_sign = "+" if request.unrealizedPnLPct >= 0 else ""

        return f"""Đánh giá danh mục "{request.portfolioName}":

Tổng quan:
- Tổng giá trị: {request.totalMarketValue:,.0f} VND
- Giá vốn: {request.totalCostBasis:,.0f} VND
- Tiền mặt: {request.cashBalance:,.0f} VND
- Lãi/Lỗ tạm tính: {pnl_sign}{request.unrealizedPnL:,.0f} ({pct_sign}{request.unrealizedPnLPct:.1f}%)

Cơ cấu danh mục ({len(request.assets)} mã):
{assets_text}
{fsi_note}

Trả lời bằng JSON object với key "insights" chứa array ĐÚNG 3 mục theo thứ tự:
1. {{"category": "nhan_xet", "message": "Tình trạng tổng thể: danh mục đang lãi/lỗ bao nhiêu, có ổn không (1 câu ngắn)"}}
2. {{"category": "canh_bao", "message": "Rủi ro chính: tập trung quá cao, thiếu đa dạng hóa, hoặc vấn đề tài chính cá nhân (1 câu ngắn)"}}
3. {{"category": "loi_khuyen", "message": "Hành động cụ thể: nên làm gì tiếp theo để cải thiện (1 câu ngắn)"}}

Format: {{"insights": [{{"category": "nhan_xet", "message": "..."}}, {{"category": "canh_bao", "message": "..."}}, {{"category": "loi_khuyen", "message": "..."}}]}}

Chỉ trả JSON object, không thêm text nào khác. Mỗi message ngắn gọn, có số liệu cụ thể.
"""

    def _parse_insights(self, raw: dict) -> list[PortfolioInsightItem]:
        """Parse LLM response to list of insights."""
        insights_raw = raw.get("insights", [])
        if not isinstance(insights_raw, list):
            logger.warning("LLM response 'insights' is not a list: %s", type(insights_raw))
            return []

        insights = []
        for item in insights_raw:
            if not isinstance(item, dict):
                continue
            category = item.get("category", "")
            message = item.get("message", "")
            if category in ["nhan_xet", "canh_bao", "loi_khuyen"] and message:
                insights.append(PortfolioInsightItem(category=category, message=message))

        return insights
