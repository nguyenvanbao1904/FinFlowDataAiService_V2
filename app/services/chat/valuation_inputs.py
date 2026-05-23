"""Fan-out backend calls needed by compute_fair_value, then shape them into
the input dict expected by valuation_engine.compute_fair_value().

Kept separate from agent_tools.py so the data-fetch logic stays close to
its consumer (the valuation engine) rather than the LLM tool registry.
"""
from __future__ import annotations

import asyncio
import datetime
from typing import Any

from app.infrastructure.market_data_client import MarketDataToolClient


async def fetch_valuation_inputs(
    market_client: MarketDataToolClient,
    symbol: str,
    target_year: int | None = None,
) -> dict[str, Any]:
    """Gather every input compute_fair_value needs, in parallel.

    Returns either the inputs dict or {"error": "..."} when a critical
    dependency could not be fetched.
    """
    today = datetime.date.today()
    five_years_ago = today.replace(year=today.year - 5)
    yr = target_year or today.year

    forecast_years = [yr]
    if yr > today.year:
        forecast_years = list(range(today.year, yr + 1))

    metrics_r, financial_r, daily_val_r, live_r, *forecast_results = (
        await asyncio.gather(
            market_client.execute_tool_call("get_company_metrics", {"symbol": symbol}),
            market_client.execute_tool_call(
                "get_company_financial_series",
                {"symbol": symbol, "annualLimit": 6},
            ),
            market_client.execute_tool_call(
                "get_company_daily_valuations",
                {
                    "symbol": symbol,
                    "startDate": five_years_ago.isoformat(),
                    "endDate": today.isoformat(),
                },
            ),
            market_client.execute_tool_call(
                "get_company_live_valuation_snapshot", {"symbol": symbol},
            ),
            *[
                market_client.execute_tool_call(
                    "get_company_forecast",
                    {"symbol": symbol, "targetYear": forecast_year},
                )
                for forecast_year in forecast_years
            ],
        )
    )

    errors: list[str] = []
    for label, result in [
        ("metrics", metrics_r),
        ("financial", financial_r),
        ("live", live_r),
    ]:
        if not result.get("ok"):
            errors.append(f"{label}: {result.get('error_message', 'unknown')}")
    if errors:
        return {"error": f"Không lấy được dữ liệu: {'; '.join(errors)}"}

    overview = (metrics_r.get("data") or {}).get("overview") or {}

    fin_data = financial_r.get("data") or {}
    raw_entries = fin_data.get("nonBank") or fin_data.get("bank") or []
    profit_history = [
        {
            "year": item["year"],
            "profit_after_tax": item.get("profitAfterTax", 0),
            "eps": item.get("eps"),
            "bvps": item.get("bvps"),
            "roe": item.get("roe"),
            "pe": item.get("pe"),
            "pb": item.get("pb"),
            "payout_ratio": item.get("payoutRatio"),
            "cash_dividend": item.get("cashDividend"),
            "profit_growth": item.get("profitGrowth"),
            "shares_outstanding": item.get("shareAtPeriodEnd"),
        }
        for item in raw_entries
        if isinstance(item, dict) and "year" in item
        and (item.get("quarter") is None or item.get("quarter") == 0)
        and item.get("quarterCount", 0) == 4
    ]

    daily_summary = (daily_val_r.get("data") or {}).get("summary") or {}
    live_data = live_r.get("data") or {}

    forecast_series: list[dict[str, Any]] = []
    forecast_top_factors: dict = {}
    forecast_quality: dict = {}
    for result in forecast_results:
        data = result.get("data") if result.get("ok") else None
        if not isinstance(data, dict):
            continue
        forecast_series.append({
            "year": data.get("predict_target_year"),
            "revenue_pred": data.get("revenue_pred"),
            "profit_pred": data.get("profit_pred"),
            "feature_year": data.get("feature_year"),
        })
        if data.get("top_factors") and not forecast_top_factors:
            forecast_top_factors = data["top_factors"]
        if data.get("quality") and not forecast_quality:
            forecast_quality = data["quality"]

    forecast_data = next(
        (item for item in forecast_series if item.get("year") == yr),
        forecast_series[-1] if forecast_series else {},
    )

    return {
        "eps": overview.get("eps", 0),
        "bvps": overview.get("bvps", 0),
        "roe": overview.get("roe", 0),
        "live_price": live_data.get("livePriceVnd", 0),
        "profit_history": profit_history,
        "industry_icb_code": overview.get("industryIcbCode"),
        "industry_label": overview.get("industryLabel"),
        "median_pe": overview.get("medianPE") or daily_summary.get("pe_median"),
        "median_pb": overview.get("medianPB") or daily_summary.get("pb_median"),
        "median_ps": overview.get("medianPS") or daily_summary.get("ps_median"),
        "live_ps": live_data.get("livePs"),
        "forecast_profit": forecast_data.get("profit_pred"),
        "forecast_revenue": forecast_data.get("revenue_pred"),
        "forecast_series": forecast_series,
        "forecast_top_factors": forecast_top_factors,
        "forecast_quality": forecast_quality,
        "cplh": overview.get("cplh", 0),
        "symbol": symbol,
        "target_year": yr,
        "company_name": overview.get("companyName", symbol),
    }
