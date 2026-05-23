import datetime

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import (
    get_analytics_service,
    get_home_insight_service,
    get_prefill_service,
    get_portfolio_insights_service,
    require_internal_api_key,
)
from app.infrastructure.market_data_client import MarketDataToolClient
from app.models.analytics import AnalyticsInsightsRequest, AnalyticsInsightsResponse
from app.models.home import HomeInsightRequest, HomeInsightResponse
from app.models.portfolio import PortfolioInsightsRequest, PortfolioInsightsResponse
from app.models.transaction import TransactionPrefillRequest, TransactionPrefillResponse
from app.models.valuation import FairValueRequest, FairValueResponse
from app.services.chat.valuation_engine import compute_fair_value
from app.services.chat.valuation_inputs import fetch_valuation_inputs

router = APIRouter(prefix=f"{settings.API_V1_STR}/ai", dependencies=[Depends(require_internal_api_key)])

_market_client: MarketDataToolClient | None = None


def _get_market_client() -> MarketDataToolClient:
    global _market_client
    if _market_client is None:
        _market_client = MarketDataToolClient()
    return _market_client


def _result_to_response(result: dict, fallback_symbol: str, fallback_year: int) -> FairValueResponse:
    """Map valuation_engine output dict → FairValueResponse.

    Single place to update if valuation_engine adds/renames fields.
    """
    return FairValueResponse(
        symbol=result.get("symbol", fallback_symbol),
        companyName=result.get("company_name", fallback_symbol),
        targetYear=result.get("target_year", fallback_year),
        industryKey=result.get("industry_key", "UNKNOWN"),
        method=result.get("method", ""),
        weightsUsed=result.get("weights_used", ""),
        priceComposite=result.get("price_composite", 0),
        pricePE=result.get("price_pe", 0),
        pricePB=result.get("price_pb", 0),
        pricePS=result.get("price_ps", 0),
        livePrice=result.get("live_price", 0),
        upsidePct=result.get("upside_pct", 0),
        verdict=result.get("verdict", ""),
        peTarget=result.get("pe_target", 0),
        pbTarget=result.get("pb_target", 0),
        cagr=result.get("cagr_pct", 0),
        valuationModel=result.get("valuation_model"),
        valuationFormula=result.get("valuation_formula"),
        modelReason=result.get("model_reason"),
        modelConfidence=result.get("model_confidence"),
        keyAssumptions=result.get("key_assumptions"),
    )


@router.post("/transaction-prefill", response_model=TransactionPrefillResponse)
async def transaction_prefill(request: TransactionPrefillRequest) -> TransactionPrefillResponse:
    return await get_prefill_service().prefill(request)


@router.post("/analytics-insights", response_model=AnalyticsInsightsResponse)
async def analytics_insights(request: AnalyticsInsightsRequest) -> AnalyticsInsightsResponse:
    return await get_analytics_service().generate(request)


@router.post("/home-insight", response_model=HomeInsightResponse)
async def home_insight(request: HomeInsightRequest) -> HomeInsightResponse:
    return await get_home_insight_service().generate(request)


@router.post("/portfolio-insights", response_model=PortfolioInsightsResponse)
async def portfolio_insights(request: PortfolioInsightsRequest) -> PortfolioInsightsResponse:
    return await get_portfolio_insights_service().generate(request)


@router.post("/fair-value", response_model=FairValueResponse)
async def fair_value(request: FairValueRequest) -> FairValueResponse:
    """Compute AI-powered fair value for a stock symbol and target year.

    Uses the same fetch_valuation_inputs + compute_fair_value pipeline as the chat agent tool.
    """
    yr = request.targetYear or datetime.date.today().year
    inputs = await fetch_valuation_inputs(_get_market_client(), request.symbol, request.targetYear)

    if "error" in inputs:
        return FairValueResponse(
            symbol=request.symbol,
            companyName=request.symbol,
            targetYear=yr,
            industryKey="UNKNOWN",
            method="N/A",
            weightsUsed="N/A",
            priceComposite=0,
            pricePE=0,
            pricePB=0,
            pricePS=0,
            livePrice=0,
            upsidePct=0,
            verdict="KHÔNG CÓ DỮ LIỆU",
            peTarget=0,
            pbTarget=0,
            cagr=0,
            error=inputs["error"],
        )

    result = compute_fair_value(inputs)
    return _result_to_response(result, request.symbol, yr)
