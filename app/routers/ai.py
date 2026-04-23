from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import (
    get_analytics_service,
    get_prefill_service,
    require_internal_api_key,
)
from app.models.analytics import AnalyticsInsightsRequest, AnalyticsInsightsResponse
from app.models.transaction import TransactionPrefillRequest, TransactionPrefillResponse

router = APIRouter(prefix=f"{settings.API_V1_STR}/ai", dependencies=[Depends(require_internal_api_key)])


@router.post("/transaction-prefill", response_model=TransactionPrefillResponse)
async def transaction_prefill(request: TransactionPrefillRequest) -> TransactionPrefillResponse:
    return await get_prefill_service().prefill(request)


@router.post("/analytics-insights", response_model=AnalyticsInsightsResponse)
async def analytics_insights(request: AnalyticsInsightsRequest) -> AnalyticsInsightsResponse:
    return await get_analytics_service().generate(request)
