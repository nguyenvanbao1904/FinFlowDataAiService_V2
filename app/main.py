from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from app.core.config import settings
from app.domain.entities.analytics_insights import (
    AnalyticsInsightsRequest,
    AnalyticsInsightsResponse,
)
from app.domain.entities.transaction_prefill import (
    TransactionPrefillRequest,
    TransactionPrefillResponse,
)
from app.services.analytics_insights_service import AnalyticsInsightsService
from app.services.transaction_prefill_service import TransactionPrefillService

analytics_insights_service = AnalyticsInsightsService()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Microservice for Data Crawling and AI Analysis",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to FinFlow Data & AI Service"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post(
    f"{settings.API_V1_STR}/ai/transaction-prefill",
    response_model=TransactionPrefillResponse,
)
async def transaction_prefill(
    request: TransactionPrefillRequest,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> TransactionPrefillResponse:
    if settings.INTERNAL_API_KEY:
        if not internal_api_key or internal_api_key != settings.INTERNAL_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized internal API key")

    service = TransactionPrefillService()
    try:
        return await service.prefill(request)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="LLM timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail="LLM upstream error") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to prefill transaction") from exc


@app.post(
    f"{settings.API_V1_STR}/ai/analytics-insights",
    response_model=AnalyticsInsightsResponse,
)
async def analytics_insights(
    request: AnalyticsInsightsRequest,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> AnalyticsInsightsResponse:
    if settings.INTERNAL_API_KEY:
        if not internal_api_key or internal_api_key != settings.INTERNAL_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized internal API key")

    try:
        return await analytics_insights_service.generate(request)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="LLM timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail="LLM upstream error") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate analytics insights") from exc
