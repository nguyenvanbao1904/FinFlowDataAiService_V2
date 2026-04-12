from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

from app.core.config import settings
from app.core.http_client import close_http_client
from app.domain.entities.analytics_insights import (
    AnalyticsInsightsRequest,
    AnalyticsInsightsResponse,
)
from app.domain.entities.transaction_prefill import (
    TransactionPrefillRequest,
    TransactionPrefillResponse,
)
from app.domain.entities.chat_orchestrator import (
    ChatOrchestrateRequest,
    ChatOrchestrateResponse,
    ThreadSummaryRequest,
    ThreadSummaryResponse,
)
from app.services.analytics_insights_service import AnalyticsInsightsService
from app.services.transaction_prefill_service import TransactionPrefillService
from app.services.chat_orchestrator_service import ChatOrchestratorService

logger = logging.getLogger(__name__)

analytics_insights_service = AnalyticsInsightsService()
transaction_prefill_service = TransactionPrefillService()
chat_orchestrator_service = ChatOrchestratorService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_http_client()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Microservice for Data Crawling and AI Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers (replaces per-endpoint try/except) ─────

@app.exception_handler(httpx.TimeoutException)
async def _handle_llm_timeout(request: Request, exc: httpx.TimeoutException) -> JSONResponse:
    logger.warning("Upstream timeout: %s", exc)
    return JSONResponse(status_code=504, content={"detail": "LLM timeout"})


@app.exception_handler(httpx.HTTPStatusError)
async def _handle_llm_upstream(request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
    logger.warning("Upstream HTTP error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "LLM upstream error"})


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Welcome to FinFlow Data & AI Service"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def _ensure_internal_api_key(internal_api_key: str | None) -> None:
    if settings.INTERNAL_API_KEY:
        if not internal_api_key or internal_api_key != settings.INTERNAL_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized internal API key")


@app.post(
    f"{settings.API_V1_STR}/ai/transaction-prefill",
    response_model=TransactionPrefillResponse,
)
async def transaction_prefill(
    request: TransactionPrefillRequest,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> TransactionPrefillResponse:
    _ensure_internal_api_key(internal_api_key)
    return await transaction_prefill_service.prefill(request)


@app.post(
    f"{settings.API_V1_STR}/ai/analytics-insights",
    response_model=AnalyticsInsightsResponse,
)
async def analytics_insights(
    request: AnalyticsInsightsRequest,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> AnalyticsInsightsResponse:
    _ensure_internal_api_key(internal_api_key)
    return await analytics_insights_service.generate(request)


@app.post(
    f"{settings.API_V1_STR}/ai/chat/orchestrate",
    response_model=ChatOrchestrateResponse,
)
async def chat_orchestrate(
    request: ChatOrchestrateRequest,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> ChatOrchestrateResponse:
    _ensure_internal_api_key(internal_api_key)
    return await chat_orchestrator_service.orchestrate(request)


@app.post(
    f"{settings.API_V1_STR}/ai/chat/thread-summary",
    response_model=ThreadSummaryResponse,
)
async def chat_thread_summary(
    request: ThreadSummaryRequest,
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> ThreadSummaryResponse:
    _ensure_internal_api_key(internal_api_key)
    return await chat_orchestrator_service.summarize_thread(request)
