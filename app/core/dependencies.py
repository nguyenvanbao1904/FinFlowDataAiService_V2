import logging

from fastapi import Header, HTTPException

from app.core.config import settings
from app.services.analytics_service import AnalyticsInsightsService
from app.services.prefill_service import TransactionPrefillService
from app.services.chat.orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

_analytics_insights_service: AnalyticsInsightsService | None = None
_transaction_prefill_service: TransactionPrefillService | None = None
_chat_orchestrator: ChatOrchestrator | None = None


def get_analytics_service() -> AnalyticsInsightsService:
    global _analytics_insights_service
    if _analytics_insights_service is None:
        _analytics_insights_service = AnalyticsInsightsService()
    return _analytics_insights_service


def get_prefill_service() -> TransactionPrefillService:
    global _transaction_prefill_service
    if _transaction_prefill_service is None:
        _transaction_prefill_service = TransactionPrefillService()
    return _transaction_prefill_service


def get_chat_orchestrator() -> ChatOrchestrator:
    global _chat_orchestrator
    if _chat_orchestrator is None:
        _chat_orchestrator = ChatOrchestrator()
    return _chat_orchestrator


def require_internal_api_key(
    internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    if not settings.INTERNAL_API_KEY:
        logger.warning("INTERNAL_API_KEY is not configured — all requests are allowed")
        return
    if not internal_api_key or internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized internal API key")
