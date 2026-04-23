from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import get_chat_orchestrator, require_internal_api_key
from app.models.chat import (
    ChatOrchestrateRequest,
    ChatOrchestrateResponse,
    ThreadSummaryRequest,
    ThreadSummaryResponse,
)

router = APIRouter(prefix=f"{settings.API_V1_STR}/ai/chat", dependencies=[Depends(require_internal_api_key)])


@router.post("/orchestrate", response_model=ChatOrchestrateResponse)
async def chat_orchestrate(request: ChatOrchestrateRequest) -> ChatOrchestrateResponse:
    return await get_chat_orchestrator().orchestrate(request)


@router.post("/thread-summary", response_model=ThreadSummaryResponse)
async def chat_thread_summary(request: ThreadSummaryRequest) -> ThreadSummaryResponse:
    return await get_chat_orchestrator().summarize_thread(request)
