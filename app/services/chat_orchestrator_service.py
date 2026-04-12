"""Thin facade — preserves the old import path ``ChatOrchestratorService``.

All logic has been refactored into the ``app.services.chat`` package.
This file only exists so ``main.py`` and any other import sites continue
to work without modification.
"""
from __future__ import annotations

from app.domain.entities.chat_orchestrator import (
    ChatOrchestrateRequest,
    ChatOrchestrateResponse,
    ThreadSummaryRequest,
    ThreadSummaryResponse,
)
from app.services.chat.orchestrator import ChatOrchestrator


class ChatOrchestratorService:
    """Backwards-compatible wrapper — delegates to the new ChatOrchestrator."""

    def __init__(self) -> None:
        self._impl = ChatOrchestrator()

    async def orchestrate(self, request: ChatOrchestrateRequest) -> ChatOrchestrateResponse:
        return await self._impl.orchestrate(request)

    async def summarize_thread(self, request: ThreadSummaryRequest) -> ThreadSummaryResponse:
        return await self._impl.summarize_thread(request)
