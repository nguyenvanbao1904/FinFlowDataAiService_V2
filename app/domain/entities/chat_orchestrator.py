from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurnMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(min_length=1, max_length=12000)
    created_at: str | None = None


class ChatOrchestrateRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=64)
    user_message: str = Field(min_length=1, max_length=8000)
    context_summary: str | None = ""
    last_messages: list[ChatTurnMessage] = Field(default_factory=list)


class ChatCitation(BaseModel):
    chunk_id: str | None = None
    source_title: str | None = None
    page_number: int | None = None
    score: float | None = None


class ChatOrchestrateResponse(BaseModel):
    assistant_message: str
    needs_clarification: bool = False
    clarification_question: str | None = None
    provider: str = "deepseek"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[ChatCitation] = Field(default_factory=list)
    context_update: dict[str, Any] = Field(default_factory=dict)


class ThreadSummaryRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=64)
    existing_summary: str | None = ""
    recent_messages: list[ChatTurnMessage] = Field(default_factory=list)


class ThreadSummaryResponse(BaseModel):
    context_summary: str
    current_ticker: str | None = None
    current_period: int | None = None
    provider: str = "deepseek"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
