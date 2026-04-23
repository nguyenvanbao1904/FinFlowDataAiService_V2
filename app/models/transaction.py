from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CategoryCandidate(BaseModel):
    id: str
    name: str
    type: Optional[str] = None


class AccountCandidate(BaseModel):
    id: str
    name: str
    transactionEligible: Optional[bool] = None


class TransactionPrefillRequest(BaseModel):
    rawText: str = Field(min_length=1, max_length=4000)
    categories: list[CategoryCandidate] = Field(default_factory=list)
    accounts: list[AccountCandidate] = Field(default_factory=list)
    recentHistory: list[dict[str, Any]] = Field(default_factory=list)
    locale: str = "vi-VN"
    timezone: str = "Asia/Ho_Chi_Minh"
    source: Literal["text", "ocr"] = "text"


class TransactionPrefillResponse(BaseModel):
    amount: Optional[float] = None
    type: Optional[Literal["INCOME", "EXPENSE"]] = None
    categoryId: Optional[str] = None
    accountId: Optional[str] = None
    note: Optional[str] = None
    transactionDate: Optional[str] = None
    confidence: float = 0.0
    missingFields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
