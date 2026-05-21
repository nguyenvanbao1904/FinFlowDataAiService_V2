from __future__ import annotations

from pydantic import BaseModel, Field


class HomeInsightRequest(BaseModel):
    locale: str = "vi-VN"
    timezone: str = "Asia/Ho_Chi_Minh"
    currency: str = "VND"
    netWorth: float = 0.0
    liquidAssets: float = 0.0
    debtTotal: float = 0.0
    investmentAssets: float = 0.0
    totalBalance: float = 0.0
    totalIncome: float = 0.0
    totalExpense: float = 0.0
    budgetTargetTotal: float = 0.0
    budgetSpentTotal: float = 0.0
    portfolioCount: int = 0
    portfolioCashTotal: float = 0.0
    primaryPortfolioName: str | None = None
    investmentTotalValue: float = 0.0


class HomeInsightResponse(BaseModel):
    title: str = "Gợi ý hôm nay"
    message: str = Field(default="")
    warnings: list[str] = Field(default_factory=list)
    cached: bool = False
