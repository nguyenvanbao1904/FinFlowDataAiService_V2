from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CategoryInsightInput(BaseModel):
    name: str
    amount: float
    sharePct: Optional[float] = None


class MonthlyTrendPoint(BaseModel):
    month: str
    income: float = 0.0
    expense: float = 0.0
    net: float = 0.0
    topExpenseCategories: list[CategoryInsightInput] = Field(default_factory=list)


class CategoryDeltaInput(BaseModel):
    name: str
    previousAmount: float = 0.0
    baselineAvgAmount: float = 0.0
    deltaPct: float = 0.0


class SavingsRatePoint(BaseModel):
    month: str
    savingsRatePct: float = 0.0


class AnalyticsInsightsRequest(BaseModel):
    cacheKey: str = Field(min_length=1, max_length=128)
    locale: str = "vi-VN"
    timezone: str = "Asia/Ho_Chi_Minh"
    currency: str = "VND"
    periodLabel: str = "THANG_NAY"
    insightTier: Literal["FULL", "SPARSE"] = "FULL"
    recentTransactionCount: int | None = None
    asOfDate: str = ""
    currentMonthLabel: str = ""
    previousMonthLabel: str = ""
    lookbackLabel: str = ""
    currentDayOfMonth: int | None = None
    isBeginningOfMonth: bool | None = None
    totalIncomeLookback: float | None = None
    totalExpenseLookback: float | None = None
    netCashflowLookback: float | None = None
    avgIncomePrev2Months: float | None = None
    avgExpensePrev2Months: float | None = None
    savingsRateSeries: list[SavingsRatePoint] = Field(default_factory=list)
    previousMonthCategoryDelta: list[CategoryDeltaInput] = Field(default_factory=list)
    previousMonthTopExpenseCategories: list[CategoryInsightInput] = Field(default_factory=list)
    monthlySeries: list[MonthlyTrendPoint] = Field(default_factory=list)


class AnalyticsInsightItem(BaseModel):
    id: str
    type: Literal["WARNING", "TIP"]
    title: str
    message: str
    confidence: float = 0.0


class AnalyticsInsightsResponse(BaseModel):
    insights: list[AnalyticsInsightItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cached: bool = False
