from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PortfolioAssetInput(BaseModel):
    """Single asset in portfolio for insights generation."""
    symbol: str
    quantity: float
    averagePrice: float
    currentPrice: float
    marketValue: float
    unrealizedPnL: float
    unrealizedPnLPct: float
    industryName: str | None = None


class PortfolioInsightsRequest(BaseModel):
    """Request for portfolio insights generation."""
    portfolioName: str
    totalMarketValue: float
    totalCostBasis: float
    cashBalance: float
    unrealizedPnL: float
    unrealizedPnLPct: float
    assets: list[PortfolioAssetInput] = Field(default_factory=list)

    # Personal finance context (optional)
    monthlyExpenses: float | None = None
    liquidAssets: float | None = None
    monthlyInvestRatio: float | None = None


class PortfolioInsightItem(BaseModel):
    """Single insight about portfolio."""
    category: Literal["nhan_xet", "canh_bao", "loi_khuyen"]
    message: str


class PortfolioInsightsResponse(BaseModel):
    """Response containing portfolio insights."""
    insights: list[PortfolioInsightItem] = Field(default_factory=list)
