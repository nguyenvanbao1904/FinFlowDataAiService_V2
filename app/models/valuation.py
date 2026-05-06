from __future__ import annotations

from pydantic import BaseModel, Field


class FairValueRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    targetYear: int | None = None


class FairValueResponse(BaseModel):
    symbol: str
    companyName: str
    targetYear: int
    industryKey: str
    method: str
    weightsUsed: str
    priceComposite: float
    pricePE: float
    pricePB: float
    pricePS: float
    livePrice: float
    upsidePct: float
    verdict: str
    peTarget: float
    pbTarget: float
    cagr: float
    error: str | None = None
