"""Capture baseline for analytics_service against mocked DeepSeek."""
from __future__ import annotations

import pytest

from app.models.analytics import (
    AnalyticsInsightsRequest,
    CategoryDeltaInput,
    CategoryInsightInput,
    MonthlyTrendPoint,
    SavingsRatePoint,
)
from tests.conftest import (
    load_baseline,
    make_json_response,
    save_baseline,
)


def _build_full_request() -> AnalyticsInsightsRequest:
    return AnalyticsInsightsRequest(
        cacheKey="test-key-1",
        locale="vi-VN",
        timezone="Asia/Ho_Chi_Minh",
        currency="VND",
        periodLabel="THANG_NAY",
        insightTier="FULL",
        recentTransactionCount=42,
        asOfDate="2026-05-01",
        currentMonthLabel="tháng 5",
        previousMonthLabel="tháng 4",
        lookbackLabel="3 tháng gần nhất",
        currentDayOfMonth=1,
        isBeginningOfMonth=True,
        totalIncomeLookback=60000000.0,
        totalExpenseLookback=42000000.0,
        netCashflowLookback=18000000.0,
        avgIncomePrev2Months=20000000.0,
        avgExpensePrev2Months=14000000.0,
        savingsRateSeries=[
            SavingsRatePoint(month="2026-03", savingsRatePct=20.0),
            SavingsRatePoint(month="2026-04", savingsRatePct=25.0),
        ],
        previousMonthCategoryDelta=[
            CategoryDeltaInput(name="Ăn uống", previousAmount=4000000, baselineAvgAmount=3000000, deltaPct=33.3),
        ],
        previousMonthTopExpenseCategories=[
            CategoryInsightInput(name="Ăn uống", amount=4000000, sharePct=28.5),
            CategoryInsightInput(name="Đi lại", amount=2000000, sharePct=14.2),
        ],
        monthlySeries=[
            MonthlyTrendPoint(month="2026-03", income=20000000, expense=14000000, net=6000000),
            MonthlyTrendPoint(month="2026-04", income=20000000, expense=14000000, net=6000000),
        ],
    )


def _build_sparse_request() -> AnalyticsInsightsRequest:
    return AnalyticsInsightsRequest(
        cacheKey="test-key-2",
        locale="vi-VN",
        timezone="Asia/Ho_Chi_Minh",
        currency="VND",
        periodLabel="THANG_NAY",
        insightTier="SPARSE",
        recentTransactionCount=5,
        asOfDate="2026-05-01",
        currentMonthLabel="tháng 5",
        previousMonthLabel="tháng 4",
        lookbackLabel="3 tháng gần nhất",
        currentDayOfMonth=1,
        isBeginningOfMonth=True,
        totalIncomeLookback=2000000.0,
        totalExpenseLookback=1500000.0,
        netCashflowLookback=500000.0,
        previousMonthTopExpenseCategories=[
            CategoryInsightInput(name="Ăn uống", amount=600000, sharePct=40.0),
        ],
        monthlySeries=[
            MonthlyTrendPoint(month="2026-03", income=1000000, expense=750000, net=250000),
            MonthlyTrendPoint(month="2026-04", income=1000000, expense=750000, net=250000),
        ],
    )


CASES = [
    {
        "name": "full_warning_tip",
        "build_request": _build_full_request,
        "llm_output": {
            "insights": [
                {
                    "id": "warn-1",
                    "type": "WARNING",
                    "title": "Cảnh báo chi ăn uống",
                    "message": "Chi ăn uống tăng 33% so với trung bình. Theo dõi sát.",
                    "confidence": 0.85,
                },
                {
                    "id": "tip-1",
                    "type": "TIP",
                    "title": "Cải thiện tiết kiệm",
                    "message": "Đặt hạn mức cho danh mục Ăn uống để cân bằng dòng tiền.",
                    "confidence": 0.8,
                },
            ],
            "warnings": [],
        },
    },
    {
        "name": "sparse_data",
        "build_request": _build_sparse_request,
        "llm_output": {
            "insights": [
                {
                    "id": "warn-1",
                    "type": "WARNING",
                    "title": "Theo dõi chi tiêu",
                    "message": "Dữ liệu còn ít. Ghi đều các giao dịch trong tháng tới.",
                    "confidence": 0.6,
                },
                {
                    "id": "tip-1",
                    "type": "TIP",
                    "title": "Mẹo phân loại",
                    "message": "Phân loại giao dịch theo danh mục để dễ so sánh.",
                    "confidence": 0.6,
                },
            ],
            "warnings": [],
        },
    },
]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
async def test_analytics_baseline(case, deepseek_mock, captured_requests):
    from app.services.analytics_service import AnalyticsInsightsService

    service = AnalyticsInsightsService()
    response_payload = make_json_response(case["llm_output"])

    request = case["build_request"]()
    with deepseek_mock(responses=[response_payload]):
        result = await service.generate(request)

    assert len(captured_requests) >= 1, "Expected at least 1 DeepSeek call"
    llm_request = captured_requests[0]["body"]

    max_tok = llm_request.get("max_tokens") or llm_request.get("max_completion_tokens")
    snapshot = {
        "llm_request": {
            "model": llm_request.get("model"),
            "temperature": llm_request.get("temperature"),
            "max_tokens_eff": max_tok,
            "response_format": llm_request.get("response_format"),
            "messages": llm_request.get("messages"),
        },
        "service_response": result.model_dump(mode="json"),
    }

    name = f"analytics_{case['name']}"
    existing = load_baseline(name)
    if existing is None:
        save_baseline(name, snapshot)
        pytest.skip(f"Baseline created: {name}.json")

    assert existing["llm_request"]["messages"] == snapshot["llm_request"]["messages"]
    assert existing["service_response"] == snapshot["service_response"]
