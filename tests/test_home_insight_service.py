from __future__ import annotations

from app.models.home import HomeInsightRequest
from app.services.home_insight_service import HomeInsightService
from tests.conftest import make_json_response


def _request() -> HomeInsightRequest:
    return HomeInsightRequest(
        liquidAssets=25_000_000,
        totalIncome=30_000_000,
        totalExpense=18_000_000,
        portfolioCashTotal=8_000_000,
    )


async def test_home_insight_accepts_valid_llm_json(deepseek_mock) -> None:
    service = HomeInsightService()
    payload = {
        "title": "Gợi ý hôm nay",
        "message": "Dòng tiền tháng đang dương; hãy chuyển một phần vào quỹ dự phòng.",
        "warnings": [],
    }

    with deepseek_mock(responses=[make_json_response(payload)]):
        result = await service.generate(_request())

    assert result.title == "Gợi ý hôm nay"
    assert result.message == payload["message"]
    assert result.warnings == []
    assert result.cached is False


async def test_home_insight_rejects_ambiguous_money_phrase(deepseek_mock) -> None:
    service = HomeInsightService()
    payload = {
        "title": "Gợi ý hôm nay",
        "message": "Bạn có tiền nhàn rỗi; hãy cân nhắc đầu tư thêm.",
        "warnings": [],
    }

    with deepseek_mock(responses=[make_json_response(payload)]):
        result = await service.generate(_request())

    assert "tiền nhàn rỗi" not in result.message.lower()
    assert result.warnings == ["ambiguous_money_reference"]
