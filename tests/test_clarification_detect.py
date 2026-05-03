"""needs_clarification heuristic: only flag genuinely blocked turns."""
from __future__ import annotations

import pytest

from app.services.chat.orchestrator import _needs_clarification


@pytest.mark.parametrize("msg", [
    "Bạn muốn hỏi về mã cổ phiếu nào?",
    "Vui lòng cho tôi biết mã cổ phiếu cần định giá.",
    "Bạn đang hỏi về mã nào ạ?",
    "Cho tôi biết mã cổ phiếu để tôi tra cứu giúp.",
])
def test_blocking_patterns_flagged(msg):
    assert _needs_clarification(msg) is True


@pytest.mark.parametrize("msg", [
    "Có gì đó không ổn?",     # short trailing question, ambiguous
    "Mã cổ phiếu nào ạ?",     # short, just asking
])
def test_short_trailing_question_flagged(msg):
    assert _needs_clarification(msg) is True


def test_long_answer_with_followup_question_not_flagged():
    """The bug: long ACB analysis ending with 'Bạn có muốn xem thêm...?'
    must NOT be flagged as clarification."""
    msg = (
        "Định giá ACB đã hoàn tất. Giá hợp lý 2030 ước ~45,700đ, giá hiện tại "
        "23,500đ, tiềm năng tăng +94.5%. Đây là cơ hội đầu tư dài hạn hấp dẫn. "
        "ACB duy trì ROE >20% là điểm cộng lớn. Tỷ lệ nợ xấu thấp, "
        "chi phí hoạt động đang được tối ưu. "
        "Bạn có muốn xem thêm thông tin chiến lược chi tiết không?"
    )
    assert _needs_clarification(msg) is False


def test_plain_answer_not_flagged():
    msg = "Tháng này bạn thu 20 triệu, chi 14 triệu, dòng tiền dương 6 triệu."
    assert _needs_clarification(msg) is False


def test_empty_not_flagged():
    assert _needs_clarification("") is False
