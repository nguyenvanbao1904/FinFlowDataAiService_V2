"""Capture baseline behavior of prefill_service against mocked DeepSeek.

Goal: snapshot the request body sent to DeepSeek + the final response, so
after pydantic-ai migration we can verify both stay equivalent.
"""
from __future__ import annotations

import re

import pytest

from app.models.transaction import (
    AccountCandidate,
    CategoryCandidate,
    TransactionPrefillRequest,
)
from tests.conftest import (
    load_baseline,
    make_json_response,
    save_baseline,
)


CATEGORIES = [
    CategoryCandidate(id="cat-food", name="Ăn uống", type="EXPENSE"),
    CategoryCandidate(id="cat-fuel", name="Đi lại", type="EXPENSE"),
    CategoryCandidate(id="cat-salary", name="Lương", type="INCOME"),
]
ACCOUNTS = [
    AccountCandidate(id="acc-cash", name="Tiền mặt", transactionEligible=True),
    AccountCandidate(id="acc-bank", name="Vietcombank", transactionEligible=True),
]


PREFILL_CASES = [
    {
        "name": "expense_food_30k",
        "raw_text": "ăn trưa 30k tại quán cơm",
        "llm_output": {
            "amount": 30000,
            "type": "EXPENSE",
            "categoryId": "cat-food",
            "accountId": "acc-cash",
            "note": "ăn trưa",
            "transactionDate": "2026-05-01T12:00:00+07:00",
            "confidence": 0.9,
            "warnings": [],
        },
    },
    {
        "name": "income_salary_20m",
        "raw_text": "nhận lương 20 triệu",
        "llm_output": {
            "amount": 20000000,
            "type": "INCOME",
            "categoryId": "cat-salary",
            "accountId": "acc-bank",
            "note": "nhận lương",
            "transactionDate": "2026-05-01T09:00:00+07:00",
            "confidence": 0.95,
            "warnings": [],
        },
    },
    {
        "name": "fuel_2_xi",
        "raw_text": "đổ xăng 2 xị",
        "llm_output": {
            "amount": 200000,
            "type": "EXPENSE",
            "categoryId": "cat-fuel",
            "accountId": "acc-cash",
            "note": "đổ xăng",
            "transactionDate": "2026-05-01T08:00:00+07:00",
            "confidence": 0.85,
            "warnings": [],
        },
    },
]


def _build_request(raw: str) -> TransactionPrefillRequest:
    return TransactionPrefillRequest(
        rawText=raw,
        categories=CATEGORIES,
        accounts=ACCOUNTS,
        recentHistory=[],
        locale="vi-VN",
        timezone="Asia/Ho_Chi_Minh",
        source="text",
    )


def _mask_volatile(messages: list[dict]) -> list[dict]:
    """Replace CURRENT_TIME=... timestamp with a placeholder for stable diff."""
    out = []
    for msg in messages:
        m = dict(msg)
        if isinstance(m.get("content"), str):
            m["content"] = re.sub(
                r"CURRENT_TIME=[^;]+;",
                "CURRENT_TIME=<MASKED>;",
                m["content"],
            )
        out.append(m)
    return out


@pytest.mark.parametrize("case", PREFILL_CASES, ids=[c["name"] for c in PREFILL_CASES])
async def test_prefill_baseline(case, deepseek_mock, captured_requests):
    """Snapshot request body + final response for each prefill case."""
    from app.services.prefill_service import TransactionPrefillService

    service = TransactionPrefillService()
    response_payload = make_json_response(case["llm_output"])

    with deepseek_mock(responses=[response_payload]):
        result = await service.prefill(_build_request(case["raw_text"]))

    # Capture LLM request body (the prompts sent to DeepSeek).
    assert len(captured_requests) >= 1, "Expected at least 1 DeepSeek call"
    llm_request = captured_requests[0]["body"]

    # Snapshot semantically — token cap can be either max_tokens or max_completion_tokens (OpenAI rename).
    max_tok = llm_request.get("max_tokens") or llm_request.get("max_completion_tokens")
    snapshot = {
        "llm_request": {
            "model": llm_request.get("model"),
            "temperature": llm_request.get("temperature"),
            "max_tokens_eff": max_tok,
            "response_format": llm_request.get("response_format"),
            "messages": _mask_volatile(llm_request.get("messages", [])),
        },
        "service_response": result.model_dump(mode="json"),
    }

    name = f"prefill_{case['name']}"
    existing = load_baseline(name)
    if existing is None:
        save_baseline(name, snapshot)
        pytest.skip(f"Baseline created: {name}.json")
    else:
        # Diff: messages must be identical (system + user prompt).
        baseline_msgs = existing["llm_request"]["messages"]
        current_msgs = snapshot["llm_request"]["messages"]
        assert baseline_msgs == current_msgs, (
            "Messages drift detected between baseline and current. "
            "If migration is intentional, delete the baseline file and re-run."
        )
        # Response shape stays equivalent.
        assert existing["service_response"] == snapshot["service_response"], (
            "Service response drift detected"
        )
