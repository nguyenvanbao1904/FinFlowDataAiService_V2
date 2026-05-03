"""Capture baseline for ChatOrchestrator against mocked DeepSeek + backend."""
from __future__ import annotations

import json
import re

import pytest

from app.models.chat import (
    ChatOrchestrateRequest,
    ChatTurnMessage,
)
from tests.conftest import (
    load_baseline,
    make_text_response,
    make_tool_call_response,
    save_baseline,
)


def _mask_volatile(messages: list[dict]) -> list[dict]:
    """Replace timestamps + normalize tool-call args formatting for stable diff."""
    out = []
    for msg in messages:
        m = dict(msg)
        c = m.get("content")
        if isinstance(c, str):
            c = re.sub(
                r"Thời gian hiện tại: [^\n]+",
                "Thời gian hiện tại: <MASKED>",
                c,
            )
            # Tool result content may contain JSON object — normalize spacing for diff.
            if c.startswith("{") or c.startswith("["):
                try:
                    c = json.dumps(json.loads(c), ensure_ascii=False, sort_keys=True)
                except (json.JSONDecodeError, TypeError):
                    pass
            m["content"] = c
        # Normalize tool_calls.arguments JSON string for stable diff.
        if "tool_calls" in m and isinstance(m["tool_calls"], list):
            new_calls = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                func = dict(tc_copy.get("function", {}))
                args_raw = func.get("arguments", "")
                if isinstance(args_raw, str) and args_raw:
                    try:
                        # Strip args injected by orchestrator that pydantic-ai
                        # now injects via deps (user_id stays out of LLM schema).
                        parsed = json.loads(args_raw)
                        if isinstance(parsed, dict):
                            parsed.pop("user_id", None)
                        func["arguments"] = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
                    except (json.JSONDecodeError, TypeError):
                        pass
                # Drop the model-generated tool call id — pydantic-ai uses its own.
                tc_copy.pop("id", None)
                tc_copy["function"] = func
                new_calls.append(tc_copy)
            m["tool_calls"] = new_calls
        m.pop("tool_call_id", None)
        out.append(m)
    return out


def _summarize_request(body: dict) -> dict:
    max_tok = body.get("max_tokens") or body.get("max_completion_tokens")
    return {
        "model": body.get("model"),
        "temperature": body.get("temperature"),
        "max_tokens_eff": max_tok,
        "tools_count": len(body.get("tools") or []),
        "tool_names": sorted(
            t["function"]["name"] for t in (body.get("tools") or [])
        ),
        "messages": _mask_volatile(body.get("messages") or []),
        "has_response_format": "response_format" in body,
    }


CASES = [
    {
        "name": "direct_greeting",
        "user_message": "Xin chào",
        "deepseek_responses": [
            make_text_response("Xin chào! Tôi là trợ lý CFO. Bạn cần phân tích gì?"),
        ],
        "backend_routes": {},
    },
    {
        "name": "personal_finance_report",
        "user_message": "Tình hình thu chi của tôi tháng này thế nào?",
        "deepseek_responses": [
            make_tool_call_response([
                {"id": "call-1", "name": "get_personal_finance_report", "arguments": {"user_id": "u-test"}},
            ]),
            make_text_response("Tháng này bạn thu 20tr, chi 14tr, dòng tiền dương 6tr."),
        ],
        "backend_routes": {
            r".*/transaction/finance-report.*": {
                "summary": "income=20m, expense=14m",
                "totals": {"income": 20000000, "expense": 14000000},
            },
        },
    },
    {
        "name": "compute_fair_value_hpg",
        "user_message": "HPG có rẻ không?",
        "deepseek_responses": [
            make_tool_call_response([
                {"id": "call-fv", "name": "compute_fair_value", "arguments": {"symbol": "HPG"}},
            ]),
            make_text_response("HPG đang rẻ tương đối với target ~30,000đ."),
        ],
        "backend_routes": {
            r".*/companies/HPG/analysis$": {
                "overview": {
                    "eps": 1500, "bvps": 20000, "roe": 0.15, "cplh": 6000000000,
                    "industryIcbCode": "1750", "industryLabel": "Steel",
                    "medianPE": 8, "medianPB": 1.2, "medianPS": 0.8,
                    "companyName": "Hoa Phat Group",
                },
            },
            r".*/companies/HPG/analysis/financials.*": {
                "nonBank": [
                    {"year": 2023, "quarter": 0, "quarterCount": 4, "profitAfterTax": 6000000000000},
                    {"year": 2024, "quarter": 0, "quarterCount": 4, "profitAfterTax": 8000000000000},
                ],
            },
            r".*/companies/HPG/analysis/valuations/daily.*": [
                {"pe": 8.5, "pb": 1.3, "ps": 0.9},
                {"pe": 7.5, "pb": 1.1, "ps": 0.7},
            ],
        },
    },
]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
async def test_orchestrator_baseline(case, deepseek_mock, backend_mock, captured_requests):
    from app.services.chat.orchestrator import ChatOrchestrator

    orchestrator = ChatOrchestrator()
    request = ChatOrchestrateRequest(
        thread_id="thread-1",
        user_id="u-test",
        user_message=case["user_message"],
        context_summary="",
        last_messages=[],
    )

    with backend_mock(routes=case["backend_routes"]):
        with deepseek_mock(responses=list(case["deepseek_responses"])):
            result = await orchestrator.orchestrate(request)

    # All DeepSeek requests captured (each agent turn is one).
    ds_requests = [c for c in captured_requests if c["target"] == "deepseek"]
    be_requests = [c for c in captured_requests if c["target"] == "backend"]

    snapshot = {
        "deepseek_turns": [_summarize_request(r["body"]) for r in ds_requests],
        "backend_calls": [
            {"method": r["method"], "url_path": r["url"].split("?")[0].split("/api/internal")[-1]}
            for r in be_requests
        ],
        "service_response": {
            "assistant_message": result.assistant_message,
            "tool_calls": [
                {"name": tc["name"], "args_keys": sorted(tc.get("arguments", {}).keys())}
                for tc in result.tool_calls
            ],
            "tool_results_ok": [r.get("ok") for r in result.tool_results],
            "needs_clarification": result.needs_clarification,
        },
    }

    name = f"orchestrator_{case['name']}"
    existing = load_baseline(name)
    if existing is None:
        save_baseline(name, snapshot)
        pytest.skip(f"Baseline created: {name}.json")

    assert existing["deepseek_turns"] == snapshot["deepseek_turns"], (
        f"DeepSeek request drift in case {case['name']}"
    )
    assert existing["service_response"] == snapshot["service_response"]
    # Backend calls — order may not be deterministic for parallel calls; compare as sets.
    assert sorted(json.dumps(c) for c in existing["backend_calls"]) == sorted(
        json.dumps(c) for c in snapshot["backend_calls"]
    )
