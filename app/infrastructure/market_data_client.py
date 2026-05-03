"""HTTP wrapper around the Spring Boot internal investment APIs.

Each tool name maps to a (method, path-template, param-builder) triple,
post-processing for token-heavy payloads is centralised, and the result
is wrapped in the {ok, data, error_*} envelope the agent expects.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import httpx

from app.core.config import settings
from app.core.http_client import get_http_client
from app.services.chat.utils.math_helpers import as_float, safe_mean, safe_median

logger = logging.getLogger(__name__)


# ── Param builders ────────────────────────────────────────────────────


def _required_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _series_params(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (v := args.get("annualLimit")) is not None:
        out["annualLimit"] = int(v)
    if (v := args.get("quarterlyLimit")) is not None:
        out["quarterlyLimit"] = int(v)
    return out


def _market_data_params(args: dict[str, Any]) -> dict[str, Any]:
    params = _series_params(args)
    include = args.get("include")
    if isinstance(include, list) and include:
        params["include"] = [str(v) for v in include]
    elif isinstance(include, str) and include.strip():
        params["include"] = [c.strip() for c in include.split(",") if c.strip()]
    return params


def _financial_series_params(args: dict[str, Any]) -> dict[str, Any]:
    params = _series_params(args)
    params.setdefault("annualLimit", 3)
    params.setdefault("quarterlyLimit", 0)
    return params


def _cash_flows_params(args: dict[str, Any]) -> dict[str, Any]:
    params = _series_params(args)
    params.setdefault("annualLimit", 5)
    params.setdefault("quarterlyLimit", 0)
    return params


def _valuations_params(args: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if (v := args.get("annualLimit")) is not None:
        params["annualLimit"] = int(v)
    for key in ("startDate", "endDate"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            params[key] = val.strip()
    if isinstance(args.get("showQuarterly"), bool):
        params["showQuarterly"] = args["showQuarterly"]
    return params


def _daily_valuations_params(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "startDate": _required_str(args, "startDate"),
        "endDate": _required_str(args, "endDate"),
    }


def _dividends_params(args: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if (v := args.get("annualLimit")) is not None:
        params["annualLimit"] = int(v)
    return params


def _suggest_params(args: dict[str, Any]) -> dict[str, Any]:
    params = {"q": _required_str(args, "q")}
    if (limit := args.get("limit")) is not None:
        params["limit"] = int(limit)
    return params


def _industries_params(args: dict[str, Any]) -> dict[str, Any]:
    symbols = args.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols must be a non-empty array")
    return {"symbols": [str(s) for s in symbols]}


def _live_snapshot_params(_args: dict[str, Any]) -> dict[str, Any]:
    # Keep payload small while still allowing the overview to be computed.
    return {"annualLimit": 1, "quarterlyLimit": 1}


# ── Route table ───────────────────────────────────────────────────────

# (path-template, param-builder, requires_symbol)
_ROUTES: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, Any]], bool]] = {
    "get_company_market_data":             ("/investment/query/companies/{symbol}/market-data", _market_data_params, True),
    "get_industry_nodes":                  ("/investment/query/industries/nodes", lambda _: {}, False),
    "suggest_companies":                   ("/investment/query/companies/suggest", _suggest_params, False),
    "get_company_industries":              ("/investment/query/companies/industries", _industries_params, False),
    "get_company_analysis":                ("/investment/query/companies/{symbol}/analysis", _series_params, True),
    "get_company_metrics":                 ("/investment/query/companies/{symbol}/analysis", _series_params, True),
    "get_company_live_valuation_snapshot": ("/investment/query/companies/{symbol}/analysis", _live_snapshot_params, True),
    "get_company_financial_series":        ("/investment/query/companies/{symbol}/analysis/financials", _financial_series_params, True),
    "get_company_cash_flows":              ("/investment/query/companies/{symbol}/analysis/financials", _cash_flows_params, True),
    "get_company_valuations":              ("/investment/query/companies/{symbol}/analysis/valuations", _valuations_params, True),
    "get_company_daily_valuations":        ("/investment/query/companies/{symbol}/analysis/valuations/daily", _daily_valuations_params, True),
    "get_company_dividends":               ("/investment/query/companies/{symbol}/analysis/dividends", _dividends_params, True),
}


# ── Post-processors (executed after a successful HTTP response) ──────


def _summarize_daily_valuations(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list):
        return {"raw_payload": payload}
    pe_list, pb_list, ps_list = [], [], []
    for item in payload:
        if not isinstance(item, dict):
            continue
        for src, dst in (("pe", pe_list), ("pb", pb_list), ("ps", ps_list)):
            v = as_float(item.get(src))
            if v is not None:
                dst.append(v)
    return {
        "summary": {
            "pe_median": safe_median(pe_list), "pe_mean": safe_mean(pe_list),
            "pb_median": safe_median(pb_list), "pb_mean": safe_mean(pb_list),
            "ps_median": safe_median(ps_list), "ps_mean": safe_mean(ps_list),
            "data_points_count": len(pe_list),
        }
    }


def _extract_live_valuation_snapshot(symbol: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"symbol": symbol}
    overview = payload.get("overview") if isinstance(payload.get("overview"), dict) else {}

    live_pe = as_float(overview.get("livePe"))
    live_pb = as_float(overview.get("livePb"))
    live_ps = as_float(overview.get("livePs"))
    median_pe = as_float(overview.get("medianPE"))
    median_pb = as_float(overview.get("medianPB"))
    median_ps = as_float(overview.get("medianPS"))

    return {
        "symbol": symbol or str(overview.get("symbol") or "").strip().upper() or None,
        "livePriceVnd": as_float(overview.get("livePriceVnd")),
        "livePriceSource": overview.get("livePriceSource"),
        "livePe": live_pe, "livePb": live_pb, "livePs": live_ps,
        "currentPe": as_float(overview.get("currentPE")),
        "currentPb": as_float(overview.get("currentPB")),
        "currentPs": as_float(overview.get("currentPS")),
        "medianPe": median_pe, "medianPb": median_pb, "medianPs": median_ps,
        "peView": _relative_view(live_pe, median_pe),
        "pbView": _relative_view(live_pb, median_pb),
        "psView": _relative_view(live_ps, median_ps),
    }


def _relative_view(live: float | None, median: float | None) -> str | None:
    if live is None or median is None or median == 0:
        return None
    delta = (live - median) / abs(median)
    if delta <= -0.15:
        return "rẻ tương đối"
    if delta <= -0.05:
        return "hơi rẻ"
    if delta >= 0.15:
        return "đắt tương đối"
    if delta >= 0.05:
        return "hơi đắt"
    return "gần trung vị lịch sử"


def _extract_company_metrics(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    return {"overview": payload.get("overview")}


def _extract_cash_flows(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"cashFlows": []}
    cash_flows = payload.get("cashFlows")
    if cash_flows is None and isinstance(payload.get("financialSeries"), dict):
        cash_flows = payload["financialSeries"].get("cashFlows")
    return {"cashFlows": cash_flows if isinstance(cash_flows, list) else []}


_POST_PROCESSORS: dict[str, Callable[[str, Any], Any]] = {
    "get_company_live_valuation_snapshot": _extract_live_valuation_snapshot,
    "get_company_metrics":                 lambda _s, p: _extract_company_metrics(p),
    "get_company_daily_valuations":        lambda _s, p: _summarize_daily_valuations(p),
    "get_company_cash_flows":              lambda _s, p: _extract_cash_flows(p),
}


# ── Client ────────────────────────────────────────────────────────────


class MarketDataToolClient:
    def __init__(self, forecast_service: Any = None) -> None:
        self.base_url = settings.JAVA_BACKEND_URL.rstrip("/")
        self.timeout = httpx.Timeout(max(5, int(settings.CHAT_TOOL_TIMEOUT_SECONDS)))
        self._forecast_service = forecast_service
        self._debug = bool(settings.CHAT_DEBUG_LOG_PROMPTS)
        self._debug_max_chars = max(500, int(settings.CHAT_DEBUG_LOG_MAX_CHARS))

    @property
    def forecast_tool_service(self) -> Any:
        if self._forecast_service is None:
            from app.services.forecast_service import ForecastToolService
            self._forecast_service = ForecastToolService()
        return self._forecast_service

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        name = (tool_name or "").strip()

        if name == "get_company_forecast":
            return await self._run_forecast(arguments)

        try:
            payload = await self._fetch_route(name, arguments)
        except ValueError as exc:
            return self._log(name, arguments, _err(name, "INVALID_TOOL_ARGS", str(exc)))
        except Exception as exc:
            logger.exception("Tool %s upstream error", name)
            return self._log(name, arguments, _err(name, "TOOL_UPSTREAM_ERROR", f"{type(exc).__name__}: {exc}"))

        if isinstance(payload, dict) and "_http_error" in payload:
            return self._log(name, arguments, _err(name, payload["_http_error"], payload.get("_text", "")[:500]))

        post = _POST_PROCESSORS.get(name)
        if post:
            symbol = str(arguments.get("symbol") or "").strip().upper()
            payload = post(symbol, payload)

        return self._log(name, arguments, {
            "name": name, "ok": True, "data": payload,
            "error_code": None, "error_message": None, "source_refs": [],
        })

    async def _run_forecast(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target_year = arguments.get("targetYear")
        result = await asyncio.to_thread(
            self.forecast_tool_service.get_company_forecast,
            str(arguments.get("symbol") or ""),
            int(target_year) if isinstance(target_year, int) else None,
        )
        result["name"] = "get_company_forecast"
        return self._log("get_company_forecast", arguments, result)

    async def _fetch_route(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in _ROUTES:
            raise ValueError(f"unsupported tool: {name}")
        path_template, build_params, requires_symbol = _ROUTES[name]
        params = build_params(arguments)
        if requires_symbol:
            symbol = _required_str(arguments, "symbol")
            path = path_template.format(symbol=symbol)
        else:
            path = path_template

        headers = {}
        if (key := (settings.INTERNAL_API_KEY or "").strip()):
            headers["X-Internal-Api-Key"] = key

        client = get_http_client()
        response = await client.get(
            f"{self.base_url}{path}", params=params, headers=headers, timeout=self.timeout,
        )
        if response.status_code < 200 or response.status_code >= 300:
            return {"_http_error": f"HTTP_{response.status_code}", "_text": response.text}
        try:
            return response.json()
        except Exception:
            logger.warning("Failed to parse JSON from tool %s response", name)
            return {"raw": response.text}

    # ── Debug logging ──

    def _log(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        if self._debug:
            logger.debug(
                "[CHAT][TOOL][%s] args=%s result=%s",
                name,
                _truncate(json.dumps(args, ensure_ascii=False), self._debug_max_chars),
                _truncate(json.dumps(result, ensure_ascii=False, default=str), self._debug_max_chars),
            )
        return result


# ── Module helpers ────────────────────────────────────────────────────


def _err(name: str, code: str, message: str) -> dict[str, Any]:
    return {
        "name": name, "ok": False, "data": None,
        "error_code": code, "error_message": message, "source_refs": [],
    }


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"
