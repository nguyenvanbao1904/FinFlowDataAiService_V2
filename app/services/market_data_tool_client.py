from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.http_client import get_http_client
from app.services.chat.utils.math_helpers import as_float, safe_mean, safe_median
from app.services.forecast_tool_service import ForecastToolService

logger = logging.getLogger(__name__)


class MarketDataToolClient:
    def __init__(self) -> None:
        self.base_url = (settings.JAVA_BACKEND_URL or "http://localhost:8080/api/internal").rstrip("/")
        self.internal_api_key = (settings.INTERNAL_API_KEY or "").strip()
        self.timeout_seconds = max(5, int(settings.CHAT_TOOL_TIMEOUT_SECONDS))
        self.forecast_tool_service = ForecastToolService()
        self.debug_log_prompts = bool(settings.CHAT_DEBUG_LOG_PROMPTS)
        self.debug_log_max_chars = max(500, int(settings.CHAT_DEBUG_LOG_MAX_CHARS))

    async def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tool_calls:
            return []
        tasks = [self.execute_tool_call(call.get("name", ""), call.get("arguments") or {}) for call in tool_calls]
        return await asyncio.gather(*tasks)

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if (tool_name or "").strip() == "get_company_forecast":
            symbol = arguments.get("symbol")
            target_year = arguments.get("targetYear")
            result = await asyncio.to_thread(
                self.forecast_tool_service.get_company_forecast,
                str(symbol or ""),
                int(target_year) if isinstance(target_year, int) else None,
            )
            result["name"] = "get_company_forecast"
            self._debug_log_tool(
                "get_company_forecast",
                arguments,
                result,
            )
            return result

        try:
            method, path, params = self._build_request(tool_name, arguments)
        except Exception as exc:
            result = {
                "name": tool_name,
                "ok": False,
                "data": None,
                "error_code": "INVALID_TOOL_ARGS",
                "error_message": str(exc),
                "source_refs": [],
            }
            self._debug_log_tool(str(tool_name), arguments, result)
            return result

        headers: dict[str, str] = {}
        if self.internal_api_key:
            headers["X-Internal-Api-Key"] = self.internal_api_key

        url = f"{self.base_url}{path}"
        timeout = httpx.Timeout(self.timeout_seconds)

        try:
            client = get_http_client()
            response = await client.request(
                method=method, url=url, params=params, headers=headers, timeout=timeout,
            )

            if response.status_code < 200 or response.status_code >= 300:
                result = {
                    "name": tool_name,
                    "ok": False,
                    "data": None,
                    "error_code": f"HTTP_{response.status_code}",
                    "error_message": response.text[:500],
                    "source_refs": [],
                }
                self._debug_log_tool(str(tool_name), arguments, result)
                return result

            try:
                payload = response.json()
            except Exception:
                payload = {"raw": response.text}

            if (tool_name or "").strip() == "get_company_live_valuation_snapshot":
                payload = self._extract_live_valuation_snapshot(
                    symbol=str(arguments.get("symbol") or "").strip().upper(),
                    payload=payload,
                )

            if (tool_name or "").strip() == "get_company_metrics":
                payload = self._extract_company_metrics(payload)

            # Always summarize daily valuations to save tokens — no need for flag
            if (tool_name or "").strip() == "get_company_daily_valuations":
                payload = self._summarize_daily_valuations(payload)

            result = {
                "name": tool_name,
                "ok": True,
                "data": payload,
                "error_code": None,
                "error_message": None,
                "source_refs": [],
            }
            self._debug_log_tool(str(tool_name), arguments, result)
            return result
        except Exception as exc:
            result = {
                "name": tool_name,
                "ok": False,
                "data": None,
                "error_code": "TOOL_UPSTREAM_ERROR",
                "error_message": f"{type(exc).__name__}: {exc}",
                "source_refs": [],
            }
            self._debug_log_tool(str(tool_name), arguments, result)
            return result

    def _build_request(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        name = (tool_name or "").strip()

        if name == "get_company_market_data":
            symbol = self._required_str(arguments, "symbol")
            params = self._optional_common_series_params(arguments)
            include = arguments.get("include")
            if isinstance(include, list) and include:
                params["include"] = [str(v) for v in include]
            elif isinstance(include, str) and include.strip():
                params["include"] = [chunk.strip() for chunk in include.split(",") if chunk.strip()]
            return "GET", f"/investment/query/companies/{symbol}/market-data", params

        if name == "get_industry_nodes":
            return "GET", "/investment/query/industries/nodes", {}

        if name == "suggest_companies":
            query = self._required_str(arguments, "q")
            params = {"q": query}
            limit = arguments.get("limit")
            if limit is not None:
                params["limit"] = int(limit)
            return "GET", "/investment/query/companies/suggest", params

        if name == "get_company_industries":
            symbols = arguments.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("symbols must be a non-empty array")
            return "GET", "/investment/query/companies/industries", {"symbols": [str(s) for s in symbols]}

        if name == "get_company_analysis":
            symbol = self._required_str(arguments, "symbol")
            return "GET", f"/investment/query/companies/{symbol}/analysis", self._optional_common_series_params(arguments)

        if name == "get_company_metrics":
            # Alias for get_company_analysis but stripped down
            symbol = self._required_str(arguments, "symbol")
            return "GET", f"/investment/query/companies/{symbol}/analysis", self._optional_common_series_params(arguments)

        if name == "get_company_live_valuation_snapshot":
            symbol = self._required_str(arguments, "symbol")
            # Keep payload small while still allowing overview/live valuation calculation.
            return "GET", f"/investment/query/companies/{symbol}/analysis", {"annualLimit": 1, "quarterlyLimit": 1}

        if name == "get_company_financial_series":
            symbol = self._required_str(arguments, "symbol")
            params = self._optional_common_series_params(arguments)
            # Default to annualLimit=3 and quarterlyLimit=0 to prevent pulling data back to 2013 if forgot.
            if "annualLimit" not in params:
                params["annualLimit"] = 3
            if "quarterlyLimit" not in params:
                params["quarterlyLimit"] = 0
            return "GET", f"/investment/query/companies/{symbol}/analysis/financials", params

        if name == "get_company_valuations":
            symbol = self._required_str(arguments, "symbol")
            params: dict[str, Any] = {}
            annual_limit = arguments.get("annualLimit")
            if annual_limit is not None:
                params["annualLimit"] = int(annual_limit)
            for key in ("startDate", "endDate"):
                val = arguments.get(key)
                if isinstance(val, str) and val.strip():
                    params[key] = val.strip()
            show_quarterly = arguments.get("showQuarterly")
            if isinstance(show_quarterly, bool):
                params["showQuarterly"] = show_quarterly
            return "GET", f"/investment/query/companies/{symbol}/analysis/valuations", params

        if name == "get_company_daily_valuations":
            symbol = self._required_str(arguments, "symbol")
            start_date = self._required_str(arguments, "startDate")
            end_date = self._required_str(arguments, "endDate")
            return "GET", f"/investment/query/companies/{symbol}/analysis/valuations/daily", {
                "startDate": start_date,
                "endDate": end_date,
            }

        if name == "get_company_dividends":
            symbol = self._required_str(arguments, "symbol")
            params: dict[str, Any] = {}
            annual_limit = arguments.get("annualLimit")
            if annual_limit is not None:
                params["annualLimit"] = int(annual_limit)
            return "GET", f"/investment/query/companies/{symbol}/analysis/dividends", params

        raise ValueError(f"unsupported tool: {name}")

    @staticmethod
    def _required_str(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} is required")
        return value.strip()

    @staticmethod
    def _optional_common_series_params(arguments: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        annual_limit = arguments.get("annualLimit")
        quarterly_limit = arguments.get("quarterlyLimit")
        if annual_limit is not None:
            params["annualLimit"] = int(annual_limit)
        if quarterly_limit is not None:
            params["quarterlyLimit"] = int(quarterly_limit)
        return params

    def _debug_log_tool(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        if not self.debug_log_prompts:
            return
        logger.warning(
            "[CHAT][TOOL][%s] args=%s result=%s",
            name,
            self._truncate(json.dumps(args, ensure_ascii=False)),
            self._truncate(json.dumps(result, ensure_ascii=False)),
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.debug_log_max_chars:
            return text
        return text[: self.debug_log_max_chars] + "...[truncated]"

    @classmethod
    def _extract_live_valuation_snapshot(cls, *, symbol: str, payload: Any) -> dict[str, Any]:
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
            "livePe": live_pe,
            "livePb": live_pb,
            "livePs": live_ps,
            "currentPe": as_float(overview.get("currentPE")),
            "currentPb": as_float(overview.get("currentPB")),
            "currentPs": as_float(overview.get("currentPS")),
            "medianPe": median_pe,
            "medianPb": median_pb,
            "medianPs": median_ps,
            "peView": cls._relative_valuation_view(live_pe, median_pe),
            "pbView": cls._relative_valuation_view(live_pb, median_pb),
            "psView": cls._relative_valuation_view(live_ps, median_ps),
        }

    @staticmethod
    def _relative_valuation_view(live_value: float | None, median_value: float | None) -> str | None:
        if live_value is None or median_value is None or median_value == 0:
            return None
        delta = (live_value - median_value) / abs(median_value)
        if delta <= -0.15:
            return "rẻ tương đối"
        if delta <= -0.05:
            return "hơi rẻ"
        if delta >= 0.15:
            return "đắt tương đối"
        if delta >= 0.05:
            return "hơi đắt"
        return "gần trung vị lịch sử"

    @staticmethod
    def _extract_company_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        """Strip down the analysis response to save tokens for Synthesizer."""
        if not isinstance(payload, dict):
            return payload
        return {"overview": payload.get("overview")}

    @staticmethod
    def _summarize_daily_valuations(payload: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert large daily valuations array into a smart summary to save tokens."""
        if not isinstance(payload, list):
            return {"raw_payload": payload}

        pe_list: list[float] = []
        pb_list: list[float] = []
        ps_list: list[float] = []

        for item in payload:
            if not isinstance(item, dict):
                continue
            v_pe = as_float(item.get("pe"))
            if v_pe is not None:
                pe_list.append(v_pe)
            v_pb = as_float(item.get("pb"))
            if v_pb is not None:
                pb_list.append(v_pb)
            v_ps = as_float(item.get("ps"))
            if v_ps is not None:
                ps_list.append(v_ps)

        return {
            "summary": {
                "pe_median": safe_median(pe_list),
                "pe_mean": safe_mean(pe_list),
                "pb_median": safe_median(pb_list),
                "pb_mean": safe_mean(pb_list),
                "ps_median": safe_median(ps_list),
                "ps_mean": safe_mean(ps_list),
                "data_points_count": len(pe_list),
            }
        }

