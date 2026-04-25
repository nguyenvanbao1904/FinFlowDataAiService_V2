"""
FireAnt REST v2 — financial data client.

Endpoints:
  GET /symbols/{symbol}/financial-data?type=Q&count=100  → quarterly financials (up to 85 quarters)
  GET /symbols                                            → all listed symbols
  GET /symbols/{symbol}/holders                           → shareholder list

Response shape for financial-data:
  [{symbol, year, quarter, companyType, icbCode, icbName, financialValues: {340 fields…}}]
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        token = settings.FIREANT_ACCESS_TOKEN.strip()
        if not token:
            raise RuntimeError("FIREANT_ACCESS_TOKEN is required for FireAnt crawler")
        _SESSION = requests.Session()
        _SESSION.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "FinFlow-data-crawler/2.0",
            }
        )
    return _SESSION


def _base_url() -> str:
    return settings.FIREANT_API_BASE.rstrip("/")


def fetch_financial_data(
    symbol: str, report_type: str = "Q", count: int = 100
) -> list[dict[str, Any]]:
    """
    Fetch quarterly financial data for a single symbol.
    Returns a list of dicts, each containing top-level fields
    (symbol, year, quarter, companyType, icbCode, icbName)
    plus a nested ``financialValues`` dict with ~340 fields.
    """
    url = f"{_base_url()}/symbols/{symbol.upper()}/financial-data"
    params = {"type": report_type, "count": count}
    try:
        r = _get_session().get(url, params=params, timeout=30)
        if r.status_code == 401:
            logger.error("FireAnt 401 — check FIREANT_ACCESS_TOKEN")
            return []
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("FireAnt financial-data failed for %s: %s", symbol, e)
        return []
    except ValueError as e:
        logger.warning("FireAnt financial-data invalid JSON for %s: %s", symbol, e)
        return []

    if not isinstance(data, list):
        logger.warning("FireAnt financial-data unexpected shape for %s", symbol)
        return []

    return data


def fetch_all_symbols() -> list[dict[str, Any]]:
    """
    Fetch the full list of listed symbols from FireAnt.
    Uses ``/symbols/search?keywords=%25&limit=5000`` which returns all instruments,
    then filters to ``type == "stock"`` and ``isListing == True``.
    Returns list of dicts with at least: {symbol, exchange, type}.
    """
    url = f"{_base_url()}/symbols/search"
    params = {"keywords": "%", "limit": 5000}
    try:
        r = _get_session().get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("FireAnt GET /symbols/search failed: %s", e)
        return []
    except ValueError:
        return []

    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("type") == "stock" and d.get("isListing")]


def fetch_holders(symbol: str) -> list[dict[str, Any]]:
    """Fetch shareholder list for a symbol."""
    url = f"{_base_url()}/symbols/{symbol.upper()}/holders"
    try:
        r = _get_session().get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("FireAnt holders failed for %s: %s", symbol, e)
        return []
    except ValueError:
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data") or data.get("items") or []
    return []


def fetch_dividends(symbol: str) -> list[dict[str, Any]]:
    """
    Fetch yearly dividend summary from ``/symbols/{symbol}/dividends``.
    Returns list of dicts: {year, cashDividend, stockDividend, totalAssets, stockHolderEquity}.
    """
    url = f"{_base_url()}/symbols/{symbol.upper()}/dividends"
    try:
        r = _get_session().get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("FireAnt dividends failed for %s: %s", symbol, e)
        return []
    except ValueError:
        return []

    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]
