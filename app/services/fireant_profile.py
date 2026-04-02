"""
FireAnt REST v2 — company profile (official API, not HTML scraping).

The web UI at https://fireant.vn/charts/content/symbols/FPT loads the same data
via authenticated APIs. Public OpenAPI spec:
  https://restv2.fireant.vn/swagger/docs/v1

Endpoint used: GET /symbols/{symbol}/profile
  → model CompanyInfo (companyName, businessAreas, overview, ...)

Auth: OAuth2 Bearer token with scope ``symbols-read``.
Set env ``FIREANT_ACCESS_TOKEN`` (alias ``FIREANT_BEARER_TOKEN``).

Legal: use only with a token you obtained under FireAnt's terms; do not bypass
their access controls.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.services.icb_normalization import normalize_icb_code

logger = logging.getLogger(__name__)

# Match Java Company.description length where possible
_MAX_DESCRIPTION_LEN = 2000
# Backend Industry.label (single source for sector text)
_MAX_INDUSTRY_LABEL_LEN = 2000


def _empty_meta() -> Dict[str, Optional[str]]:
    return {
        "companyName": None,
        "industryLabelFull": None,
        "description": None,
        "icbCode": None,
    }


def _truncate(s: str, max_len: int, ellipsis: str = "…") -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + ellipsis


def _strip_html(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if "<" in s and ">" in s:
        try:
            from bs4 import BeautifulSoup

            s = BeautifulSoup(s, "html.parser").get_text(separator="\n", strip=True)
        except Exception:
            pass
    s = s.strip()
    return s if s else None


def _business_areas_plain(raw: Optional[Any]) -> Optional[str]:
    """
    FireAnt ``businessAreas`` is often HTML (<ul><li>…</li>).
    Returns a single plain-text line (unbounded); caller truncates for DB columns.
    """
    plain = _strip_html(raw)
    if not plain:
        return None
    lines = [ln.strip().rstrip(".").strip() for ln in plain.splitlines() if ln.strip()]
    seen: set[str] = set()
    uniq: List[str] = []
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(ln)
    joined = "; ".join(uniq)
    return joined if joined else None


def fetch_fireant_company_meta(symbol: str) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    Returns companyName, industryLabelFull (for ``industries.label``), description, icbCode.
    If no token is configured, returns all-None dict and no warnings.
    """
    token = (
        os.getenv("FIREANT_ACCESS_TOKEN") or os.getenv("FIREANT_BEARER_TOKEN") or ""
    ).strip()
    if not token:
        return _empty_meta(), []

    base = (os.getenv("FIREANT_API_BASE") or "https://restv2.fireant.vn").rstrip("/")
    url = f"{base}/symbols/{symbol.upper()}/profile"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "FinFlow-data-crawler/1.0",
    }
    warnings: List[str] = []
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 401:
            warnings.append(
                "FireAnt: 401 Unauthorized — kiểm tra FIREANT_ACCESS_TOKEN (scope symbols-read)."
            )
            return _empty_meta(), warnings
        if r.status_code == 403:
            warnings.append(
                "FireAnt: 403 Forbidden — token thiếu quyền symbols-read hoặc tài khoản bị chặn."
            )
            return _empty_meta(), warnings
        r.raise_for_status()
        data: Any = r.json()
    except requests.RequestException as e:
        warnings.append(f"FireAnt: request failed: {e}")
        return _empty_meta(), warnings
    except ValueError as e:
        warnings.append(f"FireAnt: invalid JSON: {e}")
        return _empty_meta(), warnings

    if not isinstance(data, dict):
        warnings.append("FireAnt: unexpected response (not a JSON object)")
        return _empty_meta(), warnings

    name = (data.get("companyName") or data.get("shortName") or "").strip() or None
    areas_plain = _business_areas_plain(data.get("businessAreas"))
    # Swagger có thể dùng camelCase khác nhau giữa các bản build
    icb_raw = (
        data.get("icbCode")
        or data.get("icb_code")
        or data.get("industryIcbCode")
        or data.get("industryCode")
    )
    icb_code = normalize_icb_code(icb_raw)

    industry_label_full: Optional[str] = None
    if areas_plain:
        industry_label_full = _truncate(areas_plain, _MAX_INDUSTRY_LABEL_LEN)
    elif icb_code:
        industry_label_full = f"ICB {icb_code}"[:_MAX_INDUSTRY_LABEL_LEN]

    desc = _strip_html(data.get("overview"))
    if desc and len(desc) > _MAX_DESCRIPTION_LEN:
        desc = desc[: _MAX_DESCRIPTION_LEN - 3] + "..."

    if not any([name, industry_label_full, desc, icb_code]):
        logger.debug("[%s] FireAnt profile returned empty name/industry/overview", symbol)

    out = _empty_meta()
    out.update(
        {
            "companyName": name,
            "industryLabelFull": industry_label_full,
            "description": desc,
            "icbCode": icb_code,
        }
    )
    return out, warnings
