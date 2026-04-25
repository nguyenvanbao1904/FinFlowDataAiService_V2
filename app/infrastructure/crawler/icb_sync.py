"""
Xây payload đồng bộ cây ngành ICB → Java ``POST .../industry-nodes``.

Source: FireAnt ``GET /industries`` (4-digit codes, matches ``profile.icbCode``).

Suy luận cha (khi mã là 4-digit):
- level 2: ABCD -> A000
- level 3: ABCD -> AB00
- level 4: ABCD -> ABC0
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

# Cố định để UUID trùng giữa các lần chạy / máy
_ICB_NODE_NS = uuid.uuid5(uuid.NAMESPACE_URL, "https://finflow.local/industry-node/icb-code")


def icb_node_uuid(icb_code: str) -> str:
    return str(uuid.uuid5(_ICB_NODE_NS, icb_code.strip()))


def _normalize_api_item(obj: dict[str, Any]) -> tuple[str, int, str] | None:
    """
    Chuẩn hoá item từ FireAnt ``/industries`` hoặc fallback.
    Trả về: (industryCode/icbCode, level, name).
    """
    code = (
        obj.get("industryCode")
        or obj.get("icbCode")
        or obj.get("icb_code")
        or obj.get("code")
    )
    name = obj.get("name") or obj.get("icbName") or obj.get("icb_name")
    level = obj.get("level")
    if code is None or name is None or level is None:
        return None
    c = str(code).strip()
    n = str(name).strip()
    try:
        lv = int(level)
    except (TypeError, ValueError):
        return None
    if not c or not n:
        return None
    return c, lv, n


def fetch_icb_rows_from_fireant() -> list[tuple[str, int, str]]:
    """
    FireAnt: dùng ``GET /industries`` để lấy mã 4-digit khớp với ``profile.icbCode``.
    """
    token = settings.FIREANT_ACCESS_TOKEN.strip()
    if not token:
        return []
    base = settings.FIREANT_API_BASE.rstrip("/")
    url = f"{base}/industries"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=45,
        )
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning("FireAnt GET /industries failed: %s", e)
        return []

    raw_list: Any
    if isinstance(body, list):
        raw_list = body
    elif isinstance(body, dict):
        raw_list = body.get("data") or body.get("items") or body.get("result") or []
    else:
        raw_list = []

    out: list[tuple[str, int, str]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        row = _normalize_api_item(item)
        if row:
            out.append(row)
    return out


def _longest_prefix_parent(
    child_code: str, child_level: int, nodes: list[dict[str, Any]]
) -> str | None:
    best: str | None = None
    best_len = -1
    for n in nodes:
        if n["level"] >= child_level:
            continue
        c = n["code"]
        if child_code.startswith(c) and len(c) < len(child_code) and len(c) > best_len:
            best_len = len(c)
            best = c
    return best


def _infer_parent_code_by_level(code: str, level: int) -> str | None:
    """
    FireAnt ``/industries``: mã luôn là 4-digit theo format level.
    """
    if level <= 1:
        return None
    c = str(code).strip()
    if len(c) != 4:
        return None
    if level == 2:
        return c[0] + "000"
    if level == 3:
        return c[:2] + "00"
    if level == 4:
        return c[:3] + "0"
    return None


def build_industry_node_payloads() -> list[dict[str, Any]]:
    rows = fetch_icb_rows_from_fireant()
    if not rows:
        logger.error("FireAnt GET /industries returned empty — skip industry-nodes sync")
        return []

    # dedupe by code (keep first)
    seen: set[str] = set()
    uniq: list[tuple[str, int, str]] = []
    for code, lv, name in sorted(rows, key=lambda x: (x[1], x[0])):
        if code in seen:
            continue
        seen.add(code)
        uniq.append((code, lv, name))

    nodes: list[dict[str, Any]] = [
        {"code": c, "level": lv, "nameVi": n} for c, lv, n in uniq
    ]

    payloads: list[dict[str, Any]] = []
    code_set = {n["code"] for n in nodes}
    for n in nodes:
        parent_code = _infer_parent_code_by_level(n["code"], n["level"])
        if parent_code is None:
            parent_code = _longest_prefix_parent(n["code"], n["level"], nodes)
        # Nếu suy luận ra parentCode nhưng parent không tồn tại trong payload
        # (cây FireAnt có nhánh không đầy đủ), thì bỏ relation để tránh lỗi FK.
        parent_id = icb_node_uuid(parent_code) if parent_code and parent_code in code_set else None
        payloads.append(
            {
                "id": icb_node_uuid(n["code"]),
                "parentId": parent_id,
                "nameVi": n["nameVi"],
                "level": n["level"],
                "icbCode": n["code"],
                "detailLabel": None,
            }
        )

    logger.info("Built %d industry-nodes from FireAnt", len(payloads))
    return payloads
