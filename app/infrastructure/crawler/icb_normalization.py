"""Chuẩn hoá mã ICB từ API (số / float / chuỗi) → chuỗi khớp ``industry_nodes.icb_code``."""

from __future__ import annotations

import math
from typing import Any


def normalize_icb_code(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val):
            return None
        if val == int(val):
            return str(int(val))
        return str(val).strip()
    s = str(val).strip()
    if not s:
        return None
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s
