"""Chuẩn hoá mã ICB từ API (số / float / chuỗi) → chuỗi khớp ``industry_nodes.icb_code``."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def normalize_icb_code(val: Any) -> Optional[str]:
    if val is None or pd.isna(val):
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val != val:  # NaN
            return None
        if val == int(val):
            return str(int(val))
        s = str(val).strip()
        return s
    s = str(val).strip()
    if not s:
        return None
    # Chuỗi kiểu "50205040.0" hoặc "8,355" (hiếm)
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s
