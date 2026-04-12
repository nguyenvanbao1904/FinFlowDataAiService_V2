"""Math helpers for chat — safe statistics, formatting, valuation status."""
from __future__ import annotations

from typing import Any


def as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_mean(values: list[float]) -> float | None:
    xs = [float(v) for v in values if isinstance(v, (int, float))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def safe_median(values: list[float]) -> float | None:
    xs = sorted(float(v) for v in values if isinstance(v, (int, float)))
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def pct_diff(current: float | None, ref: float | None) -> float | None:
    if current is None or ref is None or ref == 0:
        return None
    return (current - ref) / ref * 100.0


def valuation_status_label(diff_pct: float | None, *, threshold_pct: float = 3.0) -> str | None:
    if diff_pct is None:
        return None
    if diff_pct > threshold_pct:
        return "OVERVALUED"
    if diff_pct < -threshold_pct:
        return "UNDERVALUED"
    return "FAIR"


def to_ty_dong(value: float | None) -> float | None:
    """Convert raw VND value (from DB) to tỷ đồng."""
    if value is None:
        return None
    return value / 1_000_000_000.0


def to_percent(value: float | None) -> float | None:
    """Convert ratio (0.15) to percentage (15.0)."""
    if value is None:
        return None
    return value * 100.0


def fmt_ty(value: float | None, *, already_ty: bool = False) -> str:
    """Format VND value as 'X tỷ đồng' string."""
    if value is None:
        return "N/A"
    ty = value if already_ty else value / 1_000_000_000.0
    return f"{ty:,.1f} tỷ đồng"


def normalize_cplh_to_billion(value: float | None) -> float | None:
    """Normalize cổ phiếu lưu hành to tỷ cổ phiếu (handle various unit scales)."""
    if value is None:
        return None
    v = abs(float(value))
    if v >= 1_000_000:
        return float(value) / 1_000_000_000.0
    if v >= 1_000:
        return float(value) / 1_000.0
    return float(value)


def build_metric_status(
    current: float | None,
    ref_mean: float | None,
    ref_median: float | None,
) -> dict[str, Any]:
    """Build a single metric comparison dict (PE, PB, or PS)."""
    diff_mean = pct_diff(current, ref_mean)
    diff_median = pct_diff(current, ref_median)
    return {
        "current": current,
        "ref_mean": ref_mean,
        "ref_median": ref_median,
        "diff_vs_mean_pct": diff_mean,
        "diff_vs_median_pct": diff_median,
        "status_vs_mean": valuation_status_label(diff_mean),
        "status_vs_median": valuation_status_label(diff_median),
    }
