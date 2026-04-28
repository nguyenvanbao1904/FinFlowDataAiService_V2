"""Deterministic fair-value computation — called as a local tool by the ReAct agent.

Performs the valuation math in Python (not LLM) to ensure:
- Consistent, reproducible results
- Correct handling of edge cases (negative profits, g > CoE, etc.)
- Proper CAGR calculation with guardrails

The LLM gathers data via remote tools, then calls compute_fair_value
to get the final numbers. The LLM only presents the results.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Load playbook ────────────────────────────────────────────────────

_PLAYBOOK_PATH = Path(__file__).parent / "config" / "valuation_playbook.json"

with open(_PLAYBOOK_PATH, encoding="utf-8") as _f:
    PLAYBOOK: dict[str, Any] = json.load(_f)


# ── Industry mapping ─────────────────────────────────────────────────

_ICB_PREFIX_MAP: dict[str, str] = {
    "8355": "BANK",       # Ngân hàng
    "8350": "BANK",
    "8770": "SECURITIES",  # Chứng khoán
    "8771": "SECURITIES",
    "3510": "REAL_ESTATE",
    "3520": "REAL_ESTATE",
    "5370": "RETAIL",     # Bán lẻ
    "5371": "RETAIL",
    "5373": "RETAIL",
    "5550": "RETAIL",
    "9530": "TECHNOLOGY",
    "9533": "TECHNOLOGY",
    "9535": "TECHNOLOGY",
    "9537": "TECHNOLOGY",
    "1750": "MANUFACTURING_HEAVY",
    "1730": "MANUFACTURING_HEAVY",
    "1710": "MANUFACTURING_HEAVY",
    "2350": "MANUFACTURING_HEAVY",
    "3570": "FMCG",
    "3573": "FMCG",
    "3533": "FMCG",
    "6530": "UTILITIES",
    "6570": "UTILITIES",
    "6575": "UTILITIES",
    "2770": "LOGISTICS_PORTS",
    "2771": "LOGISTICS_PORTS",
    "2773": "LOGISTICS_PORTS",
    "4530": "HEALTHCARE",
    "4535": "HEALTHCARE",
}

_LABEL_MAP: dict[str, str] = {
    "ngân hàng": "BANK",
    "chứng khoán": "SECURITIES",
    "bất động sản": "REAL_ESTATE",
    "bán lẻ": "RETAIL",
    "bán lẻ tổng hợp": "RETAIL",
    "công nghệ": "TECHNOLOGY",
    "phần mềm": "TECHNOLOGY",
    "sắt thép": "MANUFACTURING_HEAVY",
    "thép": "MANUFACTURING_HEAVY",
    "vật liệu xây dựng": "MANUFACTURING_HEAVY",
    "hóa chất": "MANUFACTURING_HEAVY",
    "thực phẩm": "FMCG",
    "đồ uống": "FMCG",
    "điện": "UTILITIES",
    "nước": "UTILITIES",
    "điện lực": "UTILITIES",
    "cảng biển": "LOGISTICS_PORTS",
    "vận tải": "LOGISTICS_PORTS",
    "logistics": "LOGISTICS_PORTS",
    "dược phẩm": "HEALTHCARE",
    "y tế": "HEALTHCARE",
}


def resolve_industry(
    icb_code: str | None = None,
    industry_label: str | None = None,
) -> str:
    """Map ICB code or industry label to playbook key."""
    if icb_code:
        code = str(icb_code).strip()
        # Try exact match first, then prefix matching.
        if code in _ICB_PREFIX_MAP:
            return _ICB_PREFIX_MAP[code]
        for prefix_len in (3, 2):
            prefix = code[:prefix_len]
            for k, v in _ICB_PREFIX_MAP.items():
                if k.startswith(prefix):
                    return v

    if industry_label:
        label_lower = industry_label.strip().lower()
        if label_lower in _LABEL_MAP:
            return _LABEL_MAP[label_lower]
        # Substring match.
        for keyword, industry in _LABEL_MAP.items():
            if keyword in label_lower:
                return industry

    return "DEFAULT"


# ── Core computation ──────────────────────────────────────────────────

def compute_fair_value(args: dict[str, Any]) -> dict[str, Any]:
    """Compute fair value deterministically from gathered data.

    Expected args (all from LLM tool calls):
        eps: float               — EPS (đồng/cổ phiếu)
        bvps: float              — BVPS (đồng/cổ phiếu)
        roe: float               — ROE hiện tại (0.xx)
        live_price: float        — Giá hiện tại (đồng)
        profit_history: list     — [{year, profit_after_tax}] (tỷ đồng hoặc đồng)
        industry_icb_code: str   — ICB code (optional)
        industry_label: str      — Tên ngành (optional)
        median_pe: float         — Trung vị PE lịch sử (optional, for cross-check)
        median_pb: float         — Trung vị PB lịch sử (optional, for cross-check)
        forecast_profit: float   — Lợi nhuận dự báo (tỷ đồng, optional)
        forecast_revenue: float  — Doanh thu dự báo (tỷ đồng, optional)
        forecast_series: list    — Chuỗi dự báo [{year, revenue_pred, profit_pred}]
        cplh: float              — Cổ phiếu lưu hành (optional, for forward EPS/revenue)
    """
    try:
        eps = float(args.get("eps", 0))
        bvps = float(args.get("bvps", 0))
        roe = float(args.get("roe", 0))
        live_price = float(args.get("live_price", 0))
        profit_history = args.get("profit_history", [])
        industry_icb_code = args.get("industry_icb_code")
        industry_label = args.get("industry_label")
        median_pe = args.get("median_pe")
        median_pb = args.get("median_pb")
        median_ps = args.get("median_ps")
        live_ps = args.get("live_ps")
        forecast_profit = args.get("forecast_profit")
        forecast_revenue = args.get("forecast_revenue")
        forecast_series = args.get("forecast_series", [])
        cplh = args.get("cplh")

        # ── Resolve industry & playbook ──
        industry_key = resolve_industry(industry_icb_code, industry_label)
        playbook_entry = PLAYBOOK.get(industry_key, PLAYBOOK["DEFAULT"])
        weights = playbook_entry["weights"]
        params = playbook_entry["params"]
        coe = params["default_coe"]
        default_g = params["default_growth"]
        margin_pct = params.get("margin_pct", 10)

        # ── Compute growth: prefer forward forecast CAGR when the user asks for a future horizon.
        historical_cagr = _compute_profit_cagr(profit_history)
        forecast_cagr = _compute_forecast_profit_cagr(profit_history, forecast_series)
        cagr = forecast_cagr if forecast_cagr is not None else historical_cagr

        # ── Guardrail: g must be well below CoE ──
        # Use a minimum spread of 3 percentage points (or 25% of CoE) to prevent
        # the Gordon model denominator from becoming too small.
        min_spread = max(0.03, coe * 0.25)
        g_max = coe - min_spread
        g = min(cagr, g_max) if cagr is not None else default_g
        g = max(g, 0.0)  # Floor at 0

        # ── Use forward EPS if forecast available ──
        forward_eps = eps
        if forecast_profit is not None and cplh and cplh > 0:
            # forecast_profit is in tỷ đồng, convert to đồng
            forecast_profit_dong = float(forecast_profit) * 1e9
            forward_eps = forecast_profit_dong / float(cplh)

        logger.info(
            "[VALUATION] symbol=%s roe=%.4f eps=%.2f bvps=%.2f cplh=%s "
            "live_price=%.0f forward_eps=%.2f cagr=%s g=%.4f coe=%.2f "
            "median_pe=%s median_pb=%s forecast_profit=%s profit_history_len=%d",
            args.get("symbol"), roe, eps, bvps, cplh,
            live_price, forward_eps, cagr, g, coe,
            median_pe, median_pb, forecast_profit, len(profit_history),
        )

        # ── Compute target multiples ──
        if roe <= 0 or (coe - g) <= 0:
            # Can't compute — use median-based approach
            pe_target = float(median_pe) if median_pe else 15.0
            pb_target = float(median_pb) if median_pb else 2.0
        else:
            pb_target = (roe - g) / (coe - g)
            pe_target = pb_target / roe

            # ── Cross-check with historical medians ──
            # Blend Gordon-derived multiples toward historical medians when
            # they diverge too much, to keep output grounded in market reality.
            # Skip when median_pe looks like bad data (< 3).
            if median_pe and float(median_pe) >= 3:
                pe_ratio = pe_target / float(median_pe)
                if pe_ratio > 1.5:
                    pe_target = float(median_pe) * 1.3
                    pb_target = pe_target * roe
                elif pe_ratio < 0.5:
                    pe_target = float(median_pe) * 0.7
                    pb_target = pe_target * roe

        # ── Compute prices ──
        price_pe = forward_eps * pe_target if forward_eps > 0 else 0
        price_pb = bvps * pb_target if bvps > 0 else 0

        # ── Weighted composite ──
        w_pe = weights.get("pe", 0.6)
        w_pb = weights.get("pb", 0.4)
        w_ps = weights.get("ps", 0.0)

        # Compute P/S-based price using forward revenue when available.
        price_ps = 0.0
        can_use_ps = False
        if w_ps > 0 and median_ps and median_ps > 0:
            revenue_per_share = 0.0
            if forecast_revenue is not None and cplh and cplh > 0:
                revenue_per_share = float(forecast_revenue) * 1e9 / float(cplh)
            elif live_ps and live_ps > 0:
                revenue_per_share = live_price / float(live_ps)
            if revenue_per_share > 0:
                price_ps = revenue_per_share * float(median_ps)
                can_use_ps = True

        if w_ps > 0 and not can_use_ps:
            if w_pb == 0:
                w_pb = w_ps
            else:
                w_pe += w_ps / 2
                w_pb += w_ps / 2
            w_ps = 0.0

        total_w = w_pe + w_pb + w_ps
        if total_w > 0:
            w_pe_norm = w_pe / total_w
            w_pb_norm = w_pb / total_w
            w_ps_norm = w_ps / total_w
        else:
            w_pe_norm = 0.6
            w_pb_norm = 0.4
            w_ps_norm = 0.0

        price_composite = (price_pe * w_pe_norm) + (price_pb * w_pb_norm) + (price_ps * w_ps_norm)
        price_composite = round(price_composite / 100) * 100  # Round to nearest 100

        # ── Comparison ──
        upside_pct = (
            ((price_composite - live_price) / live_price * 100)
            if live_price > 0
            else 0
        )

        if upside_pct > 15:
            verdict = "ĐỊNH GIÁ THẤP (rẻ)"
        elif upside_pct < -15:
            verdict = "ĐỊNH GIÁ CAO (đắt)"
        else:
            verdict = "HỢP LÝ"

        return {
            "industry_key": industry_key,
            "method": playbook_entry["method"],
            "weights_used": (
                f"{w_pe_norm*100:.0f}% P/E + {w_pb_norm*100:.0f}% P/B + {w_ps_norm*100:.0f}% P/S"
                if w_ps_norm > 0
                else f"{w_pe_norm*100:.0f}% P/E + {w_pb_norm*100:.0f}% P/B"
            ),
            "coe_pct": coe * 100,
            "g_pct": round(g * 100, 2),
            "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
            "historical_cagr_pct": round(historical_cagr * 100, 2) if historical_cagr is not None else None,
            "forecast_cagr_pct": round(forecast_cagr * 100, 2) if forecast_cagr is not None else None,
            "growth_source": "forecast" if forecast_cagr is not None else "historical",
            "pe_target": round(pe_target, 2),
            "pb_target": round(pb_target, 2),
            "eps_used": round(forward_eps, 2),
            "bvps_used": round(bvps, 2),
            "price_pe": round(price_pe / 100) * 100,
            "price_pb": round(price_pb / 100) * 100,
            "price_ps": round(price_ps / 100) * 100,
            "price_composite": price_composite,
            "live_price": live_price,
            "upside_pct": round(upside_pct, 1),
            "verdict": verdict,
            "rationale": playbook_entry.get("rationale", ""),
            "forecast_series": forecast_series if forecast_series else None,
            "forecast_top_factors": args.get("forecast_top_factors") or None,
            "forecast_quality": args.get("forecast_quality") or None,
            "symbol": args.get("symbol", ""),
            "company_name": args.get("company_name", ""),
            "target_year": args.get("target_year", ""),
            "summary": (
                f"Định giá năm {args.get('target_year', '?')}. "
                f"Ngành {industry_key}, phương pháp {'%d%% P/E + %d%% P/B + %d%% P/S' % (w_pe_norm*100, w_pb_norm*100, w_ps_norm*100) if w_ps_norm > 0 else '%d%% P/E + %d%% P/B' % (w_pe_norm*100, w_pb_norm*100)}. "
                f"CAGR lợi nhuận ({'dự phóng' if forecast_cagr is not None else 'lịch sử'}) {'N/A' if cagr is None else f'{cagr*100:.1f}%'}, "
                f"tốc độ tăng trưởng bền vững g={g*100:.1f}% (CoE={coe*100:.0f}%). "
                f"P/E target {pe_target:.1f}x, P/B target {pb_target:.1f}x. "
                f"Giá theo P/E: {round(price_pe/100)*100:,.0f}đ, "
                f"giá theo P/B: {round(price_pb/100)*100:,.0f}đ, "
                f"giá theo P/S: {round(price_ps/100)*100:,.0f}đ, "
                f"giá tổng hợp: {price_composite:,.0f}đ. "
                f"Giá hiện tại: {live_price:,.0f}đ → {verdict} ({upside_pct:+.1f}%)."
            ),
        }
    except Exception as exc:
        logger.exception("compute_fair_value error: %s", exc)
        return {"error": f"Lỗi tính toán: {str(exc)}"}


def _normalize_to_ty(value: float) -> float:
    """Normalize profit value to tỷ đồng.

    Backend financial series returns profit in đồng (e.g. 16_789_000_000_000).
    Forecast service returns profit in tỷ đồng (e.g. 17_252).
    """
    if abs(value) >= 1e9:
        return value / 1e9
    return value


def _compute_forecast_profit_cagr(
    profit_history: list[Any],
    forecast_series: list[Any],
) -> float | None:
    """Compute CAGR from latest actual profit to target-year forecast profit."""
    if not forecast_series or not isinstance(forecast_series, list):
        return None

    actual_entries: list[tuple[int, float]] = []
    for item in profit_history or []:
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        profit = item.get("profit_after_tax") or item.get("profitAfterTax")
        if year is None or profit is None:
            continue
        try:
            parsed = (int(year), _normalize_to_ty(float(profit)))
        except (ValueError, TypeError):
            continue
        if parsed[1] > 0:
            actual_entries.append(parsed)

    forecast_entries: list[tuple[int, float]] = []
    for item in forecast_series:
        if not isinstance(item, dict):
            continue
        year = item.get("year") or item.get("predict_target_year")
        profit = item.get("profit_pred")
        if year is None or profit is None:
            continue
        try:
            parsed = (int(year), _normalize_to_ty(float(profit)))
        except (ValueError, TypeError):
            continue
        if parsed[1] > 0:
            forecast_entries.append(parsed)

    if not actual_entries or not forecast_entries:
        return None

    base_year, base_profit = max(actual_entries, key=lambda x: x[0])
    target_year, target_profit = max(forecast_entries, key=lambda x: x[0])
    n_years = target_year - base_year
    if n_years <= 0:
        return None

    try:
        cagr = (target_profit / base_profit) ** (1.0 / n_years) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None

    if not (-0.5 <= cagr <= 1.0):
        return None

    return cagr


def _compute_profit_cagr(profit_history: list[Any]) -> float | None:
    """Compute CAGR from a list of {year, profit_after_tax} entries.

    Handles edge cases:
    - Skips years with negative or zero profit
    - Uses the longest positive-to-positive span
    - Returns None if no valid span exists
    """
    if not profit_history or not isinstance(profit_history, list):
        return None

    # Parse and sort by year.
    entries: list[tuple[int, float]] = []
    for item in profit_history:
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        profit = item.get("profit_after_tax") or item.get("profitAfterTax")
        if year is not None and profit is not None:
            try:
                entries.append((int(year), float(profit)))
            except (ValueError, TypeError):
                continue

    entries.sort(key=lambda x: x[0])

    # Find longest span with positive endpoints.
    positive_entries = [(y, p) for y, p in entries if p > 0]
    if len(positive_entries) < 2:
        return None

    first_year, first_profit = positive_entries[0]
    last_year, last_profit = positive_entries[-1]
    n_years = last_year - first_year

    if n_years <= 0:
        return None

    try:
        cagr = (last_profit / first_profit) ** (1.0 / n_years) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None

    # Sanity check: CAGR should be between -50% and +100%.
    if not (-0.5 <= cagr <= 1.0):
        return None

    return cagr
