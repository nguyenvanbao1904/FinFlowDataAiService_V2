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
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Load playbook ────────────────────────────────────────────────────

_PLAYBOOK_PATH = Path(__file__).parent / "config" / "valuation_playbook.json"
_MACRO_CSV_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "macro" / "macro_yearly_train.csv"

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
    "2750": "MANUFACTURING_HEAVY",
    "2757": "MANUFACTURING_HEAVY",
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
    "máy công nghiệp": "MANUFACTURING_HEAVY",
    "ô tô": "MANUFACTURING_HEAVY",
    "phụ tùng": "MANUFACTURING_HEAVY",
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
        if code in _ICB_PREFIX_MAP:
            return _ICB_PREFIX_MAP[code]

    if industry_label:
        label_lower = industry_label.strip().lower()
        if label_lower in _LABEL_MAP:
            return _LABEL_MAP[label_lower]
        # Substring match.
        for keyword, industry in _LABEL_MAP.items():
            if keyword in label_lower:
                return industry

    return "DEFAULT"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _median(values: list[float | None], *, positive: bool = True) -> float | None:
    xs = [v for v in values if v is not None and math.isfinite(v) and (v > 0 if positive else True)]
    return statistics.median(xs) if xs else None


def _mean(values: list[float | None], *, positive: bool = False) -> float | None:
    xs = [v for v in values if v is not None and math.isfinite(v) and (v > 0 if positive else True)]
    return statistics.mean(xs) if xs else None


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    out = num / den
    return out if math.isfinite(out) else None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _discount(value: float, r: float, t: int) -> float:
    return value / ((1.0 + r) ** t)


def _normalized_deposit_rate() -> float | None:
    if not _MACRO_CSV_PATH.exists():
        return None
    import csv

    values: list[tuple[int, float]] = []
    with _MACRO_CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = _safe_float(row.get("year"))
            rate = _safe_float(row.get("interest_deposit_12m_pct"))
            if year is not None and rate is not None:
                values.append((int(year), rate))
    values.sort(key=lambda x: x[0])
    tail = [rate for _, rate in values[-5:]]
    return statistics.median(tail) / 100.0 if tail else None


def _forecast_rows(forecast_series: list[Any], target_year: int) -> dict[int, dict[str, float | None]]:
    rows: dict[int, dict[str, float | None]] = {}
    for item in forecast_series or []:
        if not isinstance(item, dict):
            continue
        year = _safe_float(item.get("year") or item.get("predict_target_year"))
        if year is None:
            continue
        year_int = int(year)
        if year_int > target_year:
            continue
        rows[year_int] = {
            "profit_after_tax": _safe_float(item.get("profit_pred") or item.get("profitAfterTax")),
            "revenue": _safe_float(item.get("revenue_pred") or item.get("revenue")),
        }
    return rows


def _valuation_features(args: dict[str, Any]) -> dict[str, Any]:
    annual = [r for r in args.get("profit_history", []) if isinstance(r, dict)]
    annual.sort(key=lambda r: int(_safe_float(r.get("year"), 0) or 0))
    eps = _safe_float(args.get("eps"), 0.0) or 0.0
    bvps = _safe_float(args.get("bvps"), 0.0) or 0.0
    roe_latest = _safe_float(args.get("roe"))
    median_pe = _safe_float(args.get("median_pe")) or _median([_safe_float(r.get("pe")) for r in annual])
    median_pb = _safe_float(args.get("median_pb")) or _median([_safe_float(r.get("pb")) for r in annual])
    live_price = _safe_float(args.get("live_price"), 0.0) or 0.0
    cplh = _safe_float(args.get("cplh"), 0.0) or 0.0

    roes = [_safe_float(r.get("roe")) for r in annual]
    roe_median = _median(roes)
    roe_avg3 = _mean(roes[-3:])
    if roe_latest and roe_avg3:
        roe_recency = 0.6 * roe_latest + 0.4 * roe_avg3
    else:
        roe_recency = roe_latest or roe_avg3 or roe_median
    roe_norm = min(roe_recency, roe_median) if roe_recency and roe_median else roe_recency

    payout_values = [
        _safe_float(r.get("payout_ratio"))
        for r in annual
        if (_safe_float(r.get("payout_ratio")) or 0) > 0
    ]
    payout = _mean(payout_values[-3:]) or 0.0
    payout_for_book = _clip(payout, 0.0, 1.0)

    cash_dividends = [_safe_float(r.get("cash_dividend")) for r in annual]
    cash_dividend_median = _median(cash_dividends)
    cash_div_years = sum(1 for v in cash_dividends if (v or 0) > 0)
    positive_dividends = [v for v in cash_dividends if (v or 0) > 0]
    dividend_growths = [_safe_ratio((cur or 0) - (prev or 0), prev) for prev, cur in zip(positive_dividends, positive_dividends[1:])]
    dividend_yield = _safe_ratio(cash_dividend_median, live_price)

    profit_growths = [_safe_float(r.get("profit_growth")) for r in annual]
    growth_values = [v for v in profit_growths if v is not None and math.isfinite(v)]
    profit_volatility = statistics.pstdev(growth_values) if len(growth_values) >= 3 else None

    target_year = int(_safe_float(args.get("target_year"), 0) or 0)
    forecast_rows = _forecast_rows(args.get("forecast_series", []), target_year)
    forecast_eps_growths: list[float | None] = []
    forecast_eps: list[float] = []
    if cplh > 0:
        for year in sorted(forecast_rows):
            profit = forecast_rows[year].get("profit_after_tax")
            if profit is not None:
                forecast_eps.append(profit * 1e9 / cplh)
        for prev, cur in zip(forecast_eps, forecast_eps[1:]):
            forecast_eps_growths.append(_safe_ratio(cur - prev, prev))

    return {
        "eps": eps,
        "bvps": bvps,
        "roe_latest": roe_latest,
        "roe_norm": roe_norm,
        "roe_median": roe_median,
        "median_pe": median_pe,
        "median_pb": median_pb,
        "live_price": live_price,
        "cplh": cplh,
        "payout": payout,
        "payout_for_book": payout_for_book,
        "cash_dividend_median": cash_dividend_median,
        "cash_div_years": cash_div_years,
        "dividend_consistency": cash_div_years / max(1, len(annual)),
        "cash_dividend_growth_median": _median(dividend_growths, positive=False),
        "dividend_yield": dividend_yield,
        "profit_volatility": profit_volatility,
        "forecast_eps_growth_median": _median(forecast_eps_growths, positive=False),
        "forecast_rows": forecast_rows,
        "indicator_years": len(annual),
    }


def _is_utility_label(industry_label: str | None) -> bool:
    label = (industry_label or "").lower()
    return any(word in label for word in ("điện", "nước"))


def _is_cyclical(industry_key: str, industry_label: str | None, profit_volatility: float | None) -> bool:
    label = (industry_label or "").lower()
    if industry_key in {"MANUFACTURING_HEAVY", "LOGISTICS_PORTS"}:
        return True
    if any(word in label for word in ("thép", "dầu khí", "phân bón", "hóa chất", "cao su", "vận tải", "thủy sản")):
        return True
    return profit_volatility is not None and profit_volatility > 0.50


def _choose_valuation_model(industry_key: str, industry_label: str | None, ft: dict[str, Any]) -> tuple[str, str, float]:
    dividend_yield = ft.get("dividend_yield") or 0.0
    forecast_eps_growth = ft.get("forecast_eps_growth_median")
    has_dividend_yield = (ft.get("dividend_consistency") or 0) >= 0.80 and dividend_yield >= 0.035
    forecast_growth_low = forecast_eps_growth is not None and forecast_eps_growth <= 0.05
    utility_like = industry_key == "UTILITIES" or _is_utility_label(industry_label)
    strongly_cyclical = _is_cyclical(industry_key, industry_label, ft.get("profit_volatility"))

    if industry_key == "BANK":
        return "bank_pe_pb_blend", "Ngân hàng: kết hợp P/B theo book value với forward P/E vì ROE, BVPS và lợi nhuận dự phóng đều quan trọng.", 0.85
    if industry_key == "SECURITIES":
        return "normalized_book_exit", "Công ty chứng khoán: lợi nhuận chu kỳ theo thị trường, dùng book/exit để neo giá trị vốn chủ.", 0.70
    if industry_key == "REAL_ESTATE":
        return "book_reference_low_confidence", "Bất động sản: thiếu NAV/quỹ đất chi tiết nên chỉ dùng book/PB tham chiếu với độ tin cậy thấp.", 0.45
    if has_dividend_yield and forecast_growth_low and ft.get("cash_dividend_median"):
        return "ddm_dividend_discount", "Cổ tức tiền mặt đều, tỷ suất cổ tức cao và tăng trưởng EPS thấp: dùng phương pháp chiết khấu cổ tức vì dòng tiền cổ tức là phần trọng yếu của luận điểm đầu tư.", 0.80
    if has_dividend_yield and strongly_cyclical and not utility_like:
        return "normalized_earnings", "Có cổ tức tiền mặt cao nhưng lợi nhuận mang tính chu kỳ: dùng EPS chuẩn hóa, cổ tức là điểm cộng khi đánh giá biên an toàn.", 0.65
    if utility_like:
        return "regulated_utility_book_exit", "Tiện ích điện/nước: dùng book/exit thận trọng vì lợi nhuận có thể biến động theo chu kỳ huy động.", 0.75
    if strongly_cyclical:
        return "normalized_earnings", "Lợi nhuận biến động/chu kỳ: dùng EPS chuẩn hóa theo lịch sử thay vì forecast một năm.", 0.65
    if (ft.get("dividend_consistency") or 0) >= 0.60:
        return "earnings_exit", "Doanh nghiệp có cổ tức tương đối đều nhưng vẫn còn tăng trưởng: dùng EPS dự phóng cộng hệ số thoát.", 0.75
    return "hybrid_reference", "Không có dấu hiệu đủ mạnh: dùng tham chiếu lai giữa earnings và book.", 0.60


def _implied_required_return(roe: float | None, payout: float, pb_median: float | None) -> float | None:
    if not roe or not pb_median or pb_median <= 0:
        return None
    g = roe * _clip(1.0 - payout, 0.0, 1.0)
    r = g + (roe - g) / pb_median
    return r if r and math.isfinite(r) and r > 0 else None


def _book_exit_value(ft: dict[str, Any], target_year: int) -> float | None:
    bvps = ft.get("bvps")
    cplh = ft.get("cplh")
    roe = ft.get("roe_norm")
    pb = ft.get("median_pb")
    payout = ft.get("payout_for_book") or 0.0
    if not bvps or not cplh or not roe or not pb:
        return None
    r = _implied_required_return(roe, payout, pb)
    if not r:
        return None
    current_bvps = bvps
    pv_dividends = 0.0
    forecast_rows = ft["forecast_rows"]
    base_year = min(forecast_rows.keys(), default=target_year)
    for year in range(base_year, target_year + 1):
        profit = (forecast_rows.get(year) or {}).get("profit_after_tax")
        eps = profit * 1e9 / cplh if profit is not None else roe * current_bvps
        dividend = payout * eps
        pv_dividends += _discount(dividend, r, year - base_year + 1)
        current_bvps += eps - dividend
    return pv_dividends + _discount(current_bvps * pb, r, max(1, target_year - base_year + 1))


def _earnings_exit_value(ft: dict[str, Any], target_year: int) -> float | None:
    pe = ft.get("median_pe")
    cplh = ft.get("cplh")
    if not pe or not cplh:
        return None
    r = _implied_required_return(ft.get("roe_norm"), ft.get("payout_for_book") or 0.0, ft.get("median_pb"))
    if not r:
        r = (1.0 / pe) + ((ft.get("roe_norm") or 0.0) * _clip(1.0 - (ft.get("payout_for_book") or 0.0), 0.0, 1.0))
    if not r or r <= 0:
        return None
    forecast_rows = ft["forecast_rows"]
    base_year = min(forecast_rows.keys(), default=target_year)
    pv_dividends = 0.0
    terminal_eps = None
    for year in range(base_year, target_year + 1):
        profit = (forecast_rows.get(year) or {}).get("profit_after_tax")
        if profit is None:
            continue
        eps = profit * 1e9 / cplh
        terminal_eps = eps
        pv_dividends += _discount((ft.get("payout_for_book") or 0.0) * eps, r, year - base_year + 1)
    terminal_eps = terminal_eps or ft.get("eps")
    return pv_dividends + _discount(terminal_eps * pe, r, max(1, target_year - base_year + 1)) if terminal_eps else None


def _normalized_earnings_value(ft: dict[str, Any]) -> float | None:
    eps = ft.get("eps")
    pe = ft.get("median_pe")
    return eps * pe if eps and pe else None


def _book_reference_value(ft: dict[str, Any]) -> float | None:
    bvps = ft.get("bvps")
    pb = ft.get("median_pb")
    return bvps * pb if bvps and pb else None


def _ddm_assumptions(industry_key: str, industry_label: str | None, ft: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    base_rate = _normalized_deposit_rate()
    if not base_rate:
        return None, None, None
    if (ft.get("dividend_consistency") or 0) >= 0.80 and (ft.get("dividend_yield") or 0.0) >= 0.035:
        multiplier = 1.5
    elif _is_cyclical(industry_key, industry_label, ft.get("profit_volatility")):
        multiplier = 2.0
    else:
        multiplier = 1.75
    ke = base_rate * multiplier
    dividend_growth = ft.get("cash_dividend_growth_median") or 0.0
    if _is_cyclical(industry_key, industry_label, ft.get("profit_volatility")):
        terminal_g_cap = 0.0
    elif industry_key in {"FMCG", "HEALTHCARE"} or _is_utility_label(industry_label):
        terminal_g_cap = base_rate * 0.6
    else:
        terminal_g_cap = base_rate * 0.3
    terminal_g = _clip(dividend_growth, 0.0, terminal_g_cap)
    if terminal_g >= ke:
        terminal_g = max(0.0, ke - base_rate * 0.5)
    return base_rate, ke, terminal_g


def _ddm_value(industry_key: str, industry_label: str | None, ft: dict[str, Any], target_year: int) -> float | None:
    cplh = ft.get("cplh")
    payout = ft.get("payout_for_book") or 0.0
    if not cplh or payout <= 0:
        return None
    _, ke, terminal_g = _ddm_assumptions(industry_key, industry_label, ft)
    if not ke or terminal_g is None or ke <= terminal_g:
        return None
    forecast_rows = ft["forecast_rows"]
    base_year = min(forecast_rows.keys(), default=target_year)
    pv_dividends = 0.0
    last_dividend = None
    normalized_cash_dividend = ft.get("cash_dividend_median")
    use_normalized_terminal_dividend = (
        normalized_cash_dividend is not None
        and (ft.get("dividend_consistency") or 0.0) >= 0.80
        and (ft.get("dividend_yield") or 0.0) >= 0.035
    )
    for year in range(base_year, target_year + 1):
        profit = (forecast_rows.get(year) or {}).get("profit_after_tax")
        eps = profit * 1e9 / cplh if profit is not None else ft.get("eps")
        if not eps:
            return None
        dividend = min(max(payout * eps, 0.0), eps)
        pv_dividends += _discount(dividend, ke, year - base_year + 1)
        last_dividend = dividend
    if last_dividend is None:
        return None
    terminal_dividend = last_dividend
    if use_normalized_terminal_dividend:
        terminal_dividend = max(last_dividend, normalized_cash_dividend or 0.0)
    terminal = terminal_dividend * (1.0 + terminal_g) / (ke - terminal_g)
    return pv_dividends + _discount(terminal, ke, max(1, target_year - base_year + 1))


def _bank_pe_pb_blend_value(ft: dict[str, Any], target_year: int) -> float | None:
    cplh = ft.get("cplh")
    own_pe = ft.get("median_pe")
    if not cplh or not own_pe:
        return _book_exit_value(ft, target_year)
    target_pe = own_pe * 1.4
    forecast_rows = ft["forecast_rows"]
    base_year = min(forecast_rows.keys(), default=target_year)
    profit = (forecast_rows.get(base_year) or {}).get("profit_after_tax")
    forward_eps = profit * 1e9 / cplh if profit is not None else ft.get("eps")
    pe_value = forward_eps * target_pe if forward_eps else None
    book_value = _book_exit_value(ft, target_year)
    values = [v for v in (pe_value, book_value) if v is not None and math.isfinite(v) and v > 0]
    return max(values) if values else None


def _hybrid_value(ft: dict[str, Any], target_year: int) -> float | None:
    candidates = [
        _book_exit_value(ft, target_year),
        _earnings_exit_value(ft, target_year),
        _normalized_earnings_value(ft),
        _book_reference_value(ft),
    ]
    values = [v for v in candidates if v is not None and math.isfinite(v) and v > 0]
    return statistics.median(values) if values else None


_PUBLIC_METHODS: dict[str, dict[str, str]] = {
    "bank_pe_pb_blend": {
        "label": "Phương pháp kết hợp P/B và P/E dự phóng",
        "formula": "Giá hợp lý = kết hợp giá theo P/B dựa trên BVPS/ROE với giá theo P/E dự phóng từ lợi nhuận tương lai.",
    },
    "normalized_book_exit": {
        "label": "Phương pháp giá trị sổ sách chuẩn hóa",
        "formula": "Giá hợp lý = hiện giá cổ tức giữ lại trong giai đoạn dự phóng + giá trị sổ sách cuối kỳ nhân P/B lịch sử.",
    },
    "book_reference_low_confidence": {
        "label": "Phương pháp tham chiếu giá trị sổ sách",
        "formula": "Giá hợp lý = BVPS hiện tại x P/B trung vị lịch sử; dùng như tham chiếu khi thiếu NAV chi tiết.",
    },
    "ddm_dividend_discount": {
        "label": "Phương pháp chiết khấu cổ tức",
        "formula": "Giá hợp lý = hiện giá cổ tức tiền mặt dự phóng + hiện giá giá trị cuối kỳ; giá trị cuối kỳ = cổ tức năm kế tiếp / (tỷ suất sinh lời yêu cầu - tăng trưởng dài hạn).",
    },
    "regulated_utility_book_exit": {
        "label": "Phương pháp giá trị sổ sách cho doanh nghiệp tiện ích",
        "formula": "Giá hợp lý = hiện giá cổ tức giữ lại trong giai đoạn dự phóng + BVPS cuối kỳ x P/B trung vị lịch sử.",
    },
    "earnings_exit": {
        "label": "Phương pháp lợi nhuận dự phóng cộng hệ số thoát",
        "formula": "Giá hợp lý = hiện giá cổ tức dự kiến + EPS năm mục tiêu x P/E trung vị lịch sử.",
    },
    "normalized_earnings": {
        "label": "Phương pháp lợi nhuận chuẩn hóa theo chu kỳ",
        "formula": "Giá hợp lý = EPS chuẩn hóa x P/E trung vị lịch sử. Cách này phù hợp khi lợi nhuận biến động theo chu kỳ và không nên lấy riêng một năm làm đại diện.",
    },
    "hybrid_reference": {
        "label": "Phương pháp tham chiếu tổng hợp",
        "formula": "Giá hợp lý = trung vị các kết quả hợp lệ từ lợi nhuận dự phóng, giá trị sổ sách và tham chiếu lịch sử.",
    },
}


def _public_method(model: str) -> dict[str, str]:
    base_model = str(model or "").split("+", 1)[0]
    return _PUBLIC_METHODS.get(base_model, _PUBLIC_METHODS["hybrid_reference"])


def _compute_router_value(args: dict[str, Any]) -> dict[str, Any]:
    industry_key = resolve_industry(args.get("industry_icb_code"), args.get("industry_label"))
    playbook_entry = PLAYBOOK.get(industry_key, PLAYBOOK.get("DEFAULT", {}))
    target_year = int(_safe_float(args.get("target_year"), 0) or 0)
    ft = _valuation_features(args)
    model, reason, confidence = _choose_valuation_model(industry_key, args.get("industry_label"), ft)
    if model == "bank_pe_pb_blend":
        value = _bank_pe_pb_blend_value(ft, target_year)
    elif model == "ddm_dividend_discount":
        value = _ddm_value(industry_key, args.get("industry_label"), ft, target_year)
    elif model in {"normalized_book_exit", "regulated_utility_book_exit"}:
        value = _book_exit_value(ft, target_year)
    elif model == "earnings_exit":
        value = _earnings_exit_value(ft, target_year)
    elif model == "normalized_earnings":
        value = _normalized_earnings_value(ft)
    elif model == "book_reference_low_confidence":
        value = _book_reference_value(ft)
    else:
        value = _hybrid_value(ft, target_year)
    if value is None:
        value = _hybrid_value(ft, target_year)
        model = f"{model}+fallback_hybrid"
        confidence = min(confidence, 0.45)
    public_method = _public_method(model)
    price_composite = round((value or 0.0) / 100) * 100
    live_price = ft.get("live_price") or 0.0
    upside_pct = ((price_composite - live_price) / live_price * 100) if live_price > 0 else 0.0
    if upside_pct > 15:
        verdict = "ĐỊNH GIÁ THẤP (rẻ)"
    elif upside_pct < -15:
        verdict = "ĐỊNH GIÁ CAO (đắt)"
    else:
        verdict = "HỢP LÝ"
    base_rate, ddm_ke, ddm_g = _ddm_assumptions(industry_key, args.get("industry_label"), ft)
    pe_value = _earnings_exit_value(ft, target_year) or _normalized_earnings_value(ft) or 0.0
    pb_value = _book_exit_value(ft, target_year) or _book_reference_value(ft) or 0.0
    key_assumptions = {
        "roe_norm_pct": round((ft.get("roe_norm") or 0.0) * 100, 2),
        "cash_dividend_payout_to_profit_pct": round((ft.get("payout") or 0.0) * 100, 2),
        "cash_dividend_yield_median_pct": round((ft.get("dividend_yield") or 0.0) * 100, 2),
        "forecast_eps_growth_median_pct": round((ft.get("forecast_eps_growth_median") or 0.0) * 100, 2),
        "profit_volatility": round(ft["profit_volatility"], 4) if ft.get("profit_volatility") is not None else None,
        "base_deposit_rate_pct": round((base_rate or 0.0) * 100, 2),
        "ddm_required_return_pct": round((ddm_ke or 0.0) * 100, 2),
        "ddm_terminal_growth_pct": round((ddm_g or 0.0) * 100, 2),
        "valuation_formula": public_method["formula"],
        "playbook_method": playbook_entry.get("method"),
        "playbook_rationale": playbook_entry.get("rationale"),
        "playbook_source": playbook_entry.get("source"),
    }
    summary = (
        f"Định giá năm {target_year}. Phương pháp định giá: {public_method['label']}. "
        f"Giá hợp lý khoảng {price_composite:,.0f}đ/cp. "
        f"Giá hiện tại {live_price:,.0f}đ → {verdict} ({upside_pct:+.1f}%)."
    )
    return {
        "industry_key": industry_key,
        "industry_label": args.get("industry_label"),
        "method": public_method["label"],
        "weights_used": "model-router",
        "valuation_model": public_method["label"],
        "valuation_formula": public_method["formula"],
        "model_reason": reason,
        "model_confidence": confidence,
        "key_assumptions": key_assumptions,
        "coe_pct": key_assumptions["ddm_required_return_pct"],
        "g_pct": key_assumptions["ddm_terminal_growth_pct"],
        "cagr_pct": key_assumptions["forecast_eps_growth_median_pct"],
        "historical_cagr_pct": None,
        "forecast_cagr_pct": key_assumptions["forecast_eps_growth_median_pct"],
        "growth_source": "forecast",
        "pe_target": round((ft.get("median_pe") or 0.0), 2),
        "pb_target": round((ft.get("median_pb") or 0.0), 2),
        "eps_used": round((ft.get("eps") or 0.0), 2),
        "bvps_used": round((ft.get("bvps") or 0.0), 2),
        "price_pe": round(pe_value / 100) * 100,
        "price_pb": round(pb_value / 100) * 100,
        "price_ps": 0,
        "price_composite": price_composite,
        "live_price": live_price,
        "upside_pct": round(upside_pct, 1),
        "verdict": verdict,
        "rationale": reason,
        "forecast_series": args.get("forecast_series") or None,
        "forecast_top_factors": args.get("forecast_top_factors") or None,
        "forecast_quality": args.get("forecast_quality") or None,
        "symbol": args.get("symbol", ""),
        "company_name": args.get("company_name", ""),
        "target_year": target_year,
        "summary": summary,
    }


# ── Core computation ──────────────────────────────────────────────────

def compute_fair_value(args: dict[str, Any]) -> dict[str, Any]:
    """Compute fair value using the production valuation model router."""
    try:
        return _compute_router_value(args)
    except Exception as exc:
        logger.exception("compute_fair_value error: %s", exc)
        return {"error": f"Lỗi tính toán: {str(exc)}"}
