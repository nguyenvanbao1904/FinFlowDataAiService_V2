"""Load financial training data directly from MySQL DB.

Single source of truth for both run_final_model_pipeline.py and
test_final_models_forecast.py. Replaces the old CSV export + rebuild flow.

Queries join balance_sheets, income_statements, financial_indicators,
cash_flow_statements, and companies tables. Returns annual DataFrames
with all features needed for model training/inference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings


def get_db_connection():
    import pymysql
    return pymysql.connect(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _query_to_df(conn, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Bank data ────────────────────────────────────────────────────────

_BANK_QUERY = """
SELECT
    c.id AS symbol, c.company_name, c.exchange, c.company_type,
    bs.year, bs.quarter,
    ist.profit_after_tax,
    bis.net_interest_income,
    bis.net_fee_commission_income AS fee_and_commission_income,
    bis.net_other_income_expenses AS other_income,
    bis.interest_expense,
    bis.total_operating_income,
    bis.total_operating_expense,
    bis.credit_risk_provisions_expense,
    bis.interest_and_similar_income,
    bs.cash_and_equivalents, bs.equity, bs.total_assets, bs.total_capital,
    bbs.total_liabilities,
    bbs.balances_with_sbv           AS deposits_at_sbv,
    bbs.interbank_placements_loans  AS interbank_placements,
    bbs.trading_securities,
    bbs.investment_securities,
    bbs.loans_to_customers          AS customer_loans,
    bbs.gov_sbv_debt                AS sbv_borrowings,
    bbs.deposits_from_customers     AS customer_deposits,
    bbs.issuing_valuable_paper      AS valuable_papers,
    bbs.deposits_borrowings_others,
    fi.nim, fi.roe, fi.roa, fi.pe, fi.pb, fi.ps, fi.eps, fi.bvps,
    fi.npl_to_loan,
    fi.loanloss_reserves_to_npl,
    fi.cir,
    fi.ldr,
    fi.cof,
    fi.yoea,
    fi.sale_growth,
    fi.profit_growth
FROM companies c
JOIN balance_sheets bs ON bs.company_id = c.id AND bs.company_type = 'BANK'
JOIN bank_balance_sheets bbs ON bbs.id = bs.id
JOIN income_statements ist
     ON ist.company_id = c.id AND ist.year = bs.year
     AND ist.quarter = bs.quarter AND ist.company_type = 'BANK'
JOIN bank_income_statements bis ON bis.id = ist.id
LEFT JOIN financial_indicators fi
     ON fi.company_id = c.id AND fi.year = bs.year
     AND fi.quarter = bs.quarter AND fi.company_type = 'BANK'
WHERE c.company_type = 'BANK'
ORDER BY c.id, bs.year, bs.quarter
"""

_BANK_FLOW_COLS = [
    "profit_after_tax", "net_interest_income", "fee_and_commission_income",
    "other_income", "interest_expense", "total_operating_income",
    "total_operating_expense", "credit_risk_provisions_expense",
    "interest_and_similar_income",
]

_BANK_STOCK_COLS = [
    "cash_and_equivalents", "equity", "total_assets", "total_capital",
    "total_liabilities", "deposits_at_sbv", "interbank_placements",
    "trading_securities", "investment_securities", "customer_loans",
    "sbv_borrowings", "customer_deposits", "valuable_papers",
    "deposits_borrowings_others", "bvps", "eps", "pe", "pb", "ps",
]

_BANK_INDICATOR_COLS = [
    "nim", "roe", "roa", "npl_to_loan", "loanloss_reserves_to_npl",
    "cir", "ldr", "cof", "yoea", "sale_growth", "profit_growth",
]


# ── Non-bank data ────────────────────────────────────────────────────

_NONBANK_QUERY = """
SELECT
    c.id AS symbol, c.company_name, c.exchange, c.company_type,
    bs.year, bs.quarter,
    ist.profit_after_tax,
    nbis.net_revenue, nbis.total_revenue, nbis.net_profit,
    nbis.gross_profit,
    nbis.cost_of_goods_sold,
    nbis.selling_expense,
    nbis.managing_expense,
    bs.cash_and_equivalents, bs.equity, bs.total_assets, bs.total_capital,
    nbbs.total_liabilities,
    nbbs.short_term_investments, nbbs.short_term_receivables,
    nbbs.inventories, nbbs.fixed_assets, nbbs.long_term_receivables,
    nbbs.short_term_borrowings, nbbs.long_term_borrowings,
    nbbs.advances_from_customers,
    fi.roe, fi.roa, fi.pe, fi.pb, fi.ps, fi.eps, fi.bvps,
    fi.gross_margin, fi.net_margin,
    fi.current_ratio,
    fi.total_debt_over_equity,
    fi.ev_over_ebitda,
    fi.inventory_turnover,
    fi.sale_growth,
    fi.profit_growth,
    cf.operating_cashflow,
    cf.investing_cashflow,
    cf.financing_cashflow
FROM companies c
JOIN balance_sheets bs ON bs.company_id = c.id AND bs.company_type = 'NON_BANK'
JOIN non_bank_balance_sheets nbbs ON nbbs.id = bs.id
JOIN income_statements ist
     ON ist.company_id = c.id AND ist.year = bs.year
     AND ist.quarter = bs.quarter AND ist.company_type = 'NON_BANK'
JOIN non_bank_income_statements nbis ON nbis.id = ist.id
LEFT JOIN financial_indicators fi
     ON fi.company_id = c.id AND fi.year = bs.year
     AND fi.quarter = bs.quarter AND fi.company_type = 'NORMAL'
LEFT JOIN cash_flow_statements cf
     ON cf.company_id = c.id AND cf.year = bs.year
     AND cf.quarter = bs.quarter
WHERE c.company_type = 'NON_BANK'
ORDER BY c.id, bs.year, bs.quarter
"""

_NONBANK_FLOW_COLS = [
    "profit_after_tax", "net_revenue", "total_revenue", "gross_profit",
    "cost_of_goods_sold", "selling_expense", "managing_expense",
    "operating_cashflow", "investing_cashflow", "financing_cashflow",
]

_NONBANK_STOCK_COLS = [
    "cash_and_equivalents", "equity", "total_assets", "total_capital",
    "total_liabilities", "short_term_investments", "short_term_receivables",
    "inventories", "fixed_assets", "long_term_receivables",
    "short_term_borrowings", "long_term_borrowings", "advances_from_customers",
    "bvps", "eps", "pe", "pb", "ps",
]

_NONBANK_INDICATOR_COLS = [
    "roe", "roa", "gross_margin", "net_margin",
    "current_ratio", "total_debt_over_equity", "ev_over_ebitda",
    "inventory_turnover", "sale_growth", "profit_growth",
]


# ── Aggregation helpers ──────────────────────────────────────────────

def _aggregate_quarterly_to_annual(
    df: pd.DataFrame,
    flow_cols: list[str],
    stock_cols: list[str],
    indicator_cols: list[str],
) -> pd.DataFrame:
    """Aggregate quarterly rows to annual.

    Flow items: sum of 4 quarters, converted to tỷ đồng.
    Stock items: last quarter value, converted to tỷ đồng if > 1M.
    Indicator cols: Q4 value (TTM), fallback to last quarter.
    Requires 4 quarters per year.
    """
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "quarter"])

    id_cols = ["symbol", "company_name", "exchange", "company_type"]

    records = []
    for (sym, year), group in df.groupby(["symbol", "year"]):
        group = group.sort_values("quarter")
        if len(group) < 4:
            continue

        row: dict[str, Any] = {}
        for c in id_cols:
            if c in group.columns:
                row[c] = group.iloc[-1][c]
        row["symbol"] = sym
        row["year"] = int(year)

        for c in flow_cols:
            if c in group.columns:
                vals = pd.to_numeric(group[c], errors="coerce")
                row[c] = vals.sum() / 1e9

        for c in stock_cols:
            if c in group.columns:
                val = pd.to_numeric(group.iloc[-1][c], errors="coerce")
                row[c] = val / 1e9 if pd.notna(val) and abs(val) > 1e6 else val

        q4 = group[group["quarter"] == 4]
        src = q4.iloc[0] if not q4.empty else group.iloc[-1]
        for c in indicator_cols:
            if c in group.columns:
                val = pd.to_numeric(src[c], errors="coerce")
                row[c] = float(val) if pd.notna(val) else np.nan

        records.append(row)

    return pd.DataFrame(records)


# ── Bank derived features ────────────────────────────────────────────

def _add_bank_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["nim_pct"] = pd.to_numeric(out.get("nim", np.nan), errors="coerce") * 100
    out["roe"] = pd.to_numeric(out.get("roe", np.nan), errors="coerce")
    out["roa"] = pd.to_numeric(out.get("roa", np.nan), errors="coerce")

    eq = pd.to_numeric(out.get("equity", 0), errors="coerce").fillna(0)
    ta = pd.to_numeric(out.get("total_assets", 0), errors="coerce").fillna(0)
    out["assets_to_equity"] = np.where(eq != 0, ta / eq, 0.0)

    nii = pd.to_numeric(out.get("net_interest_income", 0), errors="coerce").fillna(0)
    out["revenue_current"] = nii

    for c in ["npl_to_loan", "loanloss_reserves_to_npl", "cir", "ldr", "cof", "yoea"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(["symbol", "year"]).reset_index(drop=True)
    by_sym = out.groupby("symbol", observed=False)

    out["revenue_next"] = by_sym["revenue_current"].shift(-1)
    out["profit_after_tax_next"] = by_sym["profit_after_tax"].shift(-1)

    rev_l1 = by_sym["revenue_current"].shift(1)
    pf_l1 = by_sym["profit_after_tax"].shift(1)
    out["revenue_yoy_pct"] = np.where(
        rev_l1.fillna(0) != 0, ((out["revenue_current"] / rev_l1) - 1) * 100, 0)
    out["profit_yoy_pct"] = np.where(
        pf_l1.fillna(0) != 0, ((out["profit_after_tax"] / pf_l1) - 1) * 100, 0)

    # Fallbacks for missing indicator values
    nim_fb = np.where(ta != 0, (nii / ta) * 100, 0)
    out["nim_pct"] = out["nim_pct"].fillna(pd.Series(nim_fb, index=out.index))
    roe_fb = np.where(eq != 0, out["profit_after_tax"] / eq, 0)
    out["roe"] = out["roe"].fillna(pd.Series(roe_fb, index=out.index))
    roa_fb = np.where(ta != 0, out["profit_after_tax"] / ta, 0)
    out["roa"] = out["roa"].fillna(pd.Series(roa_fb, index=out.index))

    out = out.drop(columns=["nim"], errors="ignore")
    return out


# ── Non-bank derived features ────────────────────────────────────────

def _add_nonbank_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["roe"] = pd.to_numeric(out.get("roe", np.nan), errors="coerce")
    out["roa"] = pd.to_numeric(out.get("roa", np.nan), errors="coerce")

    rev = pd.to_numeric(
        out.get("net_revenue", out.get("total_revenue", 0)), errors="coerce").fillna(0)
    pf = pd.to_numeric(out.get("profit_after_tax", 0), errors="coerce").fillna(0)
    eq = pd.to_numeric(out.get("equity", 0), errors="coerce").fillna(0)
    ta = pd.to_numeric(out.get("total_assets", 0), errors="coerce").fillna(0)

    out["total_assets_reported"] = ta
    out["total_capital_reported"] = pd.to_numeric(
        out.get("total_capital", 0), errors="coerce").fillna(0)
    out["revenue_current"] = rev
    out["gross_margin"] = pd.to_numeric(out.get("gross_margin", 0), errors="coerce")
    out["net_margin"] = np.where(rev != 0, (pf / rev) * 100, 0)
    out["profit_margin_calc_pct"] = out["net_margin"]

    gross_profit = pd.to_numeric(out.get("gross_profit", 0), errors="coerce").fillna(0)
    cogs = pd.to_numeric(out.get("cost_of_goods_sold", 0), errors="coerce").fillna(0)
    selling = pd.to_numeric(out.get("selling_expense", 0), errors="coerce").fillna(0)
    managing = pd.to_numeric(out.get("managing_expense", 0), errors="coerce").fillna(0)
    op_cf = pd.to_numeric(out.get("operating_cashflow", 0), errors="coerce").fillna(0)
    inv_cf = pd.to_numeric(out.get("investing_cashflow", 0), errors="coerce").fillna(0)
    fin_cf = pd.to_numeric(out.get("financing_cashflow", 0), errors="coerce").fillna(0)

    out["gross_margin_calc_pct"] = np.where(rev != 0, (gross_profit / rev) * 100, 0)
    out["cogs_ratio_pct"] = np.where(rev != 0, (cogs.abs() / rev.abs()) * 100, 0)
    out["selling_expense_ratio_pct"] = np.where(rev != 0, (selling.abs() / rev.abs()) * 100, 0)
    out["managing_expense_ratio_pct"] = np.where(rev != 0, (managing.abs() / rev.abs()) * 100, 0)
    out["opex_ratio_pct"] = out["selling_expense_ratio_pct"] + out["managing_expense_ratio_pct"]
    out["operating_cashflow_to_profit"] = np.where(pf != 0, op_cf / pf, 0)
    out["operating_cashflow_to_revenue_pct"] = np.where(rev != 0, (op_cf / rev) * 100, 0)
    out["free_cashflow_proxy"] = op_cf + inv_cf
    out["free_cashflow_to_profit"] = np.where(pf != 0, out["free_cashflow_proxy"] / pf, 0)
    out["financing_cashflow_to_assets_pct"] = np.where(ta != 0, (fin_cf / ta) * 100, 0)

    net_debt_cols = ["short_term_borrowings", "long_term_borrowings"]
    net_debt = sum(pd.to_numeric(out.get(c, 0), errors="coerce").fillna(0)
                   for c in net_debt_cols)
    cash = pd.to_numeric(out.get("cash_and_equivalents", 0), errors="coerce").fillna(0)
    out["net_debt"] = net_debt - cash
    out["net_debt_to_equity_pct"] = np.where(eq != 0, (out["net_debt"] / eq) * 100, 0)

    recv = pd.to_numeric(out.get("short_term_receivables", 0), errors="coerce").fillna(0)
    out["receivable_ratio_pct"] = np.where(rev != 0, (recv / rev) * 100, 0)

    for c in ["current_ratio", "total_debt_over_equity", "ev_over_ebitda",
              "inventory_turnover", "sale_growth", "profit_growth"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(["symbol", "year"]).reset_index(drop=True)
    by_sym = out.groupby("symbol", observed=False)

    out["revenue_lag1"] = by_sym["revenue_current"].shift(1)
    out["revenue_lag2"] = by_sym["revenue_current"].shift(2)
    out["profit_lag1"] = by_sym["profit_after_tax"].shift(1)
    out["profit_lag2"] = by_sym["profit_after_tax"].shift(2)
    out["revenue_next"] = by_sym["revenue_current"].shift(-1)
    out["profit_after_tax_next"] = by_sym["profit_after_tax"].shift(-1)

    out["revenue_roll3_mean"] = by_sym["revenue_current"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    out["revenue_roll3_std"] = by_sym["revenue_current"].transform(
        lambda s: s.rolling(3, min_periods=1).std(ddof=0)).fillna(0)
    out["profit_roll3_mean"] = by_sym["profit_after_tax"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    out["profit_roll3_std"] = by_sym["profit_after_tax"].transform(
        lambda s: s.rolling(3, min_periods=1).std(ddof=0)).fillna(0)

    rev_l1 = out["revenue_lag1"].fillna(0)
    pf_l1 = out["profit_lag1"].fillna(0)
    out["revenue_yoy_pct"] = np.where(rev_l1 != 0, ((rev / rev_l1) - 1) * 100, 0)
    out["profit_yoy_pct"] = np.where(pf_l1 != 0, ((pf / pf_l1) - 1) * 100, 0)
    out["revenue_momentum_pct"] = np.where(
        rev_l1 != 0, ((rev - rev_l1) / rev_l1.abs()) * 100, 0)
    out["profit_momentum_pct"] = np.where(
        pf_l1 != 0, ((pf - pf_l1) / pf_l1.abs()) * 100, 0)
    out["gross_margin_change"] = by_sym["gross_margin_calc_pct"].diff().fillna(0)
    out["opex_ratio_change"] = by_sym["opex_ratio_pct"].diff().fillna(0)
    out["cash_conversion_change"] = by_sym["operating_cashflow_to_profit"].diff().fillna(0)

    # Fallbacks
    roe_fb = np.where(eq != 0, pf / eq, 0)
    out["roe"] = out["roe"].fillna(pd.Series(roe_fb, index=out.index))
    roa_fb = np.where(ta != 0, pf / ta, 0)
    out["roa"] = out["roa"].fillna(pd.Series(roa_fb, index=out.index))

    return out


# ── Public API ───────────────────────────────────────────────────────

def load_bank_annual(conn=None) -> pd.DataFrame:
    """Load bank annual data from DB with all derived features."""
    close = conn is None
    if conn is None:
        conn = get_db_connection()
    try:
        raw = _query_to_df(conn, _BANK_QUERY)
        if raw.empty:
            return pd.DataFrame()
        annual = _aggregate_quarterly_to_annual(
            raw, _BANK_FLOW_COLS, _BANK_STOCK_COLS, _BANK_INDICATOR_COLS)
        return _add_bank_derived(annual)
    finally:
        if close:
            conn.close()


def load_nonbank_annual(conn=None) -> pd.DataFrame:
    """Load non-bank annual data from DB with all derived features."""
    close = conn is None
    if conn is None:
        conn = get_db_connection()
    try:
        raw = _query_to_df(conn, _NONBANK_QUERY)
        if raw.empty:
            return pd.DataFrame()
        annual = _aggregate_quarterly_to_annual(
            raw, _NONBANK_FLOW_COLS, _NONBANK_STOCK_COLS, _NONBANK_INDICATOR_COLS)
        return _add_nonbank_derived(annual)
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    print("Loading bank data from DB...")
    bank = load_bank_annual()
    print(f"  {len(bank)} rows, {bank['symbol'].nunique()} symbols")
    acb = bank[bank["symbol"] == "ACB"].sort_values("year").tail(3)
    if not acb.empty:
        print("  ACB verification:")
        for _, r in acb.iterrows():
            print(f"    {int(r['year'])}: NIM={r['nim_pct']:.2f}%, "
                  f"ROE={r['roe']*100:.2f}%, ROA={r['roa']*100:.2f}%, "
                  f"NPL={r.get('npl_to_loan', 'N/A')}, CIR={r.get('cir', 'N/A')}")

    print("\nLoading nonbank data from DB...")
    nonbank = load_nonbank_annual()
    print(f"  {len(nonbank)} rows, {nonbank['symbol'].nunique()} symbols")
    hpg = nonbank[nonbank["symbol"] == "HPG"].sort_values("year").tail(3)
    if not hpg.empty:
        print("  HPG verification:")
        for _, r in hpg.iterrows():
            print(f"    {int(r['year'])}: ROE={r['roe']*100:.2f}%, "
                  f"ROA={r['roa']*100:.2f}%, "
                  f"OpCF={r.get('operating_cashflow', 'N/A')}, "
                  f"D/E={r.get('total_debt_over_equity', 'N/A')}")
