from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.financial_training.industry_train_features import _slug


COMMON_MACROS = [
    "gdp_ty_dong_log",
    "cpi_inflation_yoy_pp",
    "usd_vnd_log",
    "interest_deposit_12m_pct",
    "interest_loan_midlong_pct",
]

SECTOR_MACRO_MAPPINGS: dict[str, dict[str, Any]] = {
    "KIM_LO_I_C_NG_NGHI_P_KHAI_KHO_NG": {
        "name": "Nganh Thep va Khai khoang",
        "specific_macros": ["hrc_log", "iron_ore_log", "coal_log", "spread_hrc_iron"],
    },
    "CH_BI_N_THUY_S_N": {
        "name": "Nganh Thuy san",
        "specific_macros": ["usd_vnd_log", "bdry_shipping_etf_log"],
    },
    "S_N_XU_T_TH_C_PH_M": {
        "name": "Nganh Thuc pham (alias cho VHC/ANV trong map hien tai)",
        "specific_macros": ["usd_vnd_log", "bdry_shipping_etf_log"],
    },
    "S_N_XU_T_V_T_LI_U_X_Y_D_NG": {
        "name": "Vat lieu xay dung",
        "specific_macros": ["interest_loan_midlong_pct", "usd_vnd_log"],
    },
    "X_Y_D_NG_V_V_T_LI_U_X_Y_D_NG": {
        "name": "Xay dung va VLXD (alias cho VCS trong map hien tai)",
        "specific_macros": ["interest_loan_midlong_pct", "usd_vnd_log"],
    },
    "U_T_B_T_NG_S_N_V_D_CH_V": {
        "name": "Bat dong san va Dich vu",
        "specific_macros": [
            "cpi_inflation_yoy_pp",
            "gdp_ty_dong_log",
            "interest_loan_short_pct",
            "interest_loan_midlong_pct",
        ],
    },
    "B_N_L_": {
        "name": "Ban le",
        "specific_macros": ["cpi_inflation_yoy_pp", "gold_gc_log"],
    },
    "N_NG_L_NG": {
        "name": "Dau khi va Nang luong",
        "specific_macros": ["oil_brent_log", "nat_gas_log"],
    },
    "NG_N_H_NG": {
        "name": "Ngan hang",
        "specific_macros": ["interest_deposit_12m_pct", "interest_loan_short_pct"],
    },
    "C_NG_TY_CH_NG_KHO_N": {
        "name": "Cong ty chung khoan",
        "specific_macros": [
            "vnindex_daily_return_mean_pct",
            "vnindex_growth_yoy_pct",
            "vnindex_trading_volume_avg",
            "vnindex_trading_value_avg",
        ],
    },
}

PERCENT_LIKE_COLUMNS = [
    "roe",
    "roa",
    "nim_pct",
    "gross_margin",
    "net_margin",
    "profit_margin_calc_pct",
    "net_debt_to_equity_pct",
    "receivable_ratio_pct",
    "revenue_yoy_pct",
    "profit_yoy_pct",
    "revenue_momentum_pct",
    "profit_momentum_pct",
    "asset_growth_yoy",
    "profit_margin_change",
    "interest_deposit_12m_pct",
    "interest_loan_short_pct",
    "interest_loan_midlong_pct",
]

MACRO_LOG1P_NONNEGATIVE_COLUMNS = [
    "vnindex_trading_volume_avg",
    "vnindex_trading_value_avg",
]

EXTRA_ROBUST_CLIP_COLUMNS = [
    "vnindex_daily_return_mean_pct",
    "vnindex_growth_yoy_pct",
] + MACRO_LOG1P_NONNEGATIVE_COLUMNS


DEBT_INTEREST_ADJUSTMENT = {
    "interest_col": "interest_loan_midlong_pct",
    "debt_col": "net_debt_to_equity_pct",
    "debt_lag_col": "net_debt_to_equity_pct_lag1",
}


def _model(objective: str) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=450,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective=objective,
        random_state=42,
        n_jobs=-1,
    )


def _merge_macro(df: pd.DataFrame, macro_csv: Path, lag_years: int = 1) -> pd.DataFrame:
    m = pd.read_csv(macro_csv)
    m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["year"]).astype({"year": int})
    if lag_years:
        m["year"] = m["year"] + int(lag_years)
    return df.merge(m, on="year", how="left")


def _merge_industry(df: pd.DataFrame, map_csv: Path) -> pd.DataFrame:
    m = pd.read_csv(map_csv)
    m["symbol"] = m["symbol"].astype(str).str.upper().str.strip()
    m["industry_group"] = m["industry_group"].astype(str).map(_slug)
    m = m[["symbol", "industry_group"]].drop_duplicates(subset=["symbol"], keep="last")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out = out.merge(m, on="symbol", how="left")
    out["industry_group"] = out["industry_group"].fillna("UNKNOWN").map(_slug)
    return out


def _drop_identifier_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    id_cols = [c for c in out.columns if str(c).lower() == "id" or str(c).lower().endswith("_id")]
    if id_cols:
        out = out.drop(columns=id_cols, errors="ignore")
    return out, sorted(id_cols)


def _normalize_percent_like_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    scaled: list[str] = []
    for c in columns:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        v = s.dropna()
        out[c] = s
        if v.empty:
            continue
        # If the bulk is in [-1.5, 1.5], treat it as ratio and convert to percentage points.
        q05 = float(v.quantile(0.05))
        q95 = float(v.quantile(0.95))
        if abs(q05) <= 1.5 and abs(q95) <= 1.5:
            out[c] = s * 100.0
            scaled.append(c)
    return out, sorted(set(scaled))


def _apply_log1p_nonnegative_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    applied: list[str] = []
    for c in columns:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = np.log1p(s.clip(lower=0.0))
        applied.append(c)
    return out, sorted(set(applied))


def _build_robust_clip_bounds(
    df: pd.DataFrame,
    columns: list[str],
    train_mask: np.ndarray,
    lower_q: float,
    upper_q: float,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    tr = df.loc[train_mask]
    for c in columns:
        if c not in tr.columns:
            continue
        s = pd.to_numeric(tr[c], errors="coerce").dropna()
        if s.empty:
            continue
        lo = float(s.quantile(lower_q))
        hi = float(s.quantile(upper_q))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds


def _apply_robust_clip_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    applied: list[str] = []
    for c, (lo, hi) in bounds.items():
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.clip(lower=float(lo), upper=float(hi))
        applied.append(c)
    return out, sorted(set(applied))


def _add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "year"], kind="mergesort").reset_index(drop=True)
    by_sym = out.groupby("symbol", observed=False)

    # Rebuild lag/time-series features from the symbol timeline to avoid using corrupted lag columns.
    if "revenue_current" in out.columns:
        out["revenue_current"] = pd.to_numeric(out["revenue_current"], errors="coerce")
        out["revenue_lag1"] = by_sym["revenue_current"].shift(1)
        out["revenue_lag2"] = by_sym["revenue_current"].shift(2)
        out["revenue_roll3_mean"] = by_sym["revenue_current"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out["revenue_roll3_std"] = (
            by_sym["revenue_current"].transform(lambda s: s.rolling(3, min_periods=1).std(ddof=0)).fillna(0.0)
        )
        out["revenue_yoy_pct"] = np.where(
            out["revenue_lag1"].fillna(0.0) != 0.0,
            ((out["revenue_current"] / out["revenue_lag1"]) - 1.0) * 100.0,
            0.0,
        )
        out["revenue_momentum_pct"] = np.where(
            out["revenue_lag1"].fillna(0.0) != 0.0,
            ((out["revenue_current"] - out["revenue_lag1"]) / out["revenue_lag1"].abs()) * 100.0,
            0.0,
        )

    if "profit_after_tax" in out.columns:
        out["profit_after_tax"] = pd.to_numeric(out["profit_after_tax"], errors="coerce")
        out["profit_lag1"] = by_sym["profit_after_tax"].shift(1)
        out["profit_lag2"] = by_sym["profit_after_tax"].shift(2)
        out["profit_roll3_mean"] = by_sym["profit_after_tax"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out["profit_roll3_std"] = (
            by_sym["profit_after_tax"].transform(lambda s: s.rolling(3, min_periods=1).std(ddof=0)).fillna(0.0)
        )
        out["profit_yoy_pct"] = np.where(
            out["profit_lag1"].fillna(0.0) != 0.0,
            ((out["profit_after_tax"] / out["profit_lag1"]) - 1.0) * 100.0,
            0.0,
        )
        out["profit_momentum_pct"] = np.where(
            out["profit_lag1"].fillna(0.0) != 0.0,
            ((out["profit_after_tax"] - out["profit_lag1"]) / out["profit_lag1"].abs()) * 100.0,
            0.0,
        )

    if {"profit_after_tax", "revenue_current"}.issubset(out.columns):
        out["profit_margin_calc_pct"] = np.where(
            out["revenue_current"].fillna(0.0) != 0.0,
            (out["profit_after_tax"] / out["revenue_current"]) * 100.0,
            0.0,
        )

    if "net_debt_to_equity_pct" in out.columns:
        out["net_debt_to_equity_pct"] = pd.to_numeric(out["net_debt_to_equity_pct"], errors="coerce")
        out["net_debt_to_equity_pct_lag1"] = by_sym["net_debt_to_equity_pct"].shift(1)

    out["asset_growth_yoy"] = (
        by_sym["total_assets_reported"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
        if "total_assets_reported" in out.columns
        else np.nan
    )
    out["profit_margin_change"] = (
        by_sym["profit_margin_calc_pct"].diff() if "profit_margin_calc_pct" in out.columns else np.nan
    )
    if {"revenue_current", "revenue_lag1"}.issubset(out.columns):
        out["revenue_momentum_delta"] = out["revenue_current"] - out["revenue_lag1"]
    else:
        out["revenue_momentum_delta"] = np.nan
    for c in ("asset_growth_yoy", "profit_margin_change", "revenue_momentum_delta"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _build_debt_interest_params(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    interest_col: str,
    debt_col: str,
    debt_lag_col: str,
    lower_q: float,
    upper_q: float,
    high_q: float,
    boost: float,
) -> dict[str, float | str]:
    if interest_col not in df.columns:
        return {}
    if debt_col not in df.columns and debt_lag_col not in df.columns:
        return {}

    debt_cur = pd.to_numeric(df[debt_col], errors="coerce") if debt_col in df.columns else pd.Series(np.nan, index=df.index)
    debt_lag = (
        pd.to_numeric(df[debt_lag_col], errors="coerce")
        if debt_lag_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    debt_src = debt_lag.where(debt_lag.notna(), debt_cur)
    v = debt_src.loc[train_mask].dropna()
    if v.empty:
        return {}

    lo = float(v.quantile(lower_q))
    hi = float(v.quantile(upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return {}
    if lo > hi:
        lo, hi = hi, lo

    v_clip = v.clip(lower=lo, upper=hi)
    thr = float(v_clip.quantile(high_q))
    if not np.isfinite(thr):
        return {}

    return {
        "interest_col": str(interest_col),
        "debt_col": str(debt_col),
        "debt_lag_col": str(debt_lag_col),
        "lower": float(lo),
        "upper": float(hi),
        "threshold": float(thr),
        "boost": float(boost),
    }


def _apply_debt_interest_adjustment_df(
    df: pd.DataFrame,
    params: dict[str, float | str],
) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    if not params:
        return out, False

    interest_col = str(params.get("interest_col", ""))
    debt_col = str(params.get("debt_col", ""))
    debt_lag_col = str(params.get("debt_lag_col", ""))
    lo = float(params.get("lower", np.nan))
    hi = float(params.get("upper", np.nan))
    thr = float(params.get("threshold", np.nan))
    boost = float(params.get("boost", 1.0))

    if interest_col not in out.columns:
        return out, False
    if not np.isfinite(lo) or not np.isfinite(hi) or not np.isfinite(thr):
        return out, False
    if lo > hi:
        lo, hi = hi, lo

    debt_cur = pd.to_numeric(out[debt_col], errors="coerce") if debt_col in out.columns else pd.Series(np.nan, index=out.index)
    debt_lag = (
        pd.to_numeric(out[debt_lag_col], errors="coerce")
        if debt_lag_col in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    debt_src = debt_lag.where(debt_lag.notna(), debt_cur).clip(lower=lo, upper=hi)
    mult = np.where(debt_src.notna() & (debt_src >= thr), boost, 1.0)
    out[interest_col] = pd.to_numeric(out[interest_col], errors="coerce").fillna(0.0) * mult
    out["debt_interest_multiplier"] = mult
    return out, True


def _add_steel_features(df: pd.DataFrame, boost: float) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    steel_slug = "KIM_LO_I_C_NG_NGHI_P_KHAI_KHO_NG"
    hpg = out[out["symbol"].astype(str).str.upper() == "HPG"]
    if not hpg.empty:
        steel_slug = str(hpg.iloc[0]["industry_group"])
    out["ind_steel"] = (out["industry_group"].astype(str) == steel_slug).astype(float)
    if {"hrc_log", "iron_ore_log"}.issubset(out.columns):
        out["spread_hrc_iron"] = out["hrc_log"] - out["iron_ore_log"]
    else:
        out["spread_hrc_iron"] = 0.0
    out["ix_spread_hrc_iron_steel"] = out["spread_hrc_iron"] * out["ind_steel"] * float(boost)
    return out, steel_slug


def _add_sector_macro_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    if "spread_hrc_iron" not in out.columns and {"hrc_log", "iron_ore_log"}.issubset(out.columns):
        out["spread_hrc_iron"] = pd.to_numeric(out["hrc_log"], errors="coerce").fillna(0.0) - pd.to_numeric(
            out["iron_ore_log"], errors="coerce"
        ).fillna(0.0)

    created: list[str] = []
    group = out["industry_group"].astype(str)
    for sector_slug, cfg in SECTOR_MACRO_MAPPINGS.items():
        mask = (group == str(sector_slug)).astype(float)
        for macro in cfg.get("specific_macros", []):
            if macro not in out.columns:
                continue
            vals = pd.to_numeric(out[macro], errors="coerce")
            col = f"ix_sector_{sector_slug}_{macro}"
            out[col] = np.where(mask == 1.0, vals, 0.0)
            created.append(col)
    return out, sorted(set(created))


def _clean_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            med = float(X[c].median()) if X[c].notna().any() else 0.0
            X[c] = X[c].fillna(med)
    return X


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float(default)


def _macro_lookup(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    m = df.copy()
    m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["year"]).astype({"year": int})
    m = m.sort_values("year", kind="mergesort").reset_index(drop=True)
    cols = [c for c in m.columns if c != "year"]
    return m, cols


def _macro_row_for_year(macro_df: pd.DataFrame, year: int) -> pd.Series:
    exact = macro_df[macro_df["year"] == int(year)]
    if not exact.empty:
        return exact.iloc[-1]
    prev = macro_df[macro_df["year"] <= int(year)]
    if not prev.empty:
        return prev.iloc[-1]
    return macro_df.iloc[0]


def _apply_macro_state(state: dict[str, Any], macro_row: pd.Series, macro_cols: list[str]) -> None:
    for c in macro_cols:
        if c in macro_row.index:
            state[c] = _to_float(macro_row[c], state.get(c, 0.0))


def _refresh_sector_macro_interactions_state(state: dict[str, Any]) -> None:
    hrc = _to_float(state.get("hrc_log", 0.0), 0.0)
    iron = _to_float(state.get("iron_ore_log", 0.0), 0.0)
    if "spread_hrc_iron" in state or ("hrc_log" in state and "iron_ore_log" in state):
        state["spread_hrc_iron"] = hrc - iron

    sector = str(state.get("industry_group", ""))
    for sector_slug, cfg in SECTOR_MACRO_MAPPINGS.items():
        for macro in cfg.get("specific_macros", []):
            macro_val = _to_float(state.get(macro, 0.0), 0.0)
            if macro == "spread_hrc_iron":
                macro_val = _to_float(state.get("spread_hrc_iron", hrc - iron), 0.0)
            ix_col = f"ix_sector_{sector_slug}_{macro}"
            state[ix_col] = macro_val if sector == str(sector_slug) else 0.0


def _apply_debt_interest_adjustment_state(state: dict[str, Any], params: dict[str, float | str] | None) -> None:
    if not isinstance(params, dict) or not params:
        return

    interest_col = str(params.get("interest_col", ""))
    debt_col = str(params.get("debt_col", ""))
    debt_lag_col = str(params.get("debt_lag_col", ""))
    lo = _to_float(params.get("lower", np.nan), np.nan)
    hi = _to_float(params.get("upper", np.nan), np.nan)
    thr = _to_float(params.get("threshold", np.nan), np.nan)
    boost = _to_float(params.get("boost", 1.0), 1.0)

    if not interest_col:
        return
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(thr)):
        return
    if lo > hi:
        lo, hi = hi, lo

    debt_lag = _to_float(state.get(debt_lag_col, np.nan), np.nan)
    debt_cur = _to_float(state.get(debt_col, np.nan), np.nan)
    debt_src = debt_lag if np.isfinite(debt_lag) else debt_cur
    if np.isfinite(debt_src):
        debt_src = float(np.clip(debt_src, lo, hi))

    mult = boost if (np.isfinite(debt_src) and debt_src >= thr) else 1.0
    cur_interest = _to_float(state.get(interest_col, 0.0), 0.0)
    state[interest_col] = cur_interest * float(mult)
    state["debt_interest_multiplier"] = float(mult)


def _recompute_derived_state(state: dict[str, Any], steel_boost: float) -> None:
    rev = _to_float(state.get("revenue_current", 0.0), 0.0)
    rev_l1 = _to_float(state.get("revenue_lag1", 0.0), 0.0)
    rev_l2 = _to_float(state.get("revenue_lag2", 0.0), 0.0)
    prof = _to_float(state.get("profit_after_tax", 0.0), 0.0)
    prof_l1 = _to_float(state.get("profit_lag1", 0.0), 0.0)
    prof_l2 = _to_float(state.get("profit_lag2", 0.0), 0.0)

    if "total_revenue" in state:
        state["total_revenue"] = rev
    if "net_revenue" in state:
        state["net_revenue"] = rev

    margin_old = _to_float(state.get("profit_margin_calc_pct", np.nan), np.nan)
    if rev != 0:
        margin_new = (prof / rev) * 100.0
    else:
        margin_new = 0.0
    state["profit_margin_calc_pct"] = margin_new

    if np.isfinite(margin_old):
        state["profit_margin_change"] = margin_new - margin_old
    else:
        state["profit_margin_change"] = 0.0

    state["revenue_momentum_delta"] = rev - rev_l1
    state["revenue_yoy_pct"] = ((rev / rev_l1) - 1.0) * 100.0 if rev_l1 != 0 else 0.0
    state["profit_yoy_pct"] = ((prof / prof_l1) - 1.0) * 100.0 if prof_l1 != 0 else 0.0
    state["revenue_momentum_pct"] = ((rev - rev_l1) / abs(rev_l1)) * 100.0 if rev_l1 != 0 else 0.0
    state["profit_momentum_pct"] = ((prof - prof_l1) / abs(prof_l1)) * 100.0 if prof_l1 != 0 else 0.0

    rev_roll = [rev, rev_l1, rev_l2]
    prof_roll = [prof, prof_l1, prof_l2]
    state["revenue_roll3_mean"] = float(np.mean(rev_roll))
    state["revenue_roll3_std"] = float(np.std(rev_roll))
    state["profit_roll3_mean"] = float(np.mean(prof_roll))
    state["profit_roll3_std"] = float(np.std(prof_roll))

    eq = _to_float(state.get("equity", 0.0), 0.0)
    ta = _to_float(state.get("total_assets_reported", state.get("total_assets", 0.0)), 0.0)
    net_debt = _to_float(state.get("net_debt", np.nan), np.nan)

    roe_existing = _to_float(state.get("roe", np.nan), np.nan)
    roa_existing = _to_float(state.get("roa", np.nan), np.nan)
    dte_existing = _to_float(state.get("net_debt_to_equity_pct", np.nan), np.nan)

    if not np.isfinite(roe_existing) and eq != 0:
        state["roe"] = (prof / eq) * 100.0
    if not np.isfinite(roa_existing) and ta != 0:
        state["roa"] = (prof / ta) * 100.0
    if not np.isfinite(dte_existing) and eq != 0 and np.isfinite(net_debt):
        state["net_debt_to_equity_pct"] = (net_debt / eq) * 100.0

    if rev != 0:
        state["net_margin"] = (prof / rev) * 100.0

    hrc = _to_float(state.get("hrc_log", 0.0), 0.0)
    iron = _to_float(state.get("iron_ore_log", 0.0), 0.0)
    ind_steel = _to_float(state.get("ind_steel", 0.0), 0.0)
    state["spread_hrc_iron"] = hrc - iron
    state["ix_spread_hrc_iron_steel"] = state["spread_hrc_iron"] * ind_steel * float(steel_boost)


def _build_feature_frame_from_state(bundle: dict[str, Any], state: dict[str, Any]) -> pd.DataFrame:
    cols = list(bundle["feature_columns"])
    row = {c: state.get(c, 0.0) for c in cols}
    X = pd.DataFrame([row])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    clip_bounds = bundle.get("clip_bounds")
    if isinstance(clip_bounds, dict):
        for c, bound in clip_bounds.items():
            if c not in X.columns:
                continue
            if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                continue
            lo = _to_float(bound[0], np.nan)
            hi = _to_float(bound[1], np.nan)
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if lo > hi:
                lo, hi = hi, lo
            X[c] = pd.to_numeric(X[c], errors="coerce").clip(lower=float(lo), upper=float(hi)).fillna(0.0)
    return X[cols]


def _predict_next_from_state(bundle: dict[str, Any], state: dict[str, Any], current_col: str) -> float:
    model = bundle["model"]
    X = _build_feature_frame_from_state(bundle, state)
    pred_t = float(model.predict(X)[0])
    pred_t = float(np.clip(pred_t, -3.0, 3.0))
    cur = _to_float(state.get(current_col, 0.0), 0.0)
    growth = float(np.sign(pred_t) * np.expm1(abs(pred_t)))
    return float(cur * (1.0 + growth))


def _target_year(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["year"], errors="coerce") + 1


def _recency_sample_weights(
    target_year: pd.Series,
    train_mask: np.ndarray,
    mode: str,
    min_weight: float,
    exp_base: float,
) -> np.ndarray:
    train_year = pd.to_numeric(target_year[train_mask], errors="coerce").to_numpy(dtype=float)
    if train_year.size == 0 or mode == "none":
        return np.ones(train_year.shape[0], dtype=float)

    min_w = float(np.clip(min_weight, 0.0, 1.0))
    if mode == "linear":
        lo = float(np.nanmin(train_year))
        hi = float(np.nanmax(train_year))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return np.ones(train_year.shape[0], dtype=float)
        rel = (train_year - lo) / span
        rel = np.clip(rel, 0.0, 1.0)
        return min_w + (1.0 - min_w) * rel

    base = float(np.clip(exp_base, 1e-6, 0.999999))
    latest = float(np.nanmax(train_year))
    age = np.clip(latest - train_year, 0.0, None)
    raw = np.power(base, age)
    return min_w + (1.0 - min_w) * raw


def _prepare_signed_log_growth(
    df: pd.DataFrame,
    target_col: str,
    current_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    y_next = pd.to_numeric(df[target_col], errors="coerce")
    y_cur = pd.to_numeric(df[current_col], errors="coerce")
    valid = y_next.notna() & y_cur.notna() & (y_cur != 0)
    growth = (y_next[valid] / y_cur[valid]) - 1.0
    y_t = np.sign(growth) * np.log1p(np.abs(growth))
    return y_t, y_next, y_cur, valid


def _inverse_signed_log_growth(pred_t: np.ndarray, cur: np.ndarray) -> np.ndarray:
    pred_t = np.clip(pred_t.astype(float), -3.0, 3.0)
    growth = np.sign(pred_t) * np.expm1(np.abs(pred_t))
    pred = cur.astype(float) * (1.0 + growth)
    return np.nan_to_num(pred, nan=0.0, posinf=np.finfo(np.float64).max, neginf=-np.finfo(np.float64).max)


def _select_with_rfe(
    df: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    candidates: list[str],
    max_features: int,
    mandatory: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> list[str]:
    mandatory = [c for c in (mandatory or []) if c in candidates]
    rem = [c for c in candidates if c not in mandatory]
    k = max(0, min(max_features - len(mandatory), len(rem)))
    if k == 0:
        return mandatory
    X_tr = _clean_X(df.loc[train_mask], rem)
    y_tr = pd.to_numeric(y.loc[train_mask], errors="coerce")
    good = y_tr.notna()
    X_tr = X_tr.loc[good].reset_index(drop=True)
    y_tr = y_tr.loc[good].reset_index(drop=True)
    sw_tr = None
    if sample_weight is not None:
        sw_raw = np.asarray(sample_weight, dtype=float)
        if sw_raw.shape[0] == len(df):
            sw_tr = sw_raw[train_mask]
        elif sw_raw.shape[0] == int(np.sum(train_mask)):
            sw_tr = sw_raw
        else:
            raise ValueError(
                f"sample_weight length mismatch: got {sw_raw.shape[0]}, expected {len(df)} or {int(np.sum(train_mask))}"
            )
        sw_tr = sw_tr[good.to_numpy()]

    if len(rem) <= k:
        return mandatory + rem
    est = XGBRegressor(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    rfe = RFE(estimator=est, n_features_to_select=k, step=0.15)
    if sw_tr is not None:
        rfe.fit(X_tr, y_tr, sample_weight=sw_tr)
    else:
        rfe.fit(X_tr, y_tr)
    sel = [c for c, keep in zip(rem, rfe.support_) if keep]
    return mandatory + sel


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    nz = y_true != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - y_mean)))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _parse_symbols(raw: str) -> list[str]:
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return list(dict.fromkeys(out))


def _fmt_num(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_signed(x: float) -> str:
    return f"{x:+,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single final pipeline: train selected signed-log-growth model and export prediction report."
    )
    parser.add_argument("--train-target-year-max", type=int, default=2025)
    parser.add_argument("--predict-target-year", type=int, default=2026)
    parser.add_argument(
        "--symbols",
        type=str,
        default="ACB,VEA,NLG,DGC,PNJ,MWG,VIB,VPB",
        help="Comma-separated symbols for final report.",
    )
    parser.add_argument(
        "--bank-preprocessed-csv",
        type=Path,
        default=ROOT / "artifacts" / "preprocessed" / "bank_annual.csv",
    )
    parser.add_argument(
        "--nonbank-preprocessed-csv",
        type=Path,
        default=ROOT / "artifacts" / "preprocessed" / "nonbank_annual.csv",
    )
    parser.add_argument(
        "--bank-macro-csv",
        type=Path,
        default=ROOT / "artifacts" / "macro" / "macro_yearly_train_bank.csv",
    )
    parser.add_argument(
        "--nonbank-macro-csv",
        type=Path,
        default=ROOT / "artifacts" / "macro" / "macro_yearly_train.csv",
    )
    parser.add_argument(
        "--industry-map-csv",
        type=Path,
        default=ROOT / "artifacts" / "financial_training" / "symbol_industry_map_l3.csv",
    )
    parser.add_argument("--macro-lag-years", type=int, default=1)
    parser.add_argument("--nonbank-feature-budget", type=int, default=50)
    parser.add_argument("--steel-boost", type=float, default=1.0)
    parser.add_argument(
        "--include-year-feature",
        dest="include_year_feature",
        action="store_true",
        default=False,
        help="Include raw year as an input feature (default: excluded).",
    )
    parser.add_argument(
        "--exclude-year-feature",
        dest="include_year_feature",
        action="store_false",
        help="Exclude raw year from input features.",
    )
    parser.add_argument(
        "--recency-weight-mode",
        type=str,
        choices=["none", "linear", "exp"],
        default="exp",
        help="Weight newer training years more heavily (none/linear/exp).",
    )
    parser.add_argument(
        "--recency-weight-min",
        type=float,
        default=0.35,
        help="Minimum weight for the oldest training years (0..1).",
    )
    parser.add_argument(
        "--recency-weight-exp-base",
        type=float,
        default=0.8,
        help="Exponential decay base for recency weighting (used when mode=exp).",
    )
    parser.add_argument(
        "--enable-robust-clip",
        dest="enable_robust_clip",
        action="store_true",
        default=True,
        help="Clip extreme ratio/percent-like values using train-only quantile bounds.",
    )
    parser.add_argument(
        "--disable-robust-clip",
        dest="enable_robust_clip",
        action="store_false",
        help="Disable robust quantile clipping for ratio/percent-like columns.",
    )
    parser.add_argument(
        "--robust-clip-lower-q",
        type=float,
        default=0.01,
        help="Lower quantile for robust clipping (train-only).",
    )
    parser.add_argument(
        "--robust-clip-upper-q",
        type=float,
        default=0.99,
        help="Upper quantile for robust clipping (train-only).",
    )
    parser.add_argument(
        "--enable-debt-interest-adjustment",
        dest="enable_debt_interest_adjustment",
        action="store_true",
        default=True,
        help="Boost interest_loan_midlong_pct by debt sensitivity rule for high-leverage firms.",
    )
    parser.add_argument(
        "--disable-debt-interest-adjustment",
        dest="enable_debt_interest_adjustment",
        action="store_false",
        help="Disable debt-sensitive interest adjustment.",
    )
    parser.add_argument(
        "--debt-interest-boost",
        type=float,
        default=1.5,
        help="Multiplier applied to lending-rate feature for high-leverage firms.",
    )
    parser.add_argument(
        "--debt-interest-high-q",
        type=float,
        default=0.8,
        help="Quantile threshold defining high leverage (train-only).",
    )
    parser.add_argument(
        "--debt-interest-lower-q",
        type=float,
        default=0.01,
        help="Lower quantile clip for leverage ratio before thresholding.",
    )
    parser.add_argument(
        "--debt-interest-upper-q",
        type=float,
        default=0.99,
        help="Upper quantile clip for leverage ratio before thresholding.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "models" / "final_model_pipeline",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("--symbols must not be empty")
    if args.enable_robust_clip:
        if not (0.0 <= float(args.robust_clip_lower_q) < float(args.robust_clip_upper_q) <= 1.0):
            raise ValueError("Robust clip quantiles must satisfy 0 <= lower < upper <= 1")
    if args.enable_debt_interest_adjustment:
        if not (0.0 <= float(args.debt_interest_lower_q) < float(args.debt_interest_upper_q) <= 1.0):
            raise ValueError("Debt-interest clip quantiles must satisfy 0 <= lower < upper <= 1")
        if not (0.0 < float(args.debt_interest_high_q) < 1.0):
            raise ValueError("Debt-interest high quantile must be in (0, 1)")

    bank = pd.read_csv(args.bank_preprocessed_csv)
    nonbank = pd.read_csv(args.nonbank_preprocessed_csv)
    if "quarter" in bank.columns:
        bank = bank[bank["quarter"] == 0].copy().reset_index(drop=True)
    if "quarter" in nonbank.columns:
        nonbank = nonbank[nonbank["quarter"] == 0].copy().reset_index(drop=True)

    bank, dropped_bank_ids = _drop_identifier_columns(bank)
    nonbank, dropped_nonbank_ids = _drop_identifier_columns(nonbank)

    bank = _merge_macro(bank, args.bank_macro_csv, lag_years=args.macro_lag_years)
    nonbank = _merge_macro(nonbank, args.nonbank_macro_csv, lag_years=args.macro_lag_years)
    bank, logged_bank_macros = _apply_log1p_nonnegative_columns(bank, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
    nonbank, logged_nonbank_macros = _apply_log1p_nonnegative_columns(nonbank, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
    nonbank = _merge_industry(nonbank, args.industry_map_csv)
    nonbank = _add_delta_features(nonbank)

    bank, scaled_bank_pct = _normalize_percent_like_columns(bank, PERCENT_LIKE_COLUMNS)
    nonbank, scaled_nonbank_pct = _normalize_percent_like_columns(nonbank, PERCENT_LIKE_COLUMNS)

    nonbank, steel_slug = _add_steel_features(nonbank, boost=args.steel_boost)
    dummies = pd.get_dummies(nonbank["industry_group"], prefix="ind", dtype=float)
    nonbank = pd.concat([nonbank, dummies], axis=1)
    nonbank, sector_macro_ix_features = _add_sector_macro_interactions(nonbank)

    dropped_id_cols = sorted(set(dropped_bank_ids + dropped_nonbank_ids))
    scaled_pct_cols = sorted(set(scaled_bank_pct + scaled_nonbank_pct))
    logged_macro_cols = sorted(set(logged_bank_macros + logged_nonbank_macros))
    robust_clip_columns = list(dict.fromkeys(PERCENT_LIKE_COLUMNS + EXTRA_ROBUST_CLIP_COLUMNS))

    bank_features = [
        "revenue_current",
        "profit_after_tax",
        "net_interest_income",
        "fee_and_commission_income",
        "other_income",
        "nim_pct",
        "assets_to_equity",
        "roe",
        "roa",
        "equity",
        "total_assets",
        "total_liabilities",
        "customer_loans",
        "customer_deposits",
        "interbank_placements",
        "deposits_at_sbv",
        "gdp_ty_dong_log",
        "usd_vnd_log",
        "interest_deposit_12m_pct",
        "interest_loan_short_pct",
        "interest_loan_midlong_pct",
        "cpi_inflation_yoy_pp",
    ]
    if args.include_year_feature:
        bank_features = ["year"] + bank_features
    bank_features = [c for c in bank_features if c in bank.columns]

    nonbank_common_macros = [c for c in COMMON_MACROS if c in nonbank.columns]

    nonbank_candidates = [
        "revenue_current",
        "profit_after_tax",
        "total_assets_reported",
        "equity",
        "net_debt",
        "net_debt_to_equity_pct",
        "receivable_ratio_pct",
        "gross_margin",
        "net_margin",
        "profit_margin_calc_pct",
        "revenue_yoy_pct",
        "profit_yoy_pct",
        "revenue_lag1",
        "revenue_lag2",
        "profit_lag1",
        "profit_lag2",
        "revenue_roll3_mean",
        "profit_roll3_mean",
        "revenue_roll3_std",
        "profit_roll3_std",
        "revenue_momentum_pct",
        "profit_momentum_pct",
        "short_term_borrowings",
        "long_term_borrowings",
        "total_revenue",
        "net_revenue",
        "roa",
        "roe",
        "asset_growth_yoy",
        "profit_margin_change",
        "revenue_momentum_delta",
    ]
    if args.include_year_feature:
        nonbank_candidates = ["year"] + nonbank_candidates
    nonbank_candidates += nonbank_common_macros
    nonbank_candidates += sector_macro_ix_features
    nonbank_candidates += [c for c in nonbank.columns if c.startswith("ind_")]
    nonbank_candidates += ["net_debt_to_equity_pct_lag1"]
    nonbank_candidates = [c for c in nonbank_candidates if c in nonbank.columns]
    nonbank_candidates = list(dict.fromkeys(nonbank_candidates))
    mandatory_nonbank = [c for c in ["revenue_current", "profit_after_tax"] if c in nonbank_candidates]
    robust_clip_columns_used: set[str] = set()

    detail_rows: list[dict[str, Any]] = []
    model_registry: dict[str, dict[str, Any]] = {}

    for target, cur_col, obj in [
        ("revenue_next", "revenue_current", "reg:squarederror"),
        ("profit_after_tax_next", "profit_after_tax", "reg:absoluteerror"),
    ]:
        d = bank[bank[target].notna()].copy().reset_index(drop=True)
        y_t, y_abs, y_cur, keep = _prepare_signed_log_growth(d, target, cur_col)
        d = d.loc[keep].reset_index(drop=True)
        y_t = y_t.reset_index(drop=True)
        y_abs = y_abs.loc[keep].reset_index(drop=True)
        y_cur = y_cur.loc[keep].reset_index(drop=True)

        target_year = _target_year(d)
        tr = (target_year <= args.train_target_year_max).to_numpy()
        te = (target_year == args.predict_target_year).to_numpy()
        w_tr = _recency_sample_weights(
            target_year=target_year,
            train_mask=tr,
            mode=args.recency_weight_mode,
            min_weight=args.recency_weight_min,
            exp_base=args.recency_weight_exp_base,
        )

        clip_bounds: dict[str, tuple[float, float]] = {}
        d_model = d
        if args.enable_robust_clip:
            clip_bounds = _build_robust_clip_bounds(
                d,
                robust_clip_columns,
                tr,
                float(args.robust_clip_lower_q),
                float(args.robust_clip_upper_q),
            )
            d_model, _ = _apply_robust_clip_bounds(d, clip_bounds)
            robust_clip_columns_used.update(clip_bounds.keys())

        X = _clean_X(d_model, bank_features)
        m = _model(obj)
        m.fit(X.loc[tr], y_t.loc[tr], sample_weight=w_tr)
        p_t = m.predict(X.loc[te])
        p_abs = _inverse_signed_log_growth(p_t, y_cur.loc[te].to_numpy(dtype=float))

        key = f"bank_{target}"
        model_registry[key] = {
            "model": m,
            "feature_columns": bank_features,
            "clip_bounds": clip_bounds,
            "debt_interest_params": {},
        }

        dv = d.loc[te].reset_index(drop=True)
        ya = y_abs.loc[te].reset_index(drop=True)
        for i, pred in enumerate(p_abs):
            detail_rows.append(
                {
                    "dataset": "bank",
                    "target": target,
                    "symbol": str(dv.iloc[i]["symbol"]).upper(),
                    "feature_year": int(dv.iloc[i]["year"]),
                    "target_year": int(dv.iloc[i]["year"] + 1),
                    "actual": float(ya.iloc[i]),
                    "predicted": float(pred),
                }
            )

    nonbank_base_target_year = _target_year(nonbank)
    nonbank_base_train_mask = (nonbank_base_target_year <= args.train_target_year_max).to_numpy()
    debt_interest_params: dict[str, float | str] = {}
    if args.enable_debt_interest_adjustment:
        debt_interest_params = _build_debt_interest_params(
            nonbank,
            nonbank_base_train_mask,
            interest_col=str(DEBT_INTEREST_ADJUSTMENT["interest_col"]),
            debt_col=str(DEBT_INTEREST_ADJUSTMENT["debt_col"]),
            debt_lag_col=str(DEBT_INTEREST_ADJUSTMENT["debt_lag_col"]),
            lower_q=float(args.debt_interest_lower_q),
            upper_q=float(args.debt_interest_upper_q),
            high_q=float(args.debt_interest_high_q),
            boost=float(args.debt_interest_boost),
        )

    debt_interest_applied = False

    for target, cur_col, obj in [
        ("revenue_next", "revenue_current", "reg:squarederror"),
        ("profit_after_tax_next", "profit_after_tax", "reg:absoluteerror"),
    ]:
        d = nonbank[nonbank[target].notna()].copy().reset_index(drop=True)
        y_t, y_abs, y_cur, keep = _prepare_signed_log_growth(d, target, cur_col)
        d = d.loc[keep].reset_index(drop=True)
        y_t = y_t.reset_index(drop=True)
        y_abs = y_abs.loc[keep].reset_index(drop=True)
        y_cur = y_cur.loc[keep].reset_index(drop=True)

        target_year = _target_year(d)
        tr = (target_year <= args.train_target_year_max).to_numpy()
        te = (target_year == args.predict_target_year).to_numpy()
        w_tr = _recency_sample_weights(
            target_year=target_year,
            train_mask=tr,
            mode=args.recency_weight_mode,
            min_weight=args.recency_weight_min,
            exp_base=args.recency_weight_exp_base,
        )

        clip_bounds: dict[str, tuple[float, float]] = {}
        d_model = d
        if args.enable_robust_clip:
            clip_bounds = _build_robust_clip_bounds(
                d,
                robust_clip_columns,
                tr,
                float(args.robust_clip_lower_q),
                float(args.robust_clip_upper_q),
            )
            d_model, _ = _apply_robust_clip_bounds(d, clip_bounds)
            robust_clip_columns_used.update(clip_bounds.keys())

        if debt_interest_params:
            d_model, applied = _apply_debt_interest_adjustment_df(d_model, debt_interest_params)
            debt_interest_applied = debt_interest_applied or applied

        feats = _select_with_rfe(
            d_model,
            y_t,
            tr,
            nonbank_candidates,
            max_features=args.nonbank_feature_budget,
            mandatory=mandatory_nonbank,
            sample_weight=w_tr,
        )
        X = _clean_X(d_model, feats)
        m = _model(obj)
        m.fit(X.loc[tr], y_t.loc[tr], sample_weight=w_tr)
        p_t = m.predict(X.loc[te])
        p_abs = _inverse_signed_log_growth(p_t, y_cur.loc[te].to_numpy(dtype=float))

        key = f"nonbank_{target}"
        model_registry[key] = {
            "model": m,
            "feature_columns": feats,
            "clip_bounds": clip_bounds,
            "debt_interest_params": debt_interest_params,
        }

        dv = d.loc[te].reset_index(drop=True)
        ya = y_abs.loc[te].reset_index(drop=True)
        for i, pred in enumerate(p_abs):
            detail_rows.append(
                {
                    "dataset": "nonbank",
                    "target": target,
                    "symbol": str(dv.iloc[i]["symbol"]).upper(),
                    "feature_year": int(dv.iloc[i]["year"]),
                    "target_year": int(dv.iloc[i]["year"] + 1),
                    "actual": float(ya.iloc[i]),
                    "predicted": float(pred),
                }
            )

    detail_columns = [
        "dataset",
        "target",
        "symbol",
        "feature_year",
        "target_year",
        "actual",
        "predicted",
    ]
    detail_df = pd.DataFrame(detail_rows, columns=detail_columns)
    if not detail_df.empty:
        detail_df = detail_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual", "predicted"]).reset_index(drop=True)

    detail_csv = out_dir / "predict_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    model_files: dict[str, str] = {}
    for name, bundle in model_registry.items():
        model_path = out_dir / f"{name}.joblib"
        joblib.dump(bundle, model_path)
        model_files[name] = str(model_path)

    bank_symbol_set = set(bank["symbol"].astype(str).str.upper())
    nonbank_symbol_set = set(nonbank["symbol"].astype(str).str.upper())
    bank_macro_src = pd.read_csv(args.bank_macro_csv).assign(year=lambda x: x["year"] + int(args.macro_lag_years))
    bank_macro_src, _ = _apply_log1p_nonnegative_columns(bank_macro_src, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
    bank_macro_df, bank_macro_cols = _macro_lookup(bank_macro_src)
    nonbank_macro_src = pd.read_csv(args.nonbank_macro_csv).assign(year=lambda x: x["year"] + int(args.macro_lag_years))
    nonbank_macro_src, _ = _apply_log1p_nonnegative_columns(nonbank_macro_src, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
    nonbank_macro_df, nonbank_macro_cols = _macro_lookup(nonbank_macro_src)

    report_rows = []
    missing = []
    feature_year_for_report = int(args.predict_target_year) - 1
    for s in symbols:
        if s in bank_symbol_set:
            ds = "bank"
            ds_df = bank
            macro_df = bank_macro_df
            macro_cols = bank_macro_cols
            rev_key = "bank_revenue_next"
            pf_key = "bank_profit_after_tax_next"
            ds_debt_params: dict[str, float | str] | None = None
        elif s in nonbank_symbol_set:
            ds = "nonbank"
            ds_df = nonbank
            macro_df = nonbank_macro_df
            macro_cols = nonbank_macro_cols
            rev_key = "nonbank_revenue_next"
            pf_key = "nonbank_profit_after_tax_next"
            ds_debt_params = debt_interest_params if debt_interest_params else None
        else:
            missing.append(s)
            continue

        base_rows = ds_df[
            (ds_df["symbol"].astype(str).str.upper() == s)
            & (pd.to_numeric(ds_df["year"], errors="coerce") == feature_year_for_report)
        ]
        actual_rows = ds_df[
            (ds_df["symbol"].astype(str).str.upper() == s)
            & (pd.to_numeric(ds_df["year"], errors="coerce") == int(args.predict_target_year))
        ]
        if base_rows.empty or actual_rows.empty:
            missing.append(s)
            continue

        state = base_rows.iloc[-1].to_dict()
        state["symbol"] = s
        state["year"] = feature_year_for_report

        macro_row = _macro_row_for_year(macro_df, feature_year_for_report)
        _apply_macro_state(state, macro_row, macro_cols)
        if ds == "nonbank":
            _apply_debt_interest_adjustment_state(state, ds_debt_params)
            _recompute_derived_state(state, args.steel_boost)
            _refresh_sector_macro_interactions_state(state)

        rev_bundle = model_registry.get(rev_key)
        pf_bundle = model_registry.get(pf_key)
        if rev_bundle is None or pf_bundle is None:
            missing.append(s)
            continue

        rev_pred = _predict_next_from_state(rev_bundle, state, current_col="revenue_current")
        pf_pred = _predict_next_from_state(pf_bundle, state, current_col="profit_after_tax")
        actual_row = actual_rows.iloc[-1]
        rev_actual = _to_float(actual_row.get("revenue_current", np.nan), np.nan)
        pf_actual = _to_float(actual_row.get("profit_after_tax", np.nan), np.nan)
        if not (np.isfinite(rev_actual) and np.isfinite(pf_actual)):
            missing.append(s)
            continue

        report_rows.append(
            {
                "symbol": s,
                "revenue_actual": float(rev_actual),
                "revenue_pred": float(rev_pred),
                "profit_actual": float(pf_actual),
                "profit_pred": float(pf_pred),
            }
        )
    report_df = pd.DataFrame(report_rows)

    table_csv = out_dir / "report_table.csv"
    report_df.to_csv(table_csv, index=False)

    md_lines = []
    md_lines.append(
        f"# Bao cao sai lech (train den target year {args.train_target_year_max}, du bao {args.predict_target_year})"
    )
    md_lines.append("")
    md_lines.append(f"| Ma | DT thuc te {args.predict_target_year} | DT du bao | Sai so tuyet doi | Sai so % |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for _, r in report_df.iterrows():
        err = float(r["revenue_pred"] - r["revenue_actual"])
        pct = (err / float(r["revenue_actual"]) * 100.0) if float(r["revenue_actual"]) != 0 else 0.0
        md_lines.append(
            f"| {r['symbol']} | {_fmt_num(float(r['revenue_actual']))} | {_fmt_num(float(r['revenue_pred']))} | {_fmt_signed(err)} | {_fmt_pct(pct)} |"
        )

    md_lines.append("")
    md_lines.append(f"| Ma | LNST thuc te {args.predict_target_year} | LNST du bao | Sai so tuyet doi | Sai so % |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for _, r in report_df.iterrows():
        err = float(r["profit_pred"] - r["profit_actual"])
        pct = (err / float(r["profit_actual"]) * 100.0) if float(r["profit_actual"]) != 0 else 0.0
        md_lines.append(
            f"| {r['symbol']} | {_fmt_num(float(r['profit_actual']))} | {_fmt_num(float(r['profit_pred']))} | {_fmt_signed(err)} | {_fmt_pct(pct)} |"
        )

    if missing:
        md_lines.append("")
        md_lines.append("## Missing Symbols")
        md_lines.append(", ".join(missing))

    report_md = out_dir / "report_style.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metrics_payload: dict[str, Any] = {
        "config": {
            "train_target_year_max": int(args.train_target_year_max),
            "predict_target_year": int(args.predict_target_year),
            "symbols": symbols,
            "macro_lag_years": int(args.macro_lag_years),
            "nonbank_feature_budget": int(args.nonbank_feature_budget),
            "nonbank_common_macros": nonbank_common_macros,
            "sector_macro_mappings": SECTOR_MACRO_MAPPINGS,
            "sector_macro_interaction_features": sector_macro_ix_features,
            "mandatory_nonbank_features": mandatory_nonbank,
            "steel_boost": float(args.steel_boost),
                        "include_year_feature": bool(args.include_year_feature),
            "steel_slug": steel_slug,
            "recency_weight_mode": args.recency_weight_mode,
            "recency_weight_min": float(args.recency_weight_min),
            "recency_weight_exp_base": float(args.recency_weight_exp_base),
            "enable_robust_clip": bool(args.enable_robust_clip),
            "robust_clip_lower_q": float(args.robust_clip_lower_q),
            "robust_clip_upper_q": float(args.robust_clip_upper_q),
            "robust_clip_columns": sorted(robust_clip_columns_used),
            "enable_debt_interest_adjustment": bool(args.enable_debt_interest_adjustment),
            "debt_interest_boost": float(args.debt_interest_boost),
            "debt_interest_high_q": float(args.debt_interest_high_q),
            "debt_interest_lower_q": float(args.debt_interest_lower_q),
            "debt_interest_upper_q": float(args.debt_interest_upper_q),
            "debt_interest_params": debt_interest_params,
            "debt_interest_applied": bool(debt_interest_applied),
            "dropped_identifier_columns": dropped_id_cols,
            "scaled_percent_columns": scaled_pct_cols,
            "log1p_nonnegative_macro_columns": logged_macro_cols,
        },
        "files": {
            "models": model_files,
            "detail_csv": str(detail_csv),
            "report_table_csv": str(table_csv),
            "report_markdown": str(report_md),
        },
        "missing_symbols": missing,
    }

    if not report_df.empty:
        y_rev = report_df["revenue_actual"].to_numpy(dtype=float)
        p_rev = report_df["revenue_pred"].to_numpy(dtype=float)
        y_pf = report_df["profit_actual"].to_numpy(dtype=float)
        p_pf = report_df["profit_pred"].to_numpy(dtype=float)
        metrics_payload["metrics"] = {
            "wape_revenue": _wape(y_rev, p_rev),
            "mape_revenue": _mape(y_rev, p_rev),
            "mae_revenue": _mae(y_rev, p_rev),
            "rmse_revenue": _rmse(y_rev, p_rev),
            "r2_revenue": _r2(y_rev, p_rev),
            "wape_profit": _wape(y_pf, p_pf),
            "mape_profit": _mape(y_pf, p_pf),
            "mae_profit": _mae(y_pf, p_pf),
            "rmse_profit": _rmse(y_pf, p_pf),
            "r2_profit": _r2(y_pf, p_pf),
            "n_symbols": int(len(report_df)),
        }

    if not detail_df.empty:
        holdout = detail_df[detail_df["target_year"] == int(args.predict_target_year)].copy()
        if not holdout.empty:
            metrics_all: dict[str, Any] = {}
            for target_key, alias in (("revenue_next", "revenue"), ("profit_after_tax_next", "profit")):
                sub_t = holdout[holdout["target"] == target_key]
                if sub_t.empty:
                    continue
                y = sub_t["actual"].to_numpy(dtype=float)
                p = sub_t["predicted"].to_numpy(dtype=float)
                metrics_all[alias] = {
                    "wape": _wape(y, p),
                    "mape": _mape(y, p),
                    "mae": _mae(y, p),
                    "rmse": _rmse(y, p),
                    "r2": _r2(y, p),
                    "n": int(len(sub_t)),
                }
            if metrics_all:
                metrics_payload["metrics_all_symbols"] = metrics_all

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Saved final pipeline outputs:")
    print("-", out_dir)
    print("- models:")
    for name in sorted(model_files.keys()):
        print(f"  - {name}: {model_files[name]}")
    print("- detail:", detail_csv)
    print("- table:", table_csv)
    print("- report:", report_md)
    print("- summary:", summary_json)
    if dropped_id_cols:
        print("- dropped id columns:", ", ".join(dropped_id_cols))
    if scaled_pct_cols:
        print("- scaled percent-like columns:", ", ".join(scaled_pct_cols))
    if robust_clip_columns_used:
        print("- robust-clip columns:", ", ".join(sorted(robust_clip_columns_used)))
    if debt_interest_applied:
        print("- debt-interest adjustment: enabled")
        print("  params:", debt_interest_params)
    if missing:
        print("Missing symbols:", ", ".join(missing))


if __name__ == "__main__":
    main()
