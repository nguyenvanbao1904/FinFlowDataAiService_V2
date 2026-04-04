from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

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


def _merge_macro(df: pd.DataFrame, macro_csv: Path, lag_years: int = 1) -> pd.DataFrame:
    macro = pd.read_csv(macro_csv)
    macro["year"] = pd.to_numeric(macro["year"], errors="coerce").astype("Int64")
    macro = macro.dropna(subset=["year"]).astype({"year": int})
    if lag_years:
        macro["year"] = macro["year"] + int(lag_years)
    return df.merge(macro, on="year", how="left")


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


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float(default)


def _build_feature_frame(bundle: dict[str, Any], state: dict[str, Any]) -> pd.DataFrame:
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


def _predict_next(bundle: dict[str, Any], state: dict[str, Any], current_col: str) -> tuple[float, float, pd.DataFrame]:
    model = bundle["model"]
    X = _build_feature_frame(bundle, state)
    pred_t = float(model.predict(X)[0])
    pred_t = float(np.clip(pred_t, -3.0, 3.0))
    cur = _to_float(state.get(current_col, 0.0), 0.0)
    growth = float(np.sign(pred_t) * np.expm1(abs(pred_t)))
    return float(cur * (1.0 + growth)), pred_t, X


def _explain_top_features(bundle: dict[str, Any], X: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    cols = list(bundle["feature_columns"])
    model = bundle["model"]
    values = X.iloc[0].to_numpy(dtype=float)
    k = max(1, min(int(top_k), len(cols)))

    try:
        import xgboost as xgb

        booster = model.get_booster()
        dm = xgb.DMatrix(X[cols], feature_names=cols)
        contrib = np.asarray(booster.predict(dm, pred_contribs=True), dtype=float)[0]
        if contrib.shape[0] == len(cols) + 1:
            contrib = contrib[:-1]
        order = np.argsort(np.abs(contrib))[::-1][:k]
        return [
            {
                "feature": cols[int(i)],
                "feature_value": float(values[int(i)]),
                "score": float(contrib[int(i)]),
                "method": "shap_contrib",
            }
            for i in order
        ]
    except Exception:
        pass

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        if imp.shape[0] == len(cols):
            score = imp * np.abs(values)
            order = np.argsort(np.abs(score))[::-1][:k]
            return [
                {
                    "feature": cols[int(i)],
                    "feature_value": float(values[int(i)]),
                    "score": float(score[int(i)]),
                    "method": "importance_x_abs_value",
                }
                for i in order
            ]

    score = np.abs(values)
    order = np.argsort(score)[::-1][:k]
    return [
        {
            "feature": cols[int(i)],
            "feature_value": float(values[int(i)]),
            "score": float(score[int(i)]),
            "method": "abs_feature_value",
        }
        for i in order
    ]


def _recompute_derived(state: dict[str, Any], steel_boost: float) -> None:
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

    # Keep ROE/ROA/debt-to-equity on the original feature scale when available.
    # Recompute only as a fallback when the source value is missing.
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


def _apply_macro(state: dict[str, Any], macro_row: pd.Series, macro_cols: list[str]) -> None:
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


def _apply_debt_interest_adjustment_state(state: dict[str, Any], params: dict[str, Any] | None) -> None:
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


def _resolve_model_paths(args: argparse.Namespace) -> dict[str, Path]:
    d = Path(args.model_dir)
    paths = {
        "bank_revenue": Path(args.bank_revenue_model) if args.bank_revenue_model else d / "bank_revenue_next.joblib",
        "bank_profit": Path(args.bank_profit_model) if args.bank_profit_model else d / "bank_profit_after_tax_next.joblib",
        "nonbank_revenue": Path(args.nonbank_revenue_model)
        if args.nonbank_revenue_model
        else d / "nonbank_revenue_next.joblib",
        "nonbank_profit": Path(args.nonbank_profit_model)
        if args.nonbank_profit_model
        else d / "nonbank_profit_after_tax_next.joblib",
    }
    return paths


def _required_model_keys(dataset_name: str, predict_target: str) -> list[str]:
    prefix = "bank" if dataset_name == "bank" else "nonbank"
    keys: list[str] = []
    if predict_target in ("both", "revenue"):
        keys.append(f"{prefix}_revenue")
    if predict_target in ("both", "profit"):
        keys.append(f"{prefix}_profit")
    return keys


def _load_required_models(paths: dict[str, Path], dataset_name: str, predict_target: str) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for key in _required_model_keys(dataset_name, predict_target):
        p = paths[key]
        if not p.exists():
            raise FileNotFoundError(f"Missing required model file for {key}: {p}")
        bundles[key] = joblib.load(p)
    return bundles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test forecast using 4 final joblib models for one symbol from base year to a target year."
    )
    parser.add_argument("--symbol", type=str, required=True, help="Ticker symbol, e.g. ACB or MWG")
    parser.add_argument("--to-year", type=int, required=True, help="Forecast until this year (>= base year)")
    parser.add_argument("--base-year", type=int, default=2025, help="Base year to start recursive forecast")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts" / "models" / "final_model_pipeline",
        help="Directory containing the 4 model joblib files",
    )
    parser.add_argument("--bank-revenue-model", type=Path, default=None)
    parser.add_argument("--bank-profit-model", type=Path, default=None)
    parser.add_argument("--nonbank-revenue-model", type=Path, default=None)
    parser.add_argument("--nonbank-profit-model", type=Path, default=None)
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
    parser.add_argument("--steel-boost", type=float, default=1.0)
    parser.add_argument(
        "--history-mode",
        type=str,
        choices=["recursive", "use-actual-when-available"],
        default="use-actual-when-available",
        help="Backtest mode: use actual rows for known years, and recurse only beyond available history.",
    )
    parser.add_argument(
        "--predict-target",
        type=str,
        choices=["both", "revenue", "profit"],
        default="both",
        help="Choose forecast target: revenue, profit, or both",
    )
    parser.add_argument("--top-features", type=int, default=8, help="Top feature drivers to print for each yearly prediction")
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    symbol = args.symbol.strip().upper()
    if args.to_year < args.base_year:
        raise ValueError("--to-year must be >= --base-year")

    paths = _resolve_model_paths(args)

    bank = pd.read_csv(args.bank_preprocessed_csv)
    nonbank = pd.read_csv(args.nonbank_preprocessed_csv)
    if "quarter" in bank.columns:
        bank = bank[bank["quarter"] == 0].copy().reset_index(drop=True)
    if "quarter" in nonbank.columns:
        nonbank = nonbank[nonbank["quarter"] == 0].copy().reset_index(drop=True)

    bank, _ = _drop_identifier_columns(bank)
    nonbank, _ = _drop_identifier_columns(nonbank)

    bank = _merge_macro(bank, args.bank_macro_csv, lag_years=args.macro_lag_years)
    nonbank = _merge_macro(nonbank, args.nonbank_macro_csv, lag_years=args.macro_lag_years)
    bank, _ = _apply_log1p_nonnegative_columns(bank, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
    nonbank, _ = _apply_log1p_nonnegative_columns(nonbank, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
    nonbank = _merge_industry(nonbank, args.industry_map_csv)
    nonbank = _add_delta_features(nonbank)
    bank, _ = _normalize_percent_like_columns(bank, PERCENT_LIKE_COLUMNS)
    nonbank, _ = _normalize_percent_like_columns(nonbank, PERCENT_LIKE_COLUMNS)
    nonbank, _ = _add_steel_features(nonbank, args.steel_boost)
    dummies = pd.get_dummies(nonbank["industry_group"], prefix="ind", dtype=float)
    nonbank = pd.concat([nonbank, dummies], axis=1)
    nonbank, _ = _add_sector_macro_interactions(nonbank)

    in_bank = symbol in set(bank["symbol"].astype(str).str.upper())
    in_nonbank = symbol in set(nonbank["symbol"].astype(str).str.upper())
    if not in_bank and not in_nonbank:
        raise ValueError(f"Symbol {symbol} not found in bank/nonbank preprocessed datasets")

    if in_bank:
        ds_name = "bank"
        df = bank.copy()
        macro_src = pd.read_csv(args.bank_macro_csv).assign(year=lambda x: x["year"] + int(args.macro_lag_years))
        macro_src, _ = _apply_log1p_nonnegative_columns(macro_src, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
        macro_df, macro_cols = _macro_lookup(macro_src)
    else:
        ds_name = "nonbank"
        df = nonbank.copy()
        macro_src = pd.read_csv(args.nonbank_macro_csv).assign(year=lambda x: x["year"] + int(args.macro_lag_years))
        macro_src, _ = _apply_log1p_nonnegative_columns(macro_src, MACRO_LOG1P_NONNEGATIVE_COLUMNS)
        macro_df, macro_cols = _macro_lookup(macro_src)

    bundles = _load_required_models(paths, ds_name, args.predict_target)
    use_revenue = args.predict_target in ("both", "revenue")
    use_profit = args.predict_target in ("both", "profit")

    b_rev = None
    b_pf = None
    if ds_name == "bank":
        if use_revenue:
            b_rev = bundles["bank_revenue"]
        if use_profit:
            b_pf = bundles["bank_profit"]
    else:
        if use_revenue:
            b_rev = bundles["nonbank_revenue"]
        if use_profit:
            b_pf = bundles["nonbank_profit"]

    debt_interest_params: dict[str, Any] | None = None
    if ds_name == "nonbank":
        for b in (b_rev, b_pf):
            if isinstance(b, dict) and isinstance(b.get("debt_interest_params"), dict):
                if b.get("debt_interest_params"):
                    debt_interest_params = b["debt_interest_params"]
                    break

    sym_df = df[df["symbol"].astype(str).str.upper() == symbol].copy().reset_index(drop=True)
    sym_df["year"] = pd.to_numeric(sym_df["year"], errors="coerce").astype("Int64")
    sym_df = sym_df.dropna(subset=["year"]).astype({"year": int}).sort_values("year", kind="mergesort").reset_index(drop=True)
    actual_rows_by_year = {int(r["year"]): r.to_dict() for _, r in sym_df.iterrows()}

    base_rows = sym_df[pd.to_numeric(sym_df["year"], errors="coerce") == int(args.base_year)]
    if base_rows.empty:
        years = sorted(pd.to_numeric(sym_df["year"], errors="coerce").dropna().astype(int).unique().tolist())
        raise ValueError(f"No base row for {symbol} at year={args.base_year}. Available years: {years}")

    state = base_rows.iloc[-1].to_dict()
    state["symbol"] = symbol
    state["year"] = int(args.base_year)
    _refresh_sector_macro_interactions_state(state)

    records: list[dict[str, Any]] = [
        {
            "symbol": symbol,
            "dataset": ds_name,
            "year": int(args.base_year),
            "revenue": _to_float(state.get("revenue_current", 0.0), 0.0),
            "profit_after_tax": _to_float(state.get("profit_after_tax", 0.0), 0.0),
            "source": "base",
        }
    ]
    explain_rows: list[dict[str, Any]] = []

    for feature_year in range(int(args.base_year), int(args.to_year)):
        if args.history_mode == "use-actual-when-available" and int(feature_year) in actual_rows_by_year:
            state = dict(actual_rows_by_year[int(feature_year)])
            state["symbol"] = symbol
            state["year"] = int(feature_year)

        macro_row = _macro_row_for_year(macro_df, feature_year)
        _apply_macro(state, macro_row, macro_cols)
        _apply_debt_interest_adjustment_state(state, debt_interest_params)
        _recompute_derived(state, args.steel_boost)
        _refresh_sector_macro_interactions_state(state)

        pred_rev = None
        pred_rev_t = None
        pred_pf = None
        pred_pf_t = None
        rev_top: list[dict[str, Any]] = []
        pf_top: list[dict[str, Any]] = []

        if use_revenue and b_rev is not None:
            pred_rev, pred_rev_t, X_rev = _predict_next(b_rev, state, current_col="revenue_current")
            rev_top = _explain_top_features(b_rev, X_rev, args.top_features)
        if use_profit and b_pf is not None:
            pred_pf, pred_pf_t, X_pf = _predict_next(b_pf, state, current_col="profit_after_tax")
            pf_top = _explain_top_features(b_pf, X_pf, args.top_features)

        pred_year = feature_year + 1

        print(f"\nPrediction step: {feature_year} -> {pred_year}")
        if pred_rev is not None and pred_rev_t is not None:
            print(f"- Revenue forecast: {pred_rev:,.2f} (target_log_growth={pred_rev_t:.6f})")
            print("  Revenue top features:")
            for rank, item in enumerate(rev_top, start=1):
                print(
                    f"    {rank}. {item['feature']} | score={item['score']:.6f} | value={item['feature_value']:.6f} | {item['method']}"
                )
                explain_rows.append(
                    {
                        "symbol": symbol,
                        "dataset": ds_name,
                        "feature_year": int(feature_year),
                        "pred_year": int(pred_year),
                        "target": "revenue",
                        "rank": int(rank),
                        "feature": str(item["feature"]),
                        "score": float(item["score"]),
                        "feature_value": float(item["feature_value"]),
                        "method": str(item["method"]),
                    }
                )
        if pred_pf is not None and pred_pf_t is not None:
            print(f"- Profit forecast: {pred_pf:,.2f} (target_log_growth={pred_pf_t:.6f})")
            print("  Profit top features:")
            for rank, item in enumerate(pf_top, start=1):
                print(
                    f"    {rank}. {item['feature']} | score={item['score']:.6f} | value={item['feature_value']:.6f} | {item['method']}"
                )
                explain_rows.append(
                    {
                        "symbol": symbol,
                        "dataset": ds_name,
                        "feature_year": int(feature_year),
                        "pred_year": int(pred_year),
                        "target": "profit_after_tax",
                        "rank": int(rank),
                        "feature": str(item["feature"]),
                        "score": float(item["score"]),
                        "feature_value": float(item["feature_value"]),
                        "method": str(item["method"]),
                    }
                )

        records.append(
            {
                "symbol": symbol,
                "dataset": ds_name,
                "year": int(pred_year),
                "revenue": float(pred_rev) if pred_rev is not None else np.nan,
                "profit_after_tax": float(pred_pf) if pred_pf is not None else np.nan,
                "source": "predicted",
            }
        )

        prev_margin = _to_float(state.get("profit_margin_calc_pct", 0.0), 0.0)
        prev_rev = _to_float(state.get("revenue_current", 0.0), 0.0)
        prev_pf = _to_float(state.get("profit_after_tax", 0.0), 0.0)
        prev_rev_l1 = _to_float(state.get("revenue_lag1", 0.0), 0.0)
        prev_pf_l1 = _to_float(state.get("profit_lag1", 0.0), 0.0)
        prev_debt_ratio = _to_float(state.get("net_debt_to_equity_pct", np.nan), np.nan)
        prev_debt_ratio_l1 = _to_float(state.get("net_debt_to_equity_pct_lag1", np.nan), np.nan)

        state["year"] = int(pred_year)
        if args.history_mode == "use-actual-when-available" and int(pred_year) in actual_rows_by_year:
            state = dict(actual_rows_by_year[int(pred_year)])
            state["symbol"] = symbol
            state["year"] = int(pred_year)
        else:
            if pred_rev is not None:
                state["revenue_lag2"] = prev_rev_l1
                state["revenue_lag1"] = prev_rev
                state["revenue_current"] = float(pred_rev)
            if pred_pf is not None:
                state["profit_lag2"] = prev_pf_l1
                state["profit_lag1"] = prev_pf
                state["profit_after_tax"] = float(pred_pf)
            if np.isfinite(prev_debt_ratio):
                state["net_debt_to_equity_pct_lag1"] = float(prev_debt_ratio)
            elif np.isfinite(prev_debt_ratio_l1):
                state["net_debt_to_equity_pct_lag1"] = float(prev_debt_ratio_l1)

        rev_now = _to_float(state.get("revenue_current", 0.0), 0.0)
        pf_now = _to_float(state.get("profit_after_tax", 0.0), 0.0)
        if rev_now != 0:
            state["profit_margin_calc_pct"] = float((pf_now / rev_now) * 100.0)
        else:
            state["profit_margin_calc_pct"] = 0.0
        state["profit_margin_change"] = float(state["profit_margin_calc_pct"] - prev_margin)

    out = pd.DataFrame(records)
    out = out[(out["year"] >= int(args.base_year)) & (out["year"] <= int(args.to_year))].reset_index(drop=True)

    if not use_revenue and "revenue" in out.columns:
        out = out.drop(columns=["revenue"])
    if not use_profit and "profit_after_tax" in out.columns:
        out = out.drop(columns=["profit_after_tax"])

    if args.out_csv is None:
        suffix = "" if args.predict_target == "both" else f"_{args.predict_target}"
        out_csv = ROOT / "artifacts" / "models" / "final_model_pipeline" / f"forecast_{symbol}_{args.base_year}_{args.to_year}{suffix}.csv"
    else:
        out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    explain_df = pd.DataFrame(explain_rows)
    explain_csv = out_csv.with_name(f"{out_csv.stem}_feature_drivers.csv")
    explain_df.to_csv(explain_csv, index=False)

    show = out.copy()
    if "revenue" in show.columns:
        show["revenue"] = show["revenue"].map(lambda x: f"{x:,.2f}")
    if "profit_after_tax" in show.columns:
        show["profit_after_tax"] = show["profit_after_tax"].map(lambda x: f"{x:,.2f}")

    print(f"Forecast for {symbol} ({ds_name}) from base {args.base_year} to {args.to_year}")
    print(show.to_string(index=False))
    print("Saved:", out_csv)
    print("Models used:")
    for key in _required_model_keys(ds_name, args.predict_target):
        label = "revenue" if key.endswith("revenue") else "profit"
        print(f"- {label}:", paths[key])
    print("Feature drivers saved:", explain_csv)


if __name__ == "__main__":
    main()
