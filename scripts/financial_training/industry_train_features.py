"""
Industry features for training: merge symbol→industry_group (CSV), one-hot or target encoding,
optional explicit macro×sector interactions (Option A: full macro + sector signals).

Train dùng file mapping nhanh/ổn định; inference có thể lấy từ DB (ngoài scope script này).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Explicit Level-3 industry -> macro whitelist.
# This avoids broad keyword mapping and keeps interaction features domain-driven.
INDUSTRY_MACROS: dict[str, tuple[str, ...]] = {
    "X_Y_D_NG_V_V_T_LI_U_X_Y_D_NG": ("hrc_log", "interest_loan_midlong_pct", "cpi_inflation_yoy_pp"),
    "V_N_T_I_KHO_B_I": ("oil_brent_log", "bdry_shipping_etf_log", "usd_vnd_log"),
    "U_T_B_T_NG_S_N_V_D_CH_V": (
        "cpi_inflation_yoy_pp",
        "gdp_ty_dong_log",
        "interest_loan_short_pct",
        "interest_loan_midlong_pct",
    ),
    "S_N_XU_T_TH_C_PH_M": ("sugar_log", "rice_log", "cpi_inflation_yoy_pp"),
    "GA_N_C_V_C_C_TI_N_CH_KH_C": ("nat_gas_log", "oil_brent_log"),
    "H_A_CH_T": ("rubber_log", "oil_brent_log", "usd_vnd_log"),
    "I_N": ("coal_log", "nat_gas_log", "oil_brent_log"),
    "D_NG_C_NH_N": ("cpi_inflation_yoy_pp", "usd_vnd_log"),
    "D_C_PH_M_V_C_NG_NGH_SINH_H_C": ("usd_vnd_log", "cpi_inflation_yoy_pp"),
    "KHAI_KHO_NG": ("coal_log", "iron_ore_log", "oil_brent_log"),
    "C_NG_TY_CH_NG_KHO_N": (
        "interest_deposit_12m_pct",
        "gdp_ty_dong_log",
        "vnindex_daily_return_mean_pct",
        "vnindex_growth_yoy_pct",
        "vnindex_trading_volume_avg",
        "vnindex_trading_value_avg",
    ),
    "D_CH_V_H_TR_T_V_N_THI_T_K": ("gdp_ty_dong_log", "interest_loan_short_pct"),
    "KIM_LO_I_C_NG_NGHI_P_KHAI_KHO_NG": ("hrc_log", "iron_ore_log", "coal_log"),
    "DU_L_CH_V_GI_I_TR": ("gdp_ty_dong_log", "oil_brent_log", "usd_vnd_log"),
    "U_NG": ("sugar_log", "cpi_inflation_yoy_pp", "coffee_log"),
    "TRUY_N_TH_NG": ("gdp_ty_dong_log",),
    "B_N_L_CHUNG": ("cpi_inflation_yoy_pp", "gdp_ty_dong_log"),
    "NG_N_H_NG": ("interest_deposit_12m_pct", "interest_loan_midlong_pct", "usd_vnd_log"),
    "C_KH_CH_T_O_M_Y": ("hrc_log", "usd_vnd_log", "iron_ore_log"),
    "H_NG_GIA_D_NG_X_Y_D_NG_NH_C_A": ("interest_loan_midlong_pct", "hrc_log"),
    "C_NG_NGHI_P_CHUNG": ("gdp_ty_dong_log", "interest_loan_short_pct"),
    "C_NG_NGH_PH_N_C_NG_V_THI_T_B": ("usd_vnd_log", "gdp_ty_dong_log"),
    "THI_T_B_I_N_I_N_T": ("hrc_log", "usd_vnd_log"),
    "L_M_NGHI_P_V_GI_Y": ("usd_vnd_log", "gdp_ty_dong_log"),
    "PH_N_M_M_V_D_CH_V_I_N_TO_N": ("gdp_ty_dong_log", "usd_vnd_log"),
    "T_V_LINH_KI_N_T": ("usd_vnd_log", "hrc_log", "interest_loan_midlong_pct"),
    "B_O_HI_M_PHI_NH_N_TH": ("interest_deposit_12m_pct", "cpi_inflation_yoy_pp"),
    "THI_T_B_V_D_CH_V_Y_T": ("usd_vnd_log", "cpi_inflation_yoy_pp"),
    "T_I_CH_NH_T_NG_H_P": ("interest_deposit_12m_pct", "gdp_ty_dong_log"),
    "B_N_L_TH_C_PH_M_V_D_C_PH_M": ("cpi_inflation_yoy_pp", "gdp_ty_dong_log"),
    "H_NG_TI_U_KHI_N": ("gdp_ty_dong_log", "cpi_inflation_yoy_pp"),
    "THU_C_L": ("cpi_inflation_yoy_pp", "gdp_ty_dong_log"),
    "VI_N_TH_NG_C_NH": ("gdp_ty_dong_log", "interest_loan_midlong_pct"),
    "B_O_HI_M_NH_N_TH": ("interest_deposit_12m_pct", "gdp_ty_dong_log"),
    "VI_N_TH_NG_DI_NG": ("gdp_ty_dong_log", "usd_vnd_log"),
}


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip()).strip("_")
    return x.upper() or "UNKNOWN"


def _normalize_vi_for_match(s: str) -> str:
    # Chuyển tiếng Việt có dấu về ASCII không dấu, loại whitespace/underscore để match substring.
    norm = unicodedata.normalize("NFKD", str(s))
    norm = norm.encode("ascii", errors="ignore").decode("ascii")
    norm = norm.upper()
    norm = norm.replace(" ", "").replace("_", "")
    return norm


def load_industry_map(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    if "symbol" not in m.columns or "industry_group" not in m.columns:
        raise ValueError(
            "industry map CSV must have columns: symbol, industry_group (industry_group_name_vi optional)"
        )
    m = m.copy()
    m["symbol"] = m["symbol"].astype(str).str.upper().str.strip()
    m["industry_group"] = m["industry_group"].map(_slug)
    if "industry_group_name_vi" in m.columns:
        m["industry_group_name_vi"] = m["industry_group_name_vi"].astype(str).str.strip()
    return m.drop_duplicates(subset=["symbol"], keep="last")


def merge_industry_column(df: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" not in out.columns:
        raise ValueError("train DataFrame must have 'symbol' for industry merge")
    out = out.drop(columns=["industry_group"], errors="ignore")
    out["_sym_key"] = out["symbol"].astype(str).str.upper().str.strip()
    m = map_df[["symbol", "industry_group"]].rename(columns={"symbol": "_sym_key"})
    out = out.merge(m, on="_sym_key", how="left")
    out = out.drop(columns=["_sym_key"])
    out["industry_group"] = out["industry_group"].fillna("UNKNOWN").map(_slug)
    return out


def _onehot_industry(df: pd.DataFrame, prefix: str = "ind") -> pd.DataFrame:
    d = pd.get_dummies(df["industry_group"], prefix=prefix, dtype=float)
    out = pd.concat([df.drop(columns=["industry_group"]), d], axis=1)
    return out


def _target_encode_industry(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    value_col: str,
    name: str = "industry_te_pat",
) -> pd.DataFrame:
    out = df.copy()
    if value_col not in out.columns:
        raise ValueError(f"target encoding needs column {value_col}")
    sub = out.loc[train_mask & out[value_col].notna()]
    if sub.empty:
        out[name] = 0.0
        return out
    means = sub.groupby("industry_group", observed=False)[value_col].mean()
    gm = float(means.mean()) if len(means) else 0.0
    out[name] = out["industry_group"].map(means).astype(float).fillna(gm)
    return out


def add_interaction_features_by_group(
    df: pd.DataFrame,
    *,
    industry_macros: dict[str, tuple[str, ...]],
    dummy_prefix: str = "ind",
    interaction_boost: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    boost = float(interaction_boost)
    if boost <= 0:
        boost = 1.0
    for group_slug, macro_cols in industry_macros.items():
        dummy_col = f"{dummy_prefix}_{group_slug}"
        if dummy_col not in out.columns:
            continue
        du = pd.to_numeric(out[dummy_col], errors="coerce").fillna(0.0)
        for macro_col in macro_cols:
            if macro_col not in out.columns:
                continue
            ma = pd.to_numeric(out[macro_col], errors="coerce").fillna(0.0)
            out[f"ix_{macro_col}__{group_slug}"] = ma * du * boost
    return out


def enrich_train_dataframe(
    df: pd.DataFrame,
    map_path: Path,
    *,
    encoding: str = "auto",
    max_onehot: int = 15,
    target_encoding_value_col: str = "profit_after_tax",
    train_mask: np.ndarray | None = None,
    interactions: bool = True,
    interaction_boost: float = 1.0,
) -> tuple[pd.DataFrame, frozenset[str]]:
    """
    Merge industry map, encode industry, add interactions (interactions only with one-hot).
    Returns (df, new_numeric_column_names) for median-fill grouping in train.
    """
    m = load_industry_map(map_path)
    out = merge_industry_column(df, m)
    new_cols: set[str] = set()

    n_groups = out["industry_group"].nunique()
    use_onehot = encoding == "onehot" or (encoding == "auto" and n_groups <= max_onehot)
    use_target = encoding == "target" or (encoding == "auto" and n_groups > max_onehot)

    if use_onehot:
        out = _onehot_industry(out)
        new_cols.update(c for c in out.columns if c.startswith("ind_"))
        if interactions:
            out = add_interaction_features_by_group(
                out,
                industry_macros=INDUSTRY_MACROS,
                dummy_prefix="ind",
                interaction_boost=interaction_boost,
            )
            new_cols.update(c for c in out.columns if c.startswith("ix_"))
    elif use_target:
        if train_mask is None:
            raise ValueError("target encoding requires train_mask")
        if target_encoding_value_col not in out.columns:
            raise ValueError(
                f"target encoding needs column {target_encoding_value_col!r} in training CSV"
            )
        out = _target_encode_industry(out, train_mask, target_encoding_value_col)
        new_cols.add("industry_te_pat")
        out = out.drop(columns=["industry_group"])
    else:
        raise ValueError(f"unknown encoding: {encoding}")

    if "industry_group" in out.columns:
        out = out.drop(columns=["industry_group"])

    added = frozenset(new_cols & set(out.columns))
    return out, added
