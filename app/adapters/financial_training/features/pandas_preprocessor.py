from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from app.domain.financial_training.entities.raw_row import FinancialRawRow
from app.domain.financial_training.ports.preprocessing import (
    FinancialPreprocessorPort,
)


class PandasFinancialPreprocessor(FinancialPreprocessorPort):
    """
    Adapter: implement pandas-based preprocessing (scaling, feature building, label creation).
    """

    _BANK_MONEY_COLS = (
        "cash_and_equivalents",
        "deposits_at_sbv",
        "interbank_placements",
        "trading_securities",
        "investment_securities",
        "customer_loans",
        "sbv_borrowings",
        "customer_deposits",
        "valuable_papers",
        "deposits_borrowings_others",
        "total_liabilities",
        "equity",
        "total_assets",
        "total_capital",
        "net_interest_income",
        "fee_and_commission_income",
        "other_income",
        "profit_after_tax",
        "interest_expense",
    )

    _NONBANK_MONEY_COLS = (
        "cash_and_equivalents",
        "short_term_investments",
        "short_term_receivables",
        "inventories",
        "fixed_assets",
        "long_term_receivables",
        "total_assets_reported",
        "equity",
        "short_term_borrowings",
        "long_term_borrowings",
        "advances_from_customers",
        "total_capital_reported",
        "total_liabilities",
        "net_revenue",
        "total_revenue",
        "profit_after_tax",
    )

    @staticmethod
    def _make_annual_from_quarters(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Build annual rows (quarter=0) from quarterly rows (quarter>0).
        Mirrors backend behavior: sum income statement flows over quarters and
        take latest quarter (typically Q4) for balance-sheet / indicator snapshot.
        """
        if df.empty:
            return df
        if "symbol" not in df.columns or "year" not in df.columns or "quarter" not in df.columns:
            return df

        d = df.copy()
        d = d[d["quarter"].fillna(0).astype(int) > 0]
        if d.empty:
            return d

        d = d.sort_values(["symbol", "year", "quarter"], kind="mergesort")

        # Identify flow columns to sum by year (income statement style)
        if dataset_type == "bank":
            flow_cols = [c for c in ["net_interest_income", "fee_and_commission_income", "other_income", "profit_after_tax", "interest_expense"] if c in d.columns]
        else:
            flow_cols = [c for c in ["net_revenue", "total_revenue", "profit_after_tax"] if c in d.columns]

        # Snapshot columns: take the latest quarter row per (symbol, year)
        snapshot_cols = [
            c
            for c in d.columns
            if c not in set(flow_cols)
            and c not in {"quarter", "symbol", "year"}
        ]

        last = d.groupby(["symbol", "year"], sort=False, as_index=False).tail(1)
        annual = last[["symbol", "year"] + [c for c in snapshot_cols if c in last.columns]].copy()
        annual["quarter"] = 0

        if flow_cols:
            summed = d.groupby(["symbol", "year"], sort=False)[flow_cols].sum(min_count=1).reset_index()
            annual = annual.merge(summed, on=["symbol", "year"], how="left")

        # Ensure company_type is consistent if present
        if "company_type" in annual.columns:
            annual["company_type"] = annual["company_type"].astype(str)

        return annual

    @staticmethod
    def _sum_if_any_present(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        parts = df[cols]
        any_present = parts.notna().any(axis=1)
        s = parts.fillna(0).sum(axis=1)
        s = s.where(any_present, pd.NA)
        return s

    @staticmethod
    def _as_numeric(df: pd.DataFrame, cols: list[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    @staticmethod
    def _scale_money_billion(df: pd.DataFrame, cols: list[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") / 1e9

    def transform(
        self,
        raw_rows: Sequence[FinancialRawRow],
        dataset_spec: Any,
    ) -> Any:
        if dataset_spec is None:
            raise ValueError("dataset_spec is required")

        extra = getattr(dataset_spec, "extra", None) or {}
        row_filter: Mapping[str, str] = extra.get("row_filter", {}) or {}
        revenue_def: Mapping[str, str] = extra.get("revenue_definition", {}) or {}

        if not raw_rows:
            return {"df": pd.DataFrame(), "features": pd.DataFrame(), "labels": pd.DataFrame(), "metadata": {}}

        df = pd.DataFrame([r.payload for r in raw_rows])

        # Coerce key columns
        for k in ("symbol", "year", "quarter", "company_type"):
            if k in df.columns:
                if k in ("year", "quarter"):
                    df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
                else:
                    df[k] = df[k].astype(str)

        # Determine dataset type
        dataset_type = None
        if "bank_formula" in revenue_def:
            dataset_type = "bank"
        if "non_bank_formula" in revenue_def:
            dataset_type = "non_bank"
        if dataset_type is None:
            # fallback by company_type column
            if "company_type" in df.columns and df["company_type"].str.upper().eq("BANK").any():
                dataset_type = "bank"
            else:
                dataset_type = "non_bank"

        # Filter by company_type if present (CsvFinancialDataSource concatenates both)
        if "company_type" in df.columns:
            if dataset_type == "bank":
                df = df[df["company_type"].str.upper() == "BANK"]
            else:
                df = df[df["company_type"].str.upper().isin(["NON_BANK", "NORMAL"])]

        # Apply row_filter quarter condition
        qc = (row_filter.get("quarter_condition") or "").strip()
        if qc == "quarter == 0":
            # Raw export is typically quarterly-only. Build annual synthetic rows from quarters.
            annual_df = self._make_annual_from_quarters(df, dataset_type=dataset_type)
            df = annual_df
        elif qc == "quarter > 0":
            df = df[df["quarter"].fillna(0).astype(int) > 0]

        # Numeric coercion for known columns used in derived features
        if dataset_type == "bank":
            self._as_numeric(df, list(self._BANK_MONEY_COLS) + ["roe", "roa", "pe", "pb", "ps", "eps", "bvps"])
            self._scale_money_billion(df, [c for c in self._BANK_MONEY_COLS if c in df.columns])
        else:
            self._as_numeric(
                df,
                list(self._NONBANK_MONEY_COLS)
                + ["roe", "roa", "pe", "pb", "ps", "eps", "bvps", "gross_margin", "net_margin"],
            )
            self._scale_money_billion(df, [c for c in self._NONBANK_MONEY_COLS if c in df.columns])

        # Compute revenue_current (scaled to billion if inputs are scaled)
        if dataset_type == "bank":
            cols = [c for c in ["net_interest_income", "fee_and_commission_income", "other_income"] if c in df.columns]
            df["revenue_current"] = self._sum_if_any_present(df, cols) if cols else pd.NA
        else:
            df["revenue_current"] = df["net_revenue"] if "net_revenue" in df.columns else pd.NA

        # Derived line-chart features
        if dataset_type == "bank":
            # NIM (%): quarter==0 no ×4; quarter>0 ×4 annualize
            if "net_interest_income" in df.columns and "total_assets" in df.columns:
                q = df["quarter"].fillna(0).astype(int)
                factor = q.apply(lambda x: 1.0 if x == 0 else 4.0)
                annualized = df["net_interest_income"] * factor
                ta = df["total_assets"]
                df["nim_pct"] = (annualized / ta) * 100.0
                df.loc[(ta.isna()) | (ta <= 0), "nim_pct"] = pd.NA

            # TS/VCSH line in bank capital chart
            if "total_assets" in df.columns and "equity" in df.columns:
                eq = pd.to_numeric(df["equity"], errors="coerce")
                ta = pd.to_numeric(df["total_assets"], errors="coerce")
                df["assets_to_equity"] = ta / eq
                df.loc[(eq.isna()) | (eq == 0), "assets_to_equity"] = pd.NA
        else:
            # receivable ratio (%)
            st_rec = df.get("short_term_receivables", pd.Series([pd.NA] * len(df)))
            lt_rec = df.get("long_term_receivables", pd.Series([pd.NA] * len(df)))
            cash = df.get("cash_and_equivalents", pd.Series([0.0] * len(df)))
            st_inv = df.get("short_term_investments", pd.Series([0.0] * len(df)))
            inv = df.get("inventories", pd.Series([0.0] * len(df)))
            fa = df.get("fixed_assets", pd.Series([0.0] * len(df)))
            ta_rep = df.get("total_assets_reported", pd.Series([pd.NA] * len(df)))

            known_sum = pd.to_numeric(cash, errors="coerce").fillna(0) + pd.to_numeric(st_inv, errors="coerce").fillna(0)
            known_sum += pd.to_numeric(st_rec, errors="coerce").fillna(0) + pd.to_numeric(inv, errors="coerce").fillna(0)
            known_sum += pd.to_numeric(fa, errors="coerce").fillna(0) + pd.to_numeric(lt_rec, errors="coerce").fillna(0)

            denom = ta_rep.where(pd.to_numeric(ta_rep, errors="coerce") > 0, pd.NA)
            denom = denom.fillna(known_sum)
            df["receivable_ratio_pct"] = ((pd.to_numeric(st_rec, errors="coerce").fillna(0) + pd.to_numeric(lt_rec, errors="coerce").fillna(0)) / denom) * 100.0
            df.loc[(denom.isna()) | (denom <= 0), "receivable_ratio_pct"] = pd.NA

            # net debt / equity (%)
            eq = df.get("equity", pd.Series([pd.NA] * len(df)))
            st_b = df.get("short_term_borrowings", pd.Series([0.0] * len(df)))
            lt_b = df.get("long_term_borrowings", pd.Series([0.0] * len(df)))
            net_debt = (pd.to_numeric(st_b, errors="coerce").fillna(0) + pd.to_numeric(lt_b, errors="coerce").fillna(0)) - (
                pd.to_numeric(cash, errors="coerce").fillna(0) + pd.to_numeric(st_inv, errors="coerce").fillna(0)
            )
            df["net_debt"] = net_debt
            eq_num = pd.to_numeric(eq, errors="coerce")
            df["net_debt_to_equity_pct"] = (net_debt / eq_num) * 100.0
            df.loc[(eq_num.isna()) | (eq_num == 0), "net_debt_to_equity_pct"] = pd.NA

        # Sort and create next-period labels
        df = df.sort_values(["symbol", "year", "quarter"], kind="mergesort").reset_index(drop=True)

        # Time-series signal features (per symbol) for next-period forecasting.
        grp = df.groupby("symbol", sort=False)
        df["revenue_lag1"] = grp["revenue_current"].shift(1)
        df["revenue_lag2"] = grp["revenue_current"].shift(2)
        df["profit_lag1"] = grp["profit_after_tax"].shift(1)
        df["profit_lag2"] = grp["profit_after_tax"].shift(2)

        df["revenue_roll3_mean"] = grp["revenue_current"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        df["profit_roll3_mean"] = grp["profit_after_tax"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        df["revenue_roll3_std"] = grp["revenue_current"].transform(
            lambda s: s.rolling(window=3, min_periods=2).std()
        )
        df["profit_roll3_std"] = grp["profit_after_tax"].transform(
            lambda s: s.rolling(window=3, min_periods=2).std()
        )

        # Safe momentum: avoid exploding ratios when previous values are too close to zero.
        rev_lag_abs = pd.to_numeric(df["revenue_lag1"], errors="coerce").abs()
        prof_lag_abs = pd.to_numeric(df["profit_lag1"], errors="coerce").abs()
        rev_denom = rev_lag_abs.where(rev_lag_abs >= 1.0, pd.NA)
        prof_denom = prof_lag_abs.where(prof_lag_abs >= 1.0, pd.NA)
        df["revenue_momentum_pct"] = (
            (pd.to_numeric(df["revenue_current"], errors="coerce") - pd.to_numeric(df["revenue_lag1"], errors="coerce"))
            / rev_denom
            * 100.0
        )
        df["profit_momentum_pct"] = (
            (pd.to_numeric(df["profit_after_tax"], errors="coerce") - pd.to_numeric(df["profit_lag1"], errors="coerce"))
            / prof_denom
            * 100.0
        )

        # Redundant but robust margin signal when gross/net margin fields are missing.
        rev_safe = pd.to_numeric(df["revenue_current"], errors="coerce")
        rev_safe = rev_safe.where(rev_safe.abs() >= 1.0, pd.NA)
        df["profit_margin_calc_pct"] = (
            pd.to_numeric(df["profit_after_tax"], errors="coerce") / rev_safe * 100.0
        )

        # YoY growth lines used by FE charts.
        prev_rev = df.groupby("symbol", sort=False)["revenue_current"].shift(1)
        prev_profit = df.groupby("symbol", sort=False)["profit_after_tax"].shift(1)

        if dataset_type == "bank":
            # Bank income YoY chart allows previous value < 0, only blocks prev == 0.
            cond_rev = prev_rev != 0
        else:
            # Non-bank revenue YoY chart requires previous revenue > 0.
            cond_rev = prev_rev > 0
        cond_rev = cond_rev.fillna(False)
        cond_profit = (prev_profit != 0).fillna(False)

        df["revenue_yoy_pct"] = ((df["revenue_current"] - prev_rev) / prev_rev) * 100.0
        df.loc[~cond_rev, "revenue_yoy_pct"] = pd.NA

        df["profit_yoy_pct"] = ((df["profit_after_tax"] - prev_profit) / prev_profit) * 100.0
        df.loc[~cond_profit, "profit_yoy_pct"] = pd.NA

        df["revenue_next"] = df.groupby("symbol", sort=False)["revenue_current"].shift(-1)
        df["profit_after_tax_next"] = df.groupby("symbol", sort=False)["profit_after_tax"].shift(-1)

        labels = df[["revenue_next", "profit_after_tax_next"]]

        # Features: raw numeric + derived. For step 2 we keep everything except labels.
        id_cols = getattr(dataset_spec, "id_columns", None) or ["symbol", "year", "quarter"]
        feature_df = df.drop(columns=[c for c in ["revenue_next", "profit_after_tax_next"] if c in df.columns])

        meta = {
            "row_filter": row_filter,
            "revenue_definition": revenue_def,
            "dataset_type": dataset_type,
            "scaled_money_unit": "ty_vnd",
            "id_columns": id_cols,
        }

        return {"df": df, "features": feature_df, "labels": labels, "metadata": meta}

