from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings


class ForecastToolService:
    def __init__(self) -> None:
        self.enabled = bool(settings.CHAT_FORECAST_ENABLED)
        self.report_table_csv = Path(settings.CHAT_FORECAST_REPORT_TABLE_CSV)
        self.detail_csv = Path(settings.CHAT_FORECAST_DETAIL_CSV)
        self.summary_json = Path(settings.CHAT_FORECAST_SUMMARY_JSON)
        self.on_demand_enabled = bool(settings.CHAT_FORECAST_ON_DEMAND_ENABLED)
        self.on_demand_script = Path(settings.CHAT_FORECAST_ON_DEMAND_SCRIPT)
        self.on_demand_output_dir = Path(settings.CHAT_FORECAST_ON_DEMAND_OUTPUT_DIR)
        self.on_demand_timeout = max(30, int(settings.CHAT_FORECAST_ON_DEMAND_TIMEOUT_SECONDS))
        self.top_factors = max(1, int(settings.CHAT_FORECAST_TOP_FACTORS))

    def get_company_forecast(self, symbol: str, target_year: int | None = None) -> dict[str, Any]:
        normalized_symbol = (symbol or "").strip().upper()
        if not normalized_symbol:
            return self._error("INVALID_ARGS", "symbol is required")

        if not self.enabled:
            return self._error("FORECAST_DISABLED", "Forecast tool is disabled")

        summary_payload = self._read_summary_json()
        default_year = self._extract_predict_year(summary_payload)
        effective_year = int(target_year) if isinstance(target_year, int) else default_year

        forecast_rows, source_ref, load_error = self._load_forecast_rows()
        if not forecast_rows:
            on_demand_row, on_demand_ref, on_demand_err = self._run_on_demand_forecast(
                symbol=normalized_symbol,
                target_year=effective_year,
            )
            if on_demand_row:
                return self._build_success_payload(
                    symbol=normalized_symbol,
                    effective_year=effective_year,
                    item=on_demand_row,
                    summary_payload=summary_payload,
                    source_ref=on_demand_ref,
                    model_version="final_model_pipeline_v1_on_demand",
                )
            missing_symbols = self._extract_missing_symbols(summary_payload)
            if missing_symbols:
                suffix = f"; on_demand_error={on_demand_err}" if on_demand_err else ""
                return self._error(
                    "FORECAST_NO_PREDICTIONS",
                    f"No forecast rows produced by pipeline. missing_symbols={missing_symbols}{suffix}",
                )
            if load_error:
                if on_demand_err:
                    return self._error("FORECAST_REPORT_READ_FAILED", f"{load_error}; on_demand_error={on_demand_err}")
                return self._error("FORECAST_REPORT_READ_FAILED", load_error)
            return self._error(
                "FORECAST_REPORT_EMPTY",
                f"No usable forecast rows in report_table/detail csv; on_demand_error={on_demand_err}",
            )

        candidates = [row for row in forecast_rows if str(row.get("symbol", "")).upper() == normalized_symbol]
        if not candidates:
            on_demand_row, on_demand_ref, on_demand_err = self._run_on_demand_forecast(
                symbol=normalized_symbol,
                target_year=effective_year,
            )
            if on_demand_row:
                return self._build_success_payload(
                    symbol=normalized_symbol,
                    effective_year=effective_year,
                    item=on_demand_row,
                    summary_payload=summary_payload,
                    source_ref=on_demand_ref,
                    model_version="final_model_pipeline_v1_on_demand",
                )
            if on_demand_err:
                return self._error(
                    "FORECAST_SYMBOL_NOT_FOUND",
                    f"No forecast found for symbol {normalized_symbol}; on_demand_error={on_demand_err}",
                )
            return self._error("FORECAST_SYMBOL_NOT_FOUND", f"No forecast found for symbol {normalized_symbol}")

        item = self._pick_best_row(candidates, effective_year)
        return self._build_success_payload(
            symbol=normalized_symbol,
            effective_year=effective_year,
            item=item,
            summary_payload=summary_payload,
            source_ref=source_ref,
            model_version="final_model_pipeline_v1",
        )

    def _build_success_payload(
        self,
        *,
        symbol: str,
        effective_year: int | None,
        item: dict[str, Any],
        summary_payload: dict[str, Any],
        source_ref: str | None,
        model_version: str,
    ) -> dict[str, Any]:
        quality = self._extract_quality(summary_payload)
        generated_at = self._extract_generated_at(summary_payload)
        assumptions = self._extract_assumptions(summary_payload)
        predict_year = int(effective_year) if isinstance(effective_year, int) else self._to_int(item.get("target_year"))
        if model_version.endswith("_on_demand") and isinstance(predict_year, int):
            assumptions = dict(assumptions)
            assumptions["predict_target_year"] = predict_year

        top_factors = item.get("top_factors") if isinstance(item.get("top_factors"), dict) else {}
        top_factors_ref = item.get("top_factors_source_ref") if isinstance(item.get("top_factors_source_ref"), str) else None
        if not top_factors:
            loaded_factors, loaded_ref = self._load_top_factors_from_source(
                source_ref=source_ref,
                symbol=symbol,
                target_year=predict_year,
            )
            if loaded_factors:
                top_factors = loaded_factors
                top_factors_ref = loaded_ref

        data = {
            "symbol": symbol,
            "predict_target_year": predict_year,
            "revenue_pred": self._to_float(item.get("revenue_pred")),
            "profit_pred": self._to_float(item.get("profit_pred")),
            "revenue_actual": self._to_float(item.get("revenue_actual")),
            "profit_actual": self._to_float(item.get("profit_actual")),
            "feature_year": self._to_int(item.get("feature_year")),
            "model_version": model_version,
            "generated_at": generated_at,
            "quality": quality,
            "assumptions": assumptions,
            "top_factors": top_factors,
        }
        refs: list[str] = []
        if source_ref:
            refs.append(source_ref)
        if top_factors_ref and top_factors_ref not in refs:
            refs.append(top_factors_ref)
        return {
            "ok": True,
            "data": data,
            "error_code": None,
            "error_message": None,
            "source_refs": refs,
        }

    def _read_summary_json(self) -> dict[str, Any]:
        if not self.summary_json.exists():
            return {}
        try:
            payload = json.loads(self.summary_json.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_predict_year(summary_payload: dict[str, Any]) -> int | None:
        config = summary_payload.get("config")
        if not isinstance(config, dict):
            return None
        year = config.get("predict_target_year")
        try:
            return int(year)
        except Exception:
            return None

    @staticmethod
    def _extract_quality(summary_payload: dict[str, Any]) -> dict[str, Any]:
        quality: dict[str, Any] = {}
        metrics = summary_payload.get("metrics")
        if isinstance(metrics, dict):
            for key in (
                "wape_revenue",
                "mape_revenue",
                "rmse_revenue",
                "r2_revenue",
                "wape_profit",
                "mape_profit",
                "rmse_profit",
                "r2_profit",
            ):
                if key in metrics:
                    quality[key] = metrics[key]
        return quality

    @staticmethod
    def _extract_assumptions(summary_payload: dict[str, Any]) -> dict[str, Any]:
        config = summary_payload.get("config")
        if not isinstance(config, dict):
            return {}
        keys = (
            "train_target_year_max",
            "predict_target_year",
            "macro_lag_years",
            "nonbank_feature_budget",
            "recency_weight_mode",
            "enable_robust_clip",
            "enable_debt_interest_adjustment",
        )
        out: dict[str, Any] = {}
        for key in keys:
            if key in config:
                out[key] = config[key]
        return out

    @staticmethod
    def _extract_generated_at(summary_payload: dict[str, Any]) -> str | None:
        config = summary_payload.get("config")
        if not isinstance(config, dict):
            return None
        value = config.get("generated_at") or config.get("generatedAt")
        return str(value).strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _extract_missing_symbols(summary_payload: dict[str, Any]) -> list[str]:
        value = summary_payload.get("missing_symbols")
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip().upper())
        return out

    def _load_forecast_rows(self) -> tuple[list[dict[str, Any]], str | None, str | None]:
        report_rows, report_err = self._read_report_table_rows()
        if report_rows:
            return report_rows, str(self.report_table_csv), None

        detail_rows, detail_err = self._read_predict_detail_rows()
        if detail_rows:
            return detail_rows, str(self.detail_csv), None

        err = report_err or detail_err
        return [], None, err

    def _run_on_demand_forecast(
        self,
        *,
        symbol: str,
        target_year: int | None,
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        if not self.on_demand_enabled:
            return None, None, "on_demand_disabled"
        if target_year is None:
            return None, None, "missing_target_year"
        if not self.on_demand_script.exists():
            return None, None, f"missing_script:{self.on_demand_script}"

        self.on_demand_output_dir.mkdir(parents=True, exist_ok=True)
        out_csv = self.on_demand_output_dir / f"forecast_{symbol}_{target_year}.csv"

        if not out_csv.exists():
            cmd = [
                sys.executable,
                str(self.on_demand_script),
                "--symbol",
                symbol,
                "--to-year",
                str(int(target_year)),
                "--out-csv",
                str(out_csv),
            ]
            try:
                subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.on_demand_timeout,
                )
            except subprocess.TimeoutExpired:
                return None, None, "on_demand_timeout"
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                detail = stderr or stdout or str(exc)
                return None, None, f"on_demand_failed:{detail[:300]}"
            except Exception as exc:
                return None, None, f"on_demand_failed:{type(exc).__name__}:{exc}"

        try:
            df = pd.read_csv(out_csv)
        except Exception as exc:
            return None, None, f"on_demand_read_failed:{type(exc).__name__}:{exc}"
        if df.empty:
            return None, None, "on_demand_empty_output"

        cols = {str(c).strip().lower(): c for c in df.columns}
        for required in ("symbol", "year", "revenue", "profit_after_tax"):
            if required not in cols:
                return None, None, f"on_demand_missing_col:{required}"

        df = df.copy()
        df["_symbol"] = df[cols["symbol"]].astype(str).str.upper().str.strip()
        df["_year"] = pd.to_numeric(df[cols["year"]], errors="coerce")
        df = df[(df["_symbol"] == symbol.upper()) & (df["_year"].notna())]
        if df.empty:
            return None, None, "on_demand_no_symbol_rows"

        target_rows = df[df["_year"] == int(target_year)]
        if target_rows.empty:
            target_rows = df.sort_values("_year").tail(1)
        target = target_rows.iloc[-1]

        base_rows = df[df.get(cols.get("source"), pd.Series(dtype=str)).astype(str).str.lower().eq("base")] if "source" in cols else pd.DataFrame()
        if base_rows.empty:
            base_rows = df.sort_values("_year").head(1)
        base = base_rows.iloc[-1] if not base_rows.empty else target

        row = {
            "symbol": symbol.upper(),
            "target_year": int(float(target["_year"])),
            "feature_year": int(float(base["_year"])) if base is not None else None,
            "revenue_actual": self._to_float(base.get(cols["revenue"])) if base is not None else None,
            "profit_actual": self._to_float(base.get(cols["profit_after_tax"])) if base is not None else None,
            "revenue_pred": self._to_float(target.get(cols["revenue"])),
            "profit_pred": self._to_float(target.get(cols["profit_after_tax"])),
        }
        explain_csv = out_csv.with_name(f"{out_csv.stem}_feature_drivers.csv")
        top_factors = self._read_feature_drivers(
            explain_csv,
            symbol=symbol,
            target_year=int(target_year),
            top_k=self.top_factors,
        )
        if top_factors:
            row["top_factors"] = top_factors
            row["top_factors_source_ref"] = str(explain_csv)
        return row, str(out_csv), None

    def _load_top_factors_from_source(
        self,
        *,
        source_ref: str | None,
        symbol: str,
        target_year: int | None,
    ) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
        if not source_ref:
            return {}, None
        src = Path(source_ref)
        candidates = [
            src.with_name(f"{src.stem}_feature_drivers.csv"),
            src.parent / "predict_detail_feature_drivers.csv",
            src.parent / "report_table_feature_drivers.csv",
        ]
        for candidate in candidates:
            factors = self._read_feature_drivers(
                candidate,
                symbol=symbol,
                target_year=target_year,
                top_k=self.top_factors,
            )
            if factors:
                return factors, str(candidate)
        return {}, None

    def _read_feature_drivers(
        self,
        csv_path: Path,
        *,
        symbol: str,
        target_year: int | None,
        top_k: int,
    ) -> dict[str, list[dict[str, Any]]]:
        if not csv_path.exists():
            return {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return {}
        if df.empty:
            return {}

        cols = {str(c).strip().lower(): c for c in df.columns}
        if "symbol" not in cols or "target" not in cols or "feature" not in cols:
            return {}

        work_all = df.copy()
        work_all["_symbol"] = work_all[cols["symbol"]].astype(str).str.upper().str.strip()
        work_all = work_all[work_all["_symbol"] == symbol.upper()]
        if work_all.empty:
            return {}

        pred_col = cols.get("pred_year") or cols.get("target_year")
        selected_year: int | None = None
        work = work_all
        if pred_col:
            work["_pred_year"] = pd.to_numeric(work[pred_col], errors="coerce")
            if isinstance(target_year, int):
                sliced = work[work["_pred_year"] == int(target_year)]
                if not sliced.empty:
                    work = sliced
                    selected_year = int(target_year)
            if "_pred_year" in work.columns and work["_pred_year"].notna().any():
                max_year = int(work["_pred_year"].max())
                work = work[work["_pred_year"] == max_year]
                selected_year = max_year

        score_col = cols.get("score")
        value_col = cols.get("feature_value")
        method_col = cols.get("method")
        rank_col = cols.get("rank")

        # Contribution share per target at the selected year.
        abs_sums: dict[str, float] = {}
        for _, row in work.iterrows():
            tgt = str(row.get(cols["target"], "")).strip().lower()
            score = self._to_float(row.get(score_col)) if score_col else None
            if score is None:
                continue
            abs_sums[tgt] = abs_sums.get(tgt, 0.0) + abs(score)

        # Previous-year lookup for "increase/decrease" narrative.
        prev_lookup: dict[tuple[str, str], float] = {}
        if selected_year is not None and pred_col:
            prev = work_all.copy()
            prev["_pred_year"] = pd.to_numeric(prev[pred_col], errors="coerce")
            prev = prev[prev["_pred_year"] == int(selected_year) - 1]
            if not prev.empty:
                for _, row in prev.iterrows():
                    tgt = str(row.get(cols["target"], "")).strip().lower()
                    feat = str(row.get(cols["feature"], "")).strip()
                    val = self._to_float(row.get(value_col)) if value_col else None
                    if feat and val is not None:
                        prev_lookup[(tgt, feat)] = float(val)

        bucket: dict[str, list[dict[str, Any]]] = {"revenue": [], "profit_after_tax": []}
        for _, row in work.iterrows():
            raw_target = str(row.get(cols["target"], "")).strip().lower()
            if "revenue" in raw_target:
                key = "revenue"
            elif "profit" in raw_target:
                key = "profit_after_tax"
            else:
                continue
            score = self._to_float(row.get(score_col)) if score_col else None
            curr_feature_val = self._to_float(row.get(value_col)) if value_col else None
            prev_feature_val = prev_lookup.get((raw_target, str(row.get(cols["feature"], "")).strip()))
            delta = None
            delta_pct = None
            trend = None
            if curr_feature_val is not None and prev_feature_val is not None:
                delta = float(curr_feature_val - prev_feature_val)
                if abs(prev_feature_val) > 1e-12:
                    delta_pct = float((delta / abs(prev_feature_val)) * 100.0)
                if delta > 1e-12:
                    trend = "increase"
                elif delta < -1e-12:
                    trend = "decrease"
                else:
                    trend = "flat"
            item = {
                "feature": str(row.get(cols["feature"], "")).strip(),
                "score": score,
                "abs_score": abs(score) if isinstance(score, float) else None,
                "score_pct": (abs(score) / abs_sums.get(raw_target, 1.0) * 100.0)
                if isinstance(score, float) and abs_sums.get(raw_target, 0.0) > 0
                else None,
                "feature_value": curr_feature_val,
                "feature_prev_value": prev_feature_val,
                "feature_delta": delta,
                "feature_delta_pct": delta_pct,
                "feature_trend": trend,
                "method": str(row.get(method_col, "")).strip() if method_col else "",
                "direction": "positive" if isinstance(score, float) and score > 0 else ("negative" if isinstance(score, float) and score < 0 else "neutral"),
                "rank": self._to_int(row.get(rank_col)) if rank_col else None,
            }
            if item["feature"]:
                bucket[key].append(item)

        out: dict[str, list[dict[str, Any]]] = {}
        for key, items in bucket.items():
            if not items:
                continue
            if all(isinstance(it.get("rank"), int) for it in items):
                items.sort(key=lambda it: int(it.get("rank") or 0))
            else:
                items.sort(key=lambda it: abs(float(it.get("score") or 0.0)), reverse=True)
            trimmed = items[: max(1, int(top_k))]
            for idx, it in enumerate(trimmed, start=1):
                it["rank"] = idx
            out[key] = trimmed
        return out

    def _read_report_table_rows(self) -> tuple[list[dict[str, Any]], str | None]:
        if not self.report_table_csv.exists():
            return [], f"Missing report_table.csv at {self.report_table_csv}"
        try:
            df = pd.read_csv(self.report_table_csv)
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"
        if df.empty:
            return [], "report_table.csv has no data rows"

        columns = {str(c).strip().lower(): c for c in df.columns}
        symbol_col = columns.get("symbol")
        rev_pred_col = columns.get("revenue_pred")
        prof_pred_col = columns.get("profit_pred")
        if not symbol_col or not rev_pred_col or not prof_pred_col:
            return [], f"report_table.csv missing required columns. columns={list(df.columns)}"

        feature_year_col = columns.get("feature_year")
        target_year_col = columns.get("target_year") or columns.get("predict_target_year")
        rev_actual_col = columns.get("revenue_actual")
        prof_actual_col = columns.get("profit_actual")

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            symbol = str(row.get(symbol_col, "")).strip().upper()
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "feature_year": self._to_int(row.get(feature_year_col)) if feature_year_col else None,
                    "target_year": self._to_int(row.get(target_year_col)) if target_year_col else None,
                    "revenue_actual": self._to_float(row.get(rev_actual_col)) if rev_actual_col else None,
                    "profit_actual": self._to_float(row.get(prof_actual_col)) if prof_actual_col else None,
                    "revenue_pred": self._to_float(row.get(rev_pred_col)),
                    "profit_pred": self._to_float(row.get(prof_pred_col)),
                }
            )
        return rows, None

    def _read_predict_detail_rows(self) -> tuple[list[dict[str, Any]], str | None]:
        if not self.detail_csv.exists():
            return [], f"Missing predict_detail.csv at {self.detail_csv}"
        try:
            df = pd.read_csv(self.detail_csv)
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"
        if df.empty:
            return [], "predict_detail.csv has no data rows"

        columns = {str(c).strip().lower(): c for c in df.columns}
        required = ("symbol", "target", "target_year", "predicted")
        missing = [col for col in required if col not in columns]
        if missing:
            return [], f"predict_detail.csv missing required columns={missing}. columns={list(df.columns)}"

        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for _, row in df.iterrows():
            symbol = str(row.get(columns["symbol"], "")).strip().upper()
            target = str(row.get(columns["target"], "")).strip().lower()
            target_year = self._to_int(row.get(columns["target_year"]))
            if not symbol or target_year is None:
                continue
            key = (symbol, target_year)
            item = grouped.setdefault(
                key,
                {
                    "symbol": symbol,
                    "feature_year": self._to_int(row.get(columns.get("feature_year"))) if columns.get("feature_year") else None,
                    "target_year": target_year,
                    "revenue_actual": None,
                    "profit_actual": None,
                    "revenue_pred": None,
                    "profit_pred": None,
                },
            )

            predicted = self._to_float(row.get(columns["predicted"]))
            actual = self._to_float(row.get(columns["actual"])) if columns.get("actual") else None
            if "revenue" in target:
                item["revenue_pred"] = predicted
                if actual is not None:
                    item["revenue_actual"] = actual
            elif "profit" in target:
                item["profit_pred"] = predicted
                if actual is not None:
                    item["profit_actual"] = actual

        rows = [value for value in grouped.values() if value.get("revenue_pred") is not None or value.get("profit_pred") is not None]
        return rows, None

    @staticmethod
    def _pick_best_row(rows: list[dict[str, Any]], target_year: int | None) -> dict[str, Any]:
        if not rows:
            return {}
        if target_year is not None:
            exact = [row for row in rows if row.get("target_year") == target_year]
            if exact:
                return exact[-1]
        with_year = [row for row in rows if isinstance(row.get("target_year"), int)]
        if with_year:
            with_year.sort(key=lambda item: int(item.get("target_year") or 0))
            return with_year[-1]
        return rows[-1]

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            if isinstance(value, str) and not value.strip():
                return None
            return int(float(value))
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "data": None,
            "error_code": code,
            "error_message": message,
            "source_refs": [],
        }
