from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.services.chat.utils.math_helpers import as_int, as_float

logger = logging.getLogger(__name__)


class ForecastToolService:
    def __init__(self) -> None:
        self.enabled = bool(settings.CHAT_FORECAST_ENABLED)
        self.summary_json = Path(settings.CHAT_FORECAST_SUMMARY_JSON)
        self.forecast_script = Path(settings.CHAT_FORECAST_SCRIPT)
        self.forecast_timeout = max(30, int(settings.CHAT_FORECAST_TIMEOUT_SECONDS))
        self.top_factors = max(1, int(settings.CHAT_FORECAST_TOP_FACTORS))

    def get_company_forecast(self, symbol: str, target_year: int | None = None) -> dict[str, Any]:
        normalized_symbol = (symbol or "").strip().upper()
        if not normalized_symbol:
            return self._error("INVALID_ARGS", "symbol is required")
        if not re.fullmatch(r"[A-Z0-9]{1,10}", normalized_symbol):
            return self._error("INVALID_SYMBOL", f"Invalid symbol format: {symbol}")

        if not self.enabled:
            return self._error("FORECAST_DISABLED", "Forecast tool is disabled")

        summary_payload = self._read_summary_json()
        default_year = self._extract_predict_year(summary_payload)
        if target_year is not None:
            try:
                effective_year = int(target_year)
            except (TypeError, ValueError):
                effective_year = default_year
        else:
            effective_year = default_year

        forecast_row, forecast_err = self._run_forecast(
            symbol=normalized_symbol,
            target_year=effective_year,
        )
        if not forecast_row:
            return self._error("FORECAST_FAILED", forecast_err or "Forecast failed")

        return self._build_success_payload(
            symbol=normalized_symbol,
            effective_year=effective_year,
            item=forecast_row,
            summary_payload=summary_payload,
            model_version="production_pipeline_v1_runtime",
        )

    def _build_success_payload(
        self,
        *,
        symbol: str,
        effective_year: int | None,
        item: dict[str, Any],
        summary_payload: dict[str, Any],
        model_version: str,
    ) -> dict[str, Any]:
        quality = self._extract_quality(summary_payload)
        generated_at = self._extract_generated_at(summary_payload)
        assumptions = self._extract_assumptions(summary_payload)
        predict_year = int(effective_year) if isinstance(effective_year, int) else self._to_int(item.get("target_year"))
        if isinstance(predict_year, int):
            assumptions = dict(assumptions)
            assumptions["predict_target_year"] = predict_year

        top_factors = item.get("top_factors") if isinstance(item.get("top_factors"), dict) else {}

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
        return {
            "ok": True,
            "data": data,
            "error_code": None,
            "error_message": None,
            "source_refs": [],
        }

    def _read_summary_json(self) -> dict[str, Any]:
        if not self.summary_json.exists():
            return {}
        try:
            payload = json.loads(self.summary_json.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning("Failed to read summary JSON %s: %s", self.summary_json, exc)
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

    def _run_forecast(
        self,
        *,
        symbol: str,
        target_year: int | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if target_year is None:
            return None, "missing_target_year"
        if not self.forecast_script.exists():
            return None, f"missing_script:{self.forecast_script}"

        with tempfile.TemporaryDirectory(prefix="finflow_forecast_") as tmp:
            tmp_dir = Path(tmp)
            out_csv = tmp_dir / f"forecast_{symbol}_{target_year}.csv"
            cmd = [
                sys.executable,
                str(self.forecast_script),
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
                    timeout=self.forecast_timeout,
                )
            except subprocess.TimeoutExpired:
                return None, "forecast_timeout"
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                detail = stderr or stdout or str(exc)
                return None, f"forecast_failed:{detail[:300]}"
            except Exception as exc:
                return None, f"forecast_failed:{type(exc).__name__}:{exc}"

            try:
                df = pd.read_csv(out_csv)
            except Exception as exc:
                return None, f"forecast_read_failed:{type(exc).__name__}:{exc}"
            if df.empty:
                return None, "forecast_empty_output"

            cols = {str(c).strip().lower(): c for c in df.columns}
            for required in ("symbol", "year", "revenue", "profit_after_tax"):
                if required not in cols:
                    return None, f"forecast_missing_col:{required}"

            df = df.copy()
            df["_symbol"] = df[cols["symbol"]].astype(str).str.upper().str.strip()
            df["_year"] = pd.to_numeric(df[cols["year"]], errors="coerce")
            df = df[(df["_symbol"] == symbol.upper()) & (df["_year"].notna())]
            if df.empty:
                return None, "forecast_no_symbol_rows"

            target_rows = df[df["_year"] == int(target_year)]
            if target_rows.empty:
                target_rows = df.sort_values("_year").tail(1)
            target = target_rows.iloc[-1]

            base = target
            source_col = cols.get("source")
            if source_col and source_col in df.columns:
                base_rows = df[df[source_col].astype(str).str.lower().eq("base")]
                if not base_rows.empty:
                    base = base_rows.iloc[-1]
                else:
                    earlier = df[df["_year"] < int(target_year)]
                    if not earlier.empty:
                        base = earlier.sort_values("_year").iloc[-1]
            else:
                earlier = df[df["_year"] < int(target_year)]
                if not earlier.empty:
                    base = earlier.sort_values("_year").iloc[-1]

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
            return row, None

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
        except Exception as exc:
            logger.warning("Failed to read feature drivers CSV %s: %s", csv_path, exc)
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
            if selected_year is None and "_pred_year" in work.columns and work["_pred_year"].notna().any():
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

    _to_int = staticmethod(as_int)
    _to_float = staticmethod(as_float)

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "data": None,
            "error_code": code,
            "error_message": message,
            "source_refs": [],
        }
