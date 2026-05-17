from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.financial_training.symbol_lists import VN30, VN100, VN100_CSV


TARGET_ALIASES = {
    "revenue_next": "revenue",
    "profit_after_tax_next": "profit",
}

METRIC_COLUMNS = ["wape", "mape", "mae", "rmse", "r2"]


def _parse_years(raw: str) -> list[int]:
    years = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not years:
        raise ValueError("--fold-years must contain at least one year")
    return sorted(dict.fromkeys(years))


def _parse_scopes(raw: str) -> list[str]:
    allowed = {"all", "vn100", "vn30"}
    scopes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    bad = [item for item in scopes if item not in allowed]
    if bad:
        raise ValueError(f"Unsupported scopes: {bad}. Allowed: {sorted(allowed)}")
    return scopes or ["all", "vn100", "vn30"]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _wape(y_true: list[float], y_pred: list[float]) -> float | None:
    denom = sum(abs(v) for v in y_true)
    if denom == 0:
        return None
    return sum(abs(a - p) for a, p in zip(y_true, y_pred)) / denom * 100.0


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    y_true: list[float] = []
    y_pred: list[float] = []
    for row in rows:
        actual = _to_float(row.get("actual"))
        predicted = _to_float(row.get("predicted"))
        if actual is None or predicted is None:
            continue
        y_true.append(actual)
        y_pred.append(predicted)

    if not y_true:
        return {
            "n": 0,
            "wape": None,
            "mape": None,
            "mae": None,
            "rmse": None,
            "r2": None,
        }

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    nonzero = y != 0

    return {
        "n": len(y_true),
        "wape": _wape(y_true, y_pred),
        "mape": float(mean_absolute_percentage_error(y[nonzero], p[nonzero]) * 100.0) if np.any(nonzero) else None,
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(root_mean_squared_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
    }


def _read_detail_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _scope_symbols(scope: str) -> set[str] | None:
    if scope == "all":
        return None
    if scope == "vn100":
        return set(VN100)
    if scope == "vn30":
        return set(VN30)
    raise ValueError(f"Unsupported scope: {scope}")


def _evaluate_fold(detail_csv: Path, fold_year: int, scopes: list[str]) -> list[dict[str, Any]]:
    rows = _read_detail_csv(detail_csv)
    out: list[dict[str, Any]] = []

    for scope in scopes:
        symbols = _scope_symbols(scope)
        scoped = [
            row
            for row in rows
            if int(float(row.get("target_year") or 0)) == int(fold_year)
            and (symbols is None or str(row.get("symbol", "")).upper() in symbols)
        ]
        for target_key, target_name in TARGET_ALIASES.items():
            target_rows = [row for row in scoped if row.get("target") == target_key]
            metrics = _metrics(target_rows)
            out.append({
                "scope": scope.upper(),
                "fold": fold_year,
                "train_until": fold_year - 1,
                "test_year": fold_year,
                "target": target_name,
                **metrics,
            })
    return out


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def _aggregate(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(row["scope"], row["target"]) for row in fold_rows})
    for scope, target in groups:
        rows = [row for row in fold_rows if row["scope"] == scope and row["target"] == target]
        item: dict[str, Any] = {
            "scope": scope,
            "target": target,
            "folds": len(rows),
            "n_total": sum(int(row.get("n") or 0) for row in rows),
            "n_mean": _mean([float(row["n"]) for row in rows if row.get("n") is not None]),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            item[f"{metric}_mean"] = _mean(values)
            item[f"{metric}_std"] = _std(values)
        out.append(item)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return "-"
    return f"{out:.{digits}f}"


def _write_report(path: Path, fold_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Walk-forward Model Evaluation",
        "",
        "Expanding-window validation: train to year T-1, test year T.",
        "",
        "## Aggregate",
        "",
        "| Scope | Target | Folds | N total | WAPE mean | WAPE std | MAPE mean | R2 mean | R2 std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {scope} | {target} | {folds} | {n_total} | {wape_mean} | {wape_std} | {mape_mean} | {r2_mean} | {r2_std} |".format(
                scope=row["scope"],
                target=row["target"],
                folds=row["folds"],
                n_total=row["n_total"],
                wape_mean=_fmt(row.get("wape_mean")),
                wape_std=_fmt(row.get("wape_std")),
                mape_mean=_fmt(row.get("mape_mean")),
                r2_mean=_fmt(row.get("r2_mean"), digits=4),
                r2_std=_fmt(row.get("r2_std"), digits=4),
            )
        )

    lines.extend([
        "",
        "## Fold Details",
        "",
        "| Scope | Fold | Train until | Target | N | WAPE | MAPE | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in fold_rows:
        lines.append(
            "| {scope} | {fold} | {train_until} | {target} | {n} | {wape} | {mape} | {mae} | {rmse} | {r2} |".format(
                scope=row["scope"],
                fold=row["fold"],
                train_until=row["train_until"],
                target=row["target"],
                n=row["n"],
                wape=_fmt(row.get("wape")),
                mape=_fmt(row.get("mape")),
                mae=_fmt(row.get("mae")),
                rmse=_fmt(row.get("rmse")),
                r2=_fmt(row.get("r2"), digits=4),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_pipeline(
    *,
    python_bin: str,
    pipeline_script: Path,
    source: str,
    fold_year: int,
    out_dir: Path,
    symbols: str,
    extra_args: list[str],
) -> None:
    cmd = [
        python_bin,
        str(pipeline_script),
        "--source",
        source,
        "--train-target-year-max",
        str(fold_year - 1),
        "--predict-target-year",
        str(fold_year),
        "--symbols",
        symbols,
        "--out-dir",
        str(out_dir),
        *extra_args,
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run expanding-window model evaluation once and report metrics for ALL, VN100, and VN30."
    )
    parser.add_argument("--fold-years", default="2022,2023,2024,2025")
    parser.add_argument("--scopes", default="all,vn100,vn30")
    parser.add_argument("--source", choices=["db", "csv"], default="db")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "models" / "walk_forward_eval")
    parser.add_argument("--pipeline-script", type=Path, default=Path(__file__).with_name("run_final_model_pipeline.py"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--symbols-for-report",
        default=VN100_CSV,
        help="Symbols passed to run_final_model_pipeline for report_table only. Scope metrics are recomputed from predict_detail.csv.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Do not rerun a fold when fold_<year>/predict_detail.csv already exists.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip all training and aggregate existing fold outputs only.",
    )
    parser.add_argument(
        "extra_pipeline_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to run_final_model_pipeline.py after a standalone -- separator.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    fold_years = _parse_years(args.fold_years)
    scopes = _parse_scopes(args.scopes)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.extra_pipeline_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    fold_rows: list[dict[str, Any]] = []
    for fold_year in fold_years:
        fold_dir = out_dir / f"fold_{fold_year}"
        detail_csv = fold_dir / "predict_detail.csv"
        if not args.aggregate_only and not (args.reuse_existing and detail_csv.exists()):
            fold_dir.mkdir(parents=True, exist_ok=True)
            _run_pipeline(
                python_bin=args.python_bin,
                pipeline_script=args.pipeline_script,
                source=args.source,
                fold_year=fold_year,
                out_dir=fold_dir,
                symbols=args.symbols_for_report,
                extra_args=extra_args,
            )
        if not detail_csv.exists():
            raise FileNotFoundError(f"Missing fold detail CSV: {detail_csv}")
        fold_rows.extend(_evaluate_fold(detail_csv, fold_year, scopes))

    aggregate_rows = _aggregate(fold_rows)

    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    _write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "fold_years": fold_years,
                "scopes": [scope.upper() for scope in scopes],
                "fold_metrics": fold_rows,
                "aggregate_metrics": aggregate_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(out_dir / "report.md", fold_rows, aggregate_rows)

    print("\nSaved walk-forward evaluation:")
    print(f"- fold metrics:      {out_dir / 'fold_metrics.csv'}")
    print(f"- aggregate metrics: {out_dir / 'aggregate_metrics.csv'}")
    print(f"- summary json:      {out_dir / 'summary.json'}")
    print(f"- report:            {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
