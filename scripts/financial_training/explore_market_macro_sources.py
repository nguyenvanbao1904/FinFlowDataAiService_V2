from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SourceResult:
    source: str
    ok: bool
    rows: int
    has_close: bool
    has_volume: bool
    has_value: bool
    message: str
    sample_columns: list[str]


def _to_date_str(x: str | date) -> str:
    if isinstance(x, date):
        return x.isoformat()
    return str(x)


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(c) for c in col if str(c) != ""]).strip("_") for col in out.columns]

    rename_map = {}
    for c in out.columns:
        cl = str(c).strip().lower()
        if cl in {"date", "datetime", "time", "timestamp"}:
            rename_map[c] = "date"
        elif cl in {"close", "adj close", "adj_close", "close_", "close_vnindex", "close_^vnindex"}:
            rename_map[c] = "close"
        elif cl in {"volume", "vol", "total_volume"}:
            rename_map[c] = "volume"
        elif cl in {"value", "trading_value", "total_value"}:
            rename_map[c] = "value"

    out = out.rename(columns=rename_map)
    if "date" not in out.columns and out.index.name is not None:
        out = out.reset_index()
        if str(out.columns[0]).lower() in {"date", "datetime", "time", "timestamp"}:
            out = out.rename(columns={out.columns[0]: "date"})
    elif "date" not in out.columns and not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index().rename(columns={out.columns[0]: "date"})

    for c in ("close", "volume", "value"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])

    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    return out


def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No rows from yfinance for {symbol}")
    df = df.reset_index()
    return _normalize_daily_frame(df)


def _fetch_stooq(symbol: str, start: str, end: str) -> pd.DataFrame:
    from pandas_datareader import data as web

    df = web.DataReader(symbol, "stooq", start, end)
    if df is None or df.empty:
        raise ValueError(f"No rows from stooq for {symbol}")
    df = df.reset_index()
    return _normalize_daily_frame(df)


def _fetch_vnstock_world_index(symbol: str, start: str, end: str) -> pd.DataFrame:
    from vnstock import Vnstock

    wi = Vnstock().world_index(symbol=symbol, source="MSN")
    errors: list[str] = []
    for kwargs in (
        {"start": start, "end": end, "interval": "1D"},
        {"start": start, "end": end},
    ):
        try:
            df = wi.quote.history(**kwargs)
            if df is not None and not df.empty:
                return _normalize_daily_frame(df)
        except Exception as ex:  # pragma: no cover - network/provider behavior
            errors.append(f"kwargs={kwargs}: {type(ex).__name__}: {ex}")
    raise RuntimeError("; ".join(errors) if errors else "No rows from vnstock world_index")


def _fetch_vnstock_stock_index(symbol: str, start: str, end: str) -> pd.DataFrame:
    from vnstock import Vnstock

    idx = Vnstock().stock(symbol=symbol, source="VCI")
    errors: list[str] = []
    for kwargs in (
        {"start": start, "end": end, "interval": "1D"},
        {"start": start, "end": end},
    ):
        try:
            df = idx.quote.history(**kwargs)
            if df is not None and not df.empty:
                return _normalize_daily_frame(df)
        except Exception as ex:  # pragma: no cover - network/provider behavior
            errors.append(f"kwargs={kwargs}: {type(ex).__name__}: {ex}")
    raise RuntimeError("; ".join(errors) if errors else "No rows from vnstock stock index")


def _build_yearly_features(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "date" not in df.columns or "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("Need at least date/close/volume to build yearly market features")

    out = df.copy()
    out = out.dropna(subset=["date", "close", "volume"]).reset_index(drop=True)
    out["year"] = out["date"].dt.year.astype(int)

    has_real_value = "value" in out.columns and out["value"].notna().any()
    if has_real_value:
        out["trading_value"] = pd.to_numeric(out["value"], errors="coerce")
        value_note = "reported"
    else:
        out["trading_value"] = out["close"] * out["volume"]
        value_note = "proxy_close_x_volume"

    out["daily_return"] = out["close"].pct_change()

    g = out.groupby("year", observed=False)
    yearly = pd.DataFrame(
        {
            "year": sorted(out["year"].unique().tolist()),
            "vnindex_close_avg": g["close"].mean().values,
            "vnindex_close_last": g["close"].last().values,
            "vnindex_daily_return_mean_pct": g["daily_return"].mean().values * 100.0,
            "vnindex_trading_volume_avg": g["volume"].mean().values,
            "vnindex_trading_volume_sum": g["volume"].sum().values,
            "vnindex_trading_value_avg": g["trading_value"].mean().values,
            "vnindex_trading_value_sum": g["trading_value"].sum().values,
        }
    )
    yearly = yearly.sort_values("year", kind="mergesort").reset_index(drop=True)
    yearly["vnindex_growth_yoy_pct"] = yearly["vnindex_close_last"].pct_change() * 100.0
    yearly["source"] = source_name
    yearly["trading_value_type"] = value_note
    return yearly


def _merge_into_macro_csv(
    base_macro_csv: Path,
    yearly_df: pd.DataFrame,
    from_year: int,
    output_csv: Path,
) -> pd.DataFrame:
    base = pd.read_csv(base_macro_csv)
    if "year" not in base.columns:
        raise ValueError(f"Base macro CSV must have 'year': {base_macro_csv}")

    base = base.copy()
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["year"]).astype({"year": int})

    add_cols = [
        "vnindex_daily_return_mean_pct",
        "vnindex_growth_yoy_pct",
        "vnindex_trading_volume_avg",
        "vnindex_trading_value_avg",
    ]

    ext = yearly_df.copy()
    ext = ext[ext["year"] >= int(from_year)]
    keep_cols = ["year"] + [c for c in add_cols if c in ext.columns]
    ext = ext[keep_cols].drop_duplicates(subset=["year"], keep="last")

    merged = base.merge(ext, on="year", how="left", suffixes=("", "_new"))
    for c in add_cols:
        new_c = f"{c}_new"
        if new_c in merged.columns:
            merged[c] = pd.to_numeric(merged[new_c], errors="coerce").where(
                merged[new_c].notna(), pd.to_numeric(merged.get(c), errors="coerce")
            )
            merged = merged.drop(columns=[new_c])

    # Keep stable order: existing base columns first, then new features if not present.
    base_cols = list(pd.read_csv(base_macro_csv, nrows=1).columns)
    final_cols = [c for c in base_cols if c in merged.columns]
    for c in add_cols:
        if c not in final_cols and c in merged.columns:
            final_cols.append(c)
    merged = merged[final_cols].sort_values("year", kind="mergesort").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def _probe_source(
    name: str,
    fetcher: Callable[[str, str, str], pd.DataFrame],
    symbol: str,
    start: str,
    end: str,
) -> tuple[SourceResult, pd.DataFrame | None]:
    try:
        df = fetcher(symbol, start, end)
        res = SourceResult(
            source=name,
            ok=bool(df is not None and not df.empty),
            rows=int(len(df)),
            has_close=bool("close" in df.columns),
            has_volume=bool("volume" in df.columns),
            has_value=bool("value" in df.columns),
            message="ok",
            sample_columns=[str(c) for c in df.columns[:20]],
        )
        return res, df
    except Exception as ex:  # pragma: no cover - provider/network behavior
        res = SourceResult(
            source=name,
            ok=False,
            rows=0,
            has_close=False,
            has_volume=False,
            has_value=False,
            message=f"{type(ex).__name__}: {ex}",
            sample_columns=[],
        )
        return res, None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore candidate data sources for VNINDEX growth/liquidity annual macro features."
    )
    parser.add_argument("--start", type=str, default="2012-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "artifacts" / "macro" / "vnindex_liquidity_yearly_explore.csv",
        help="Output yearly feature CSV",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=ROOT / "artifacts" / "macro" / "vnindex_liquidity_sources_report.json",
        help="Output source probing report JSON",
    )
    parser.add_argument(
        "--preferred-source",
        type=str,
        default="auto",
        choices=[
            "auto",
            "vnstock_world_index",
            "vnstock_stock_index",
            "yfinance_^VNINDEX",
            "yfinance_VNINDEX.VN",
            "stooq_^VNINDEX",
            "stooq_VNINDEX",
        ],
        help="Force a specific source or let script auto-select the first valid one",
    )
    parser.add_argument(
        "--merge-into-macro-csv",
        type=Path,
        default=None,
        help="If provided, merge yearly VNINDEX features into this macro CSV.",
    )
    parser.add_argument(
        "--merge-output-csv",
        type=Path,
        default=None,
        help="Optional output path for merged macro CSV. Default: overwrite --merge-into-macro-csv",
    )
    parser.add_argument(
        "--merge-from-year",
        type=int,
        default=2013,
        help="Lower bound year for VNINDEX features when merging into macro CSV.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    start = _to_date_str(args.start)
    end = _to_date_str(args.end)

    probes: list[tuple[str, Callable[[str, str, str], pd.DataFrame], str]] = [
        ("vnstock_world_index", _fetch_vnstock_world_index, "^VNINDEX"),
        ("vnstock_stock_index", _fetch_vnstock_stock_index, "VNINDEX"),
        ("yfinance_^VNINDEX", _fetch_yfinance, "^VNINDEX"),
        ("yfinance_VNINDEX.VN", _fetch_yfinance, "VNINDEX.VN"),
        ("stooq_^VNINDEX", _fetch_stooq, "^VNINDEX"),
        ("stooq_VNINDEX", _fetch_stooq, "VNINDEX"),
    ]

    source_results: list[SourceResult] = []
    frames: dict[str, pd.DataFrame] = {}

    for name, fetcher, symbol in probes:
        res, df = _probe_source(name=name, fetcher=fetcher, symbol=symbol, start=start, end=end)
        source_results.append(res)
        if df is not None and not df.empty:
            frames[name] = df

    selected_name = None
    if args.preferred_source != "auto":
        selected_name = args.preferred_source if args.preferred_source in frames else None
    else:
        for item in source_results:
            if item.ok and item.has_close and item.has_volume and item.source in frames:
                selected_name = item.source
                break

    yearly_df = None
    merged_macro_df = None
    merged_macro_path: str | None = None
    if selected_name is not None:
        yearly_df = _build_yearly_features(frames[selected_name], selected_name)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        yearly_df.to_csv(args.out_csv, index=False)

        if args.merge_into_macro_csv is not None:
            out_macro = args.merge_output_csv or args.merge_into_macro_csv
            merged_macro_df = _merge_into_macro_csv(
                base_macro_csv=args.merge_into_macro_csv,
                yearly_df=yearly_df,
                from_year=int(args.merge_from_year),
                output_csv=out_macro,
            )
            merged_macro_path = str(out_macro)

    report = {
        "request": {
            "start": start,
            "end": end,
            "preferred_source": args.preferred_source,
        },
        "selected_source": selected_name,
        "selected_output_csv": str(args.out_csv) if yearly_df is not None else None,
        "merged_macro_csv": merged_macro_path,
        "sources": [
            {
                "source": r.source,
                "ok": r.ok,
                "rows": r.rows,
                "has_close": r.has_close,
                "has_volume": r.has_volume,
                "has_value": r.has_value,
                "message": r.message,
                "sample_columns": r.sample_columns,
            }
            for r in source_results
        ],
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved source report:", args.out_report)
    if yearly_df is not None:
        print("Saved yearly features:", args.out_csv)
        print("Selected source:", selected_name)
        print(yearly_df.tail(5).to_string(index=False))
        if merged_macro_df is not None:
            print("Merged into macro CSV:", merged_macro_path)
            print(merged_macro_df.tail(5).to_string(index=False))
    else:
        print("No source returned usable close+volume data. See report for details.")


if __name__ == "__main__":
    main()
