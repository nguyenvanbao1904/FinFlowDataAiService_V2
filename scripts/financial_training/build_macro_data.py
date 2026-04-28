"""Unified macro data pipeline.

Builds macro_yearly_train.csv (nonbank, 21 cols) and
macro_yearly_train_bank.csv (bank, 16 cols) from:
  - FireAnt API: GDP growth rate, CPI, VNINDEX daily quotes
  - yfinance: commodity prices, FX rates (USD/VND → YoY change %)
  - Manual in-code: interest rates (3 values/year), HRC, iron ore, coal, rubber
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]

# ── Manual data: interest rates (GSO, updated ~once/year) ──────────────
INTEREST_RATES: dict[int, tuple[float, float, float]] = {
    # year: (deposit_12m, loan_short, loan_midlong)
    2013: (7.75, 12.0, 12.0),
    2014: (7.25, 11.0, 11.0),
    2015: (6.60, 8.7, 10.1),
    2016: (6.80, 8.4, 9.7),
    2017: (6.90, 8.9, 10.0),
    2018: (7.10, 8.8, 10.1),
    2019: (7.30, 9.2, 10.5),
    2020: (6.80, 9.1, 10.3),
    2021: (5.90, 8.2, 9.2),
    2022: (6.70, 8.9, 9.9),
    2023: (7.00, 9.3, 11.0),
    2024: (4.90, 6.2, 8.4),
    2025: (6.85, 8.9, 10.1),
}

# ── Manual data: commodities NOT available on yfinance ─────────────────
MANUAL_COMMODITIES: dict[str, dict[int, float]] = {
    "hrc": {
        2013: 630.4655870445345, 2014: 652.8964143426294, 2015: 460.48015873015873,
        2016: 520.5120967741935, 2017: 619.5102880658436, 2018: 829.9243027888447,
        2019: 602.7301587301587, 2020: 579.8849206349206, 2021: 1585.9246031746031,
        2022: 1010.6573705179284, 2023: 910.264, 2024: 774.2142857142857,
        2025: 850.1872509960159,
    },
    "iron_ore": {
        2013: 134.95004052262965, 2014: 97.7503966376895, 2015: 55.12186510600741,
        2016: 56.86868009948731, 2017: 70.50474118236527, 2018: 69.41764940968548,
        2019: 93.0672619532025, 2020: 107.4247824024306, 2021: 160.2831350962321,
        2022: 121.482868665718, 2023: 118.89680010986328, 2024: 110.30039699493892,
        2025: 101.99071706338708,
    },
    "coal": {
        2013: 81.45331833501568, 2014: 75.33066677517361, 2015: 56.70703536852271,
        2016: 59.54097057294242, 2017: 84.439801128737, 2018: 91.9980079073355,
        2019: 61.440873085506375, 2020: 50.27406364228146, 2021: 120.24761922018868,
        2022: 290.0391632323246, 2023: 127.13547970581055, 2024: 112.21746069287497,
        2025: 99.53653246356596,
    },
    "rubber": {
        2013: 126.7586963855286, 2014: 88.7512551858335, 2015: 70.73127889324162,
        2016: 74.48355778307258, 2017: 90.790599336138, 2018: 70.41133516088043,
        2019: 74.83567040706791, 2020: 79.89845493644334, 2021: 94.52003718214632,
        2022: 83.28583361735086, 2023: 71.79592032253716, 2024: 105.48508593855948,
        2025: 100.3953078142242,
    },
}

YFINANCE_TICKERS: dict[str, str] = {
    "usd_vnd": "USDVND=X",
    "gold_gc": "GC=F",
    "oil_brent": "BZ=F",
    "nat_gas": "NG=F",
    "sugar": "SB=F",
    "coffee": "KC=F",
    "rice": "ZR=F",
    "bdry_shipping_etf": "BDRY",
}

LOG_COLUMNS = [
    "gold_gc", "hrc", "iron_ore", "coal",
    "oil_brent", "nat_gas", "sugar", "rubber", "coffee", "rice",
    "bdry_shipping_etf",
]

BANK_EXCLUDE_COLUMNS = [
    "sugar_log",
    "vnindex_daily_return_mean_pct",
    "vnindex_growth_yoy_pct",
    "vnindex_trading_volume_avg",
    "vnindex_trading_value_avg",
]

NONBANK_COLUMN_ORDER = [
    "year",
    "interest_deposit_12m_pct",
    "interest_loan_short_pct",
    "interest_loan_midlong_pct",
    "cpi_inflation_yoy_pp",
    "gdp_growth_yoy_pct",
    "usd_vnd_yoy_pct",
    "gold_gc_log",
    "hrc_log",
    "iron_ore_log",
    "coal_log",
    "oil_brent_log",
    "nat_gas_log",
    "sugar_log",
    "rubber_log",
    "coffee_log",
    "rice_log",
    "bdry_shipping_etf_log",
    "vnindex_daily_return_mean_pct",
    "vnindex_growth_yoy_pct",
    "vnindex_trading_volume_avg",
    "vnindex_trading_value_avg",
]

BANK_COLUMN_ORDER = [c for c in NONBANK_COLUMN_ORDER if c not in BANK_EXCLUDE_COLUMNS]


# ── Fallback data (full precision from historical CSVs) ────────────────

_FALLBACK_GDP_GROWTH: dict[int, float] = {
    # GDP Annual Growth Rate (%), averaged from quarterly data (FireAnt id=2)
    2013: 5.33, 2014: 5.86, 2015: 6.62, 2016: 6.12,
    2017: 6.63, 2018: 7.08, 2019: 6.96, 2020: 2.81,
    2021: 2.66, 2022: 8.13, 2023: 4.91, 2024: 7.05,
    2025: 7.90,
}

_FALLBACK_CPI: dict[int, float] = {
    2013: 6.60, 2014: 4.09, 2015: 0.63, 2016: 2.66,
    2017: 3.53, 2018: 3.54, 2019: 2.79, 2020: 3.23,
    2021: 1.84, 2022: 3.15, 2023: 3.25, 2024: 3.63,
    2025: 3.31,
}

_FALLBACK_YFINANCE: dict[str, dict[int, float]] = {
    "usd_vnd": {
        2012: 20828.0,
        2013: 20783.061973177155, 2014: 20740.505957855577, 2015: 21633.613026819923,
        2016: 22015.172413793105, 2017: 22373.18076923077, 2018: 22942.74217253353,
        2019: 23178.83936298077, 2020: 23227.946594704197, 2021: 22905.417916367336,
        2022: 23399.60769230769, 2023: 23824.184615384616, 2024: 25040.618320610687,
        2025: 25987.416342412453,
    },
    "gold_gc": {
        2013: 1408.6591264028398, 2014: 1265.8174569266182, 2015: 1158.8436502123636,
        2016: 1249.7060024414063, 2017: 1257.6972126143862, 2018: 1267.5996005859374,
        2019: 1392.7222207690045, 2020: 1773.164029690588, 2021: 1797.6690455845423,
        2022: 1800.0681269064366, 2023: 1942.769197265625, 2024: 2390.006343296596,
        2025: 3447.3484090169272,
    },
    "oil_brent": {
        2013: 108.673580593533, 2014: 99.28431988525392, 2015: 53.502261752174014,
        2016: 45.16007995605469, 2017: 54.72601599522321, 2018: 71.7671713810043,
        2019: 64.1606349188184, 2020: 43.086758979224406, 2021: 70.95380932944161,
        2022: 98.96633461272098, 2023: 82.19043816798236, 2024: 79.8213096194797,
        2025: 68.10154755910237,
    },
    "nat_gas": {
        2013: 3.7307222258476984, 2014: 4.262821431197818, 2015: 2.6264523840139784,
        2016: 2.550900002002716, 2017: 3.017509951534499, 2018: 3.064370518186653,
        2019: 2.5264166612473744, 2020: 2.1300592855973677, 2021: 3.7276111046473184,
        2022: 6.541924297097195, 2023: 2.66568924278852, 2024: 2.4085912713928828,
        2025: 3.6197857137710328,
    },
    "sugar": {
        2013: 17.47432539955018, 2014: 16.339285706716872, 2015: 13.117301607888844,
        2016: 18.162360023498536, 2017: 15.780996014872397, 2018: 12.240876509373882,
        2019: 12.343650795164562, 2020: 12.86794467971259, 2021: 17.87265880524166,
        2022: 18.821593630361367, 2023: 24.091593632185127, 2024: 20.743015879676456,
        2025: 16.941825404999747,
    },
    "coffee": {
        2013: 125.86170623415993, 2014: 177.68406364167353, 2015: 132.35396845378574,
        2016: 136.1128000793457, 2017: 132.8962152488678, 2018: 112.71215112085837,
        2019: 101.14126965356252, 2020: 111.01758901121116, 2021: 168.8902776808966,
        2022: 214.50119498811395, 2023: 172.45677282325775, 2024: 235.45535671900188,
        2025: 367.87738121880426,
    },
    "rice": {
        2013: 1546.720238095238, 2014: 1393.678571428571, 2015: 1105.8789682539682,
        2016: 1034.672, 2017: 1106.3725099601593, 2018: 1154.6454183266933,
        2019: 1133.678571428571, 2020: 1360.5691699604745, 2021: 1334.218253968254,
        2022: 1650.848605577689, 2023: 1699.082, 2024: 1643.457083331214,
        2025: 1229.3107569721114,
    },
    "bdry_shipping_etf": {
        # BDRY ETF only available from 2018; 2013-2017 filled with first available value
        2013: 21.714806118789983, 2014: 21.714806118789983, 2015: 21.714806118789983,
        2016: 21.714806118789983, 2017: 21.714806118789983, 2018: 21.714806118789983,
        2019: 15.133607168046256, 2020: 7.562648242641344, 2021: 23.710765853760737,
        2022: 16.163227117394072, 2023: 7.137099988937378, 2024: 11.011972213548328,
        2025: 6.991152029037476,
    },
}

# First available BDRY value, used to fill years before the ETF existed
_BDRY_FILL_VALUE = 21.714806118789983


def _fireant_headers() -> dict[str, str]:
    token = os.environ.get("FIREANT_ACCESS_TOKEN", "")
    if not token:
        try:
            from app.core.config import settings
            token = getattr(settings, "FIREANT_ACCESS_TOKEN", "")
        except Exception:
            pass
    return {"Authorization": f"Bearer {token}"}


def _fireant_base() -> str:
    return os.environ.get("FIREANT_API_BASE", "https://restv2.fireant.vn")


# ── FireAnt: GDP Growth Rate ──────────────────────────────────────────

def fetch_gdp_growth_fireant(from_year: int = 2000) -> dict[int, float]:
    """Fetch GDP Annual Growth Rate (id=2) from FireAnt.

    FireAnt returns quarterly growth rates (Date format "Q1/YY").
    We compute the yearly average to get annual GDP growth %.
    Returns {year: avg_growth_pct}.
    """
    url = f"{_fireant_base()}/macro-data/GDP/info"
    resp = requests.get(url, headers=_fireant_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    gdp_item = None
    for item in data:
        if item.get("id") == 2 and item.get("name") == "GDP Annual Growth Rate":
            gdp_item = item
            break
    if gdp_item is None:
        raise ValueError("GDP Annual Growth Rate item (id=2) not found in FireAnt response")

    by_year: dict[int, list[float]] = {}
    for hv in gdp_item.get("historicalValue", []):
        try:
            raw = str(hv["Date"])
            parts = raw.split("/")
            yr_short = int(parts[-1])
            year = 2000 + yr_short if yr_short < 100 else yr_short
            if year >= from_year:
                by_year.setdefault(year, []).append(float(hv["Value"]))
        except (ValueError, TypeError, KeyError, IndexError):
            continue

    return {yr: sum(vals) / len(vals) for yr, vals in by_year.items() if vals}


# ── FireAnt: CPI ───────────────────────────────────────────────────────

def fetch_cpi_fireant() -> dict[int, float]:
    """Fetch Inflation Rate YoY (id=34) from FireAnt.

    FireAnt returns monthly YoY inflation rates (Date format "M/YY").
    We compute the yearly average to match GSO's annual CPI index convention.
    Returns {year: avg_inflation_pp}.
    """
    url = f"{_fireant_base()}/macro-data/Prices/info"
    resp = requests.get(url, headers=_fireant_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    cpi_item = None
    for item in data:
        if item.get("id") == 34:
            cpi_item = item
            break
    if cpi_item is None:
        raise ValueError("Inflation Rate item (id=34) not found")

    by_year: dict[int, list[float]] = {}
    for hv in cpi_item.get("historicalValue", []):
        try:
            raw = str(hv["Date"])
            parts = raw.split("/")
            yr_short = int(parts[-1])
            year = 2000 + yr_short if yr_short < 100 else yr_short
            by_year.setdefault(year, []).append(float(hv["Value"]))
        except (ValueError, TypeError, KeyError, IndexError):
            continue

    return {yr: sum(vals) / len(vals) for yr, vals in by_year.items() if vals}


# ── FireAnt: VNINDEX daily quotes ──────────────────────────────────────

def fetch_vnindex_daily_fireant(start_date: str, end_date: str) -> pd.DataFrame:
    url = f"{_fireant_base()}/symbols/VNINDEX/historical-quotes"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "offset": 0,
        "limit": 10000,
    }
    resp = requests.get(url, headers=_fireant_headers(), params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No VNINDEX data from FireAnt")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ("priceClose", "totalVolume", "totalValue"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={
        "priceClose": "close",
        "totalVolume": "volume",
        "totalValue": "value",
    })
    return df.sort_values("date").reset_index(drop=True)


def build_vnindex_yearly(df: pd.DataFrame, use_real_value: bool = False) -> pd.DataFrame:
    out = df.dropna(subset=["date", "close", "volume"]).copy()
    out["year"] = out["date"].dt.year.astype(int)
    out["daily_return"] = out["close"].pct_change()

    # Use close*volume as proxy for trading value (matches vnstock convention used in training).
    # FireAnt totalValue is in VND (đồng) and ~20x larger than the proxy.
    if use_real_value and "value" in out.columns:
        out["trading_value"] = pd.to_numeric(out["value"], errors="coerce")
    else:
        out["trading_value"] = out["close"] * out["volume"]

    g = out.groupby("year", observed=False)
    yearly = pd.DataFrame({
        "year": sorted(out["year"].unique()),
        "vnindex_close_last": g["close"].last().values,
        "vnindex_daily_return_mean_pct": g["daily_return"].mean().values * 100.0,
        "vnindex_trading_volume_avg": g["volume"].mean().values,
        "vnindex_trading_value_avg": g["trading_value"].mean().values,
    }).sort_values("year").reset_index(drop=True)

    yearly["vnindex_growth_yoy_pct"] = yearly["vnindex_close_last"].pct_change() * 100.0
    return yearly.drop(columns=["vnindex_close_last"])


# ── yfinance: commodities and FX ──────────────────────────────────────

def fetch_yfinance_annual(
    tickers: dict[str, str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    import yfinance as yf

    start = f"{start_year}-01-01"
    end = f"{end_year + 1}-01-01"
    rows: list[dict] = []

    for col_name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             auto_adjust=False, progress=False)
            if df is None or df.empty:
                print(f"  [WARN] yfinance: no data for {ticker} ({col_name})")
                continue
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(str(c) for c in col if c != "").strip("_")
                              for col in df.columns]
            close_col = None
            for c in df.columns:
                cl = str(c).lower().replace("_", "").replace(" ", "")
                if cl in ("close", "adjclose", f"close{ticker.lower()}"):
                    close_col = c
                    break
            if close_col is None:
                candidates = [c for c in df.columns if "close" in str(c).lower()]
                close_col = candidates[0] if candidates else None
            if close_col is None:
                print(f"  [WARN] yfinance: no close column for {ticker}")
                continue
            df["_close"] = pd.to_numeric(df[close_col], errors="coerce")
            date_col = [c for c in df.columns if str(c).lower() in ("date", "datetime")][0]
            df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
            annual = df.groupby("_year")["_close"].mean()
            for yr, val in annual.items():
                if pd.notna(val):
                    match = [r for r in rows if r["year"] == int(yr)]
                    if match:
                        match[0][col_name] = float(val)
                    else:
                        rows.append({"year": int(yr), col_name: float(val)})
        except Exception as e:
            print(f"  [WARN] yfinance error {ticker} ({col_name}): {e}")

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True) if rows else pd.DataFrame()


# ── Assembly ──────────────────────────────────────────────────────────

def build_macro_yearly(
    years: list[int],
    gdp_growth: dict[int, float],
    cpi: dict[int, float],
    yf_df: pd.DataFrame,
    vnindex_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    # Pre-compute USD/VND YoY % from annual average FX rates
    usd_vnd_by_year: dict[int, float] = {}
    if not yf_df.empty and "usd_vnd" in yf_df.columns:
        for _, r in yf_df.iterrows():
            yr_val = int(r["year"])
            val = r["usd_vnd"]
            if pd.notna(val):
                usd_vnd_by_year[yr_val] = float(val)

    for yr in years:
        row: dict = {"year": yr}

        if yr in INTEREST_RATES:
            dep, short, midlong = INTEREST_RATES[yr]
            row["interest_deposit_12m_pct"] = dep
            row["interest_loan_short_pct"] = short
            row["interest_loan_midlong_pct"] = midlong

        if yr in cpi:
            row["cpi_inflation_yoy_pp"] = cpi[yr]

        if yr in gdp_growth:
            row["gdp_growth_yoy_pct"] = gdp_growth[yr]

        # USD/VND YoY change %
        if yr in usd_vnd_by_year and (yr - 1) in usd_vnd_by_year:
            prev = usd_vnd_by_year[yr - 1]
            if prev != 0:
                row["usd_vnd_yoy_pct"] = ((usd_vnd_by_year[yr] / prev) - 1.0) * 100.0

        yf_row = yf_df[yf_df["year"] == yr] if not yf_df.empty else pd.DataFrame()
        if not yf_row.empty:
            for c in yf_df.columns:
                if c not in ("year", "usd_vnd"):
                    val = yf_row.iloc[0][c]
                    if pd.notna(val):
                        row[c] = float(val)

        for name, data in MANUAL_COMMODITIES.items():
            if yr in data:
                row[name] = data[yr]

        # Fill BDRY with first-available value for years before the ETF existed
        if "bdry_shipping_etf" not in row:
            row["bdry_shipping_etf"] = _BDRY_FILL_VALUE

        vn_row = vnindex_df[vnindex_df["year"] == yr] if not vnindex_df.empty else pd.DataFrame()
        if not vn_row.empty:
            for c in ["vnindex_daily_return_mean_pct", "vnindex_growth_yoy_pct",
                       "vnindex_trading_volume_avg", "vnindex_trading_value_avg"]:
                if c in vn_row.columns:
                    val = vn_row.iloc[0][c]
                    if pd.notna(val):
                        row[c] = float(val)

        records.append(row)

    df = pd.DataFrame(records)

    for col in LOG_COLUMNS:
        if col in df.columns:
            df[f"{col}_log"] = np.log(pd.to_numeric(df[col], errors="coerce"))
            df = df.drop(columns=[col])

    return df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build macro yearly CSVs for ML training.")
    p.add_argument("--start-year", type=int, default=2013)
    p.add_argument("--end-year", type=int, default=date.today().year - 1,
                   help="Last year to include (default: previous year if current year incomplete)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "macro")
    p.add_argument("--skip-fireant", action="store_true",
                   help="Skip FireAnt API calls, use fallback data")
    p.add_argument("--skip-yfinance", action="store_true",
                   help="Skip yfinance API calls, use fallback data")
    p.add_argument("--include-current-year", action="store_true",
                   help="Include current year even if incomplete")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    end_year = args.end_year
    if args.include_current_year:
        end_year = max(end_year, date.today().year)
    years = list(range(args.start_year, end_year + 1))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) GDP Growth Rate
    if args.skip_fireant:
        gdp_growth = _FALLBACK_GDP_GROWTH
        print("GDP growth: using fallback data")
    else:
        try:
            gdp_growth = fetch_gdp_growth_fireant(from_year=2000)
            if gdp_growth:
                print(f"GDP growth: {len(gdp_growth)} years from FireAnt")
            else:
                print("GDP growth: empty from FireAnt, using fallback")
                gdp_growth = _FALLBACK_GDP_GROWTH
        except Exception as e:
            print(f"GDP growth: FireAnt failed ({e}), using fallback")
            gdp_growth = _FALLBACK_GDP_GROWTH

    # 2) CPI
    if args.skip_fireant:
        cpi = _FALLBACK_CPI
        print("CPI: using fallback data")
    else:
        try:
            cpi = fetch_cpi_fireant()
            if cpi:
                print(f"CPI: {len(cpi)} years from FireAnt")
            else:
                print("CPI: empty from FireAnt, using fallback")
                cpi = _FALLBACK_CPI
        except Exception as e:
            print(f"CPI: FireAnt failed ({e}), using fallback")
            cpi = _FALLBACK_CPI

    # 3) yfinance (fetch from start_year-1 to get USD/VND YoY for start_year)
    yf_years = [args.start_year - 1] + years
    if args.skip_yfinance:
        yf_records = []
        for yr in yf_years:
            row: dict = {"year": yr}
            for name, data in _FALLBACK_YFINANCE.items():
                if yr in data:
                    row[name] = data[yr]
            yf_records.append(row)
        yf_df = pd.DataFrame(yf_records)
        print("yfinance: using fallback data")
    else:
        print("Fetching yfinance data...")
        yf_df = fetch_yfinance_annual(YFINANCE_TICKERS, args.start_year - 1, end_year)
        if yf_df.empty:
            print("yfinance: empty, using fallback")
            yf_records = []
            for yr in yf_years:
                row = {"year": yr}
                for name, data in _FALLBACK_YFINANCE.items():
                    if yr in data:
                        row[name] = data[yr]
                yf_records.append(row)
            yf_df = pd.DataFrame(yf_records)
        else:
            print(f"yfinance: {len(yf_df)} years fetched")
            # Fill gaps from fallback (BDRY pre-2018, etc.)
            for yr in yf_years:
                yr_rows = yf_df[yf_df["year"] == yr]
                if yr_rows.empty:
                    row = {"year": yr}
                    for name, data in _FALLBACK_YFINANCE.items():
                        if yr in data:
                            row[name] = data[yr]
                    yf_df = pd.concat([yf_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    idx = yr_rows.index[0]
                    for name, data in _FALLBACK_YFINANCE.items():
                        col_missing = name not in yf_df.columns or pd.isna(yf_df.at[idx, name])
                        if col_missing and yr in data:
                            yf_df.at[idx, name] = data[yr]
            yf_df = yf_df.sort_values("year").reset_index(drop=True)

    # 4) VNINDEX
    if args.skip_fireant:
        vnindex_df = pd.DataFrame()
        print("VNINDEX: skipped (--skip-fireant)")
    else:
        try:
            start_date = f"{args.start_year - 1}-01-01"
            end_date = date.today().isoformat()
            print(f"Fetching VNINDEX daily quotes ({start_date} to {end_date})...")
            vnindex_daily = fetch_vnindex_daily_fireant(start_date, end_date)
            vnindex_df = build_vnindex_yearly(vnindex_daily)
            print(f"VNINDEX: {len(vnindex_df)} years built")
        except Exception as e:
            print(f"VNINDEX: FireAnt failed ({e})")
            vnindex_df = pd.DataFrame()

    # 5) Assemble
    df = build_macro_yearly(years, gdp_growth, cpi, yf_df, vnindex_df)

    # 6) Output
    nonbank_cols = [c for c in NONBANK_COLUMN_ORDER if c in df.columns]
    bank_cols = [c for c in BANK_COLUMN_ORDER if c in df.columns]

    nonbank_out = df[nonbank_cols].sort_values("year").reset_index(drop=True)
    bank_out = df[bank_cols].sort_values("year").reset_index(drop=True)

    nonbank_csv = out_dir / "macro_yearly_train.csv"
    bank_csv = out_dir / "macro_yearly_train_bank.csv"
    nonbank_out.to_csv(nonbank_csv, index=False)
    bank_out.to_csv(bank_csv, index=False)

    print(f"\nSaved: {nonbank_csv} ({len(nonbank_out)} rows, {len(nonbank_out.columns)} cols)")
    print(f"Saved: {bank_csv} ({len(bank_out)} rows, {len(bank_out.columns)} cols)")

    missing = [c for c in NONBANK_COLUMN_ORDER if c not in df.columns]
    if missing:
        print(f"[WARN] Missing nonbank columns: {missing}")

    print("\nNonbank (last 3 rows):")
    print(nonbank_out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
