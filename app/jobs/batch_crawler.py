import logging
import json
import time
import os
import re
import asyncio

# --- INJECT API KEY EARLY ---
from dotenv import load_dotenv

load_dotenv()
if os.getenv("VNSTOCK_API_KEY"):
    os.environ["VNSTOCK_API_KEY"] = os.getenv("VNSTOCK_API_KEY")

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
from app.domain.entities.investment import CompanyModel
from app.services.icb_tree_sync import build_industry_node_payloads
from app.services.crawler_service import VnstockCrawlerService
from app.clients.java_backend_client import JavaBackendClient

logging.basicConfig(level=logging.WARNING)
log_lock = Lock()

STATE_FILE = "crawler_state.json"
MAX_WORKERS = 8  # Tăng lên 8 nhờ API Key Community (60 req/phút)
RETRY_MAX = 3
RATE_LIMIT_DEFAULT = 15
SAFE_DELAY = 1


def log(msg):
    prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
    with log_lock:
        print(f"{prefix}{msg}")


def load_state():
    """Tải trạng thái crawl để skip các mã đã thành công."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"successful": []}


def save_state(state):
    with log_lock:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)


def run_crawler_for_symbol(sym_data: tuple, manager_dict, api_lock):
    """Tiến trình crawl cho 1 mã chứng khoán."""
    symbol, is_bank, exchange_group = sym_data
    crawler = VnstockCrawlerService()
    errors = []

    debug_symbols_env = os.getenv("DEBUG_COMPANY_META_SYMBOLS", "")
    debug_symbols = (
        {s.strip().upper() for s in debug_symbols_env.split(",") if s.strip()}
        if debug_symbols_env
        else set()
    )

    # --- SAFE REQUEST WRAPPER ---
    def safe_request(func, *args):
        retries = 0
        while retries < RETRY_MAX:
            with api_lock:
                current_time = time.time()
                time_to_wait = max(0, manager_dict["rate_limit_wait_until"] - current_time)
                if time_to_wait > 0:
                    log(
                        f"[{symbol}] Đang chờ {time_to_wait:.0f}s do Rate Limit từ tiến trình khác..."
                    )
                    time.sleep(time_to_wait)
                manager_dict["rate_limit_wait_until"] = time.time() + SAFE_DELAY

            try:
                res, warnings = func(*args)

                # VCI sometimes returns empty on rate limit
                for w in warnings:
                    if "tối đa" in w.lower() or "limit" in w.lower():
                        raise Exception("Rate limit exceeded")
                return res, warnings

            except (Exception, SystemExit) as e:
                err_msg = str(e)
                # Catch generic rate limit texts from vnstock's stack traces or prints, or if SystemExit is raised by vnstock
                if (
                    isinstance(e, SystemExit)
                    or "tối đa" in err_msg.lower()
                    or "limit" in err_msg.lower()
                    or "thử lại" in err_msg.lower()
                    or "Too Many Requests" in err_msg
                ):
                    match = re.search(r"thử lại sau (\d+) giây", err_msg)
                    wait_time = int(match.group(1)) if match else RATE_LIMIT_DEFAULT
                    with api_lock:
                        new_wait = time.time() + wait_time + 5  # Thêm 5s buffer
                        if new_wait > manager_dict["rate_limit_wait_until"]:
                            manager_dict["rate_limit_wait_until"] = new_wait
                            log(
                                f"🚨 RATE LIMIT DETECTED! Hệ thống đang chờ {wait_time + 5}s trước khi chạy tiếp..."
                            )
                else:
                    retries += 1
                    time.sleep(2**retries)
                    log(f"[{symbol}] Lỗi: {err_msg}. Thử lại lần {retries}/{RETRY_MAX}...")

        return [], [f"Failed after {RETRY_MAX} retries"]

    # 1. Indicators
    inds, w_inds = safe_request(crawler.get_financial_indicators, symbol, is_bank)
    if not inds:
        errors.append(f"Indicators failed: {w_inds}")

    # 2. Income Statement
    incomes, w_inc = safe_request(crawler.get_income_statement, symbol, is_bank)
    if not incomes:
        errors.append(f"Income failed: {w_inc}")

    # 3. Balance Sheet
    balances, w_bal = safe_request(crawler.get_balance_sheet, symbol, is_bank)
    if not balances:
        errors.append(f"Balance failed: {w_bal}")

    # 4. Shareholders
    shareholders, w_sh = safe_request(crawler.get_company_shareholders, symbol)
    if not shareholders and w_sh:
        errors.append(f"Shareholders failed: {w_sh}")

    # 5. Dividends
    dividends, w_div = safe_request(crawler.get_company_dividends, symbol)
    if not dividends and w_div:
        errors.append(f"Dividends failed: {w_div}")

    # Company meta (tên, mô tả, ICB → resolve industry_node_id trên backend)
    company_meta, w_meta = safe_request(crawler.get_company_overview_meta, symbol)

    if symbol.upper() in debug_symbols:
        log(f"[{symbol}] company_meta={company_meta} warnings={w_meta}")

    # --- MAP TO DTOs AND PUSH TO JAVA BACKEND ---
    client = JavaBackendClient()

    async def push_all():
        balance_sheets = []
        income_stmts = []
        financial_inds = []

        icb_code = (company_meta.get("icbCode") or "").strip() if company_meta else ""
        company_name = company_meta.get("companyName") if company_meta else None
        description = company_meta.get("description") if company_meta else None

        # Luôn đẩy company (tên/mô tả/icb) trước — không phụ thuộc chỉ báo tài chính (để gán industry_node_id).
        if company_name is not None or description is not None or icb_code:
            companies_payload = [
                CompanyModel(
                    id=symbol,
                    exchange=exchange_group,
                    industryIcbCode=icb_code or None,
                    companyName=company_name,
                    description=description,
                    companyType="BANK" if is_bank else "NON_BANK",
                ).model_dump()
            ]
            await client.push_data("companies", companies_payload)

        if errors:
            return

        # Process Financial Indicators (separate logic for BANK vs NON-BANK)
        if inds:
            for ind in inds:
                # Helper: convert optional numeric to float (keeps 0 values)
                def f(v):
                    return float(v) if v is not None else None

                base_dto = {
                    "companyId": symbol,
                    "year": ind.year if hasattr(ind, "year") else 2026,
                    "quarter": ind.quarter if hasattr(ind, "quarter") else 1,
                    "pe": f(getattr(ind, "pe", None)),
                    "pb": f(getattr(ind, "pb", None)),
                    "ps": f(getattr(ind, "ps", None)),
                    "roe": f(getattr(ind, "roe", None)),
                    "roa": f(getattr(ind, "roa", None)),
                    "eps": f(getattr(ind, "eps", None)),
                    "bvps": f(getattr(ind, "bvps", None)),
                    "cplh": f(getattr(ind, "cplh", None)),
                }

                # Add bank or non-bank specific fields
                if is_bank:
                    # Bank only: NO lng, lnr
                    financial_inds.append(base_dto)
                else:
                    # Non-bank: add lng, lnr
                    base_dto["lng"] = f(getattr(ind, "lng", None))
                    base_dto["lnr"] = f(getattr(ind, "lnr", None))
                    financial_inds.append(base_dto)

        # Process Income Statements
        if incomes:
            for inc in incomes:
                def f(v):
                    return float(v) if v is not None else None

                if is_bank:
                    # Map only fields that exist in Java DTO to avoid unknown-property issues
                    dto = {
                        "companyId": symbol,
                        "year": inc.year if hasattr(inc, "year") else 2026,
                        "quarter": inc.quarter if hasattr(inc, "quarter") else 1,
                        "profitAfterTax": f(getattr(inc, "profitAfterTax", None)),
                        "netInterestIncome": f(getattr(inc, "netInterestIncome", None)),
                        "netFeeAndCommissionIncome": f(
                            getattr(inc, "netFeeAndCommissionIncome", None)
                        ),
                        "netOtherIncomeOrExpenses": f(
                            getattr(inc, "netOtherIncomeOrExpenses", None)
                        ),
                        # pydantic field is "interestAndSimilarExpenses"; DTO expects "interestExpense"
                        "interestExpense": f(getattr(inc, "interestAndSimilarExpenses", None)),
                        "netProfit": f(getattr(inc, "netProfit", None)),
                    }
                else:
                    dto = {
                        "companyId": symbol,
                        "year": inc.year if hasattr(inc, "year") else 2026,
                        "quarter": inc.quarter if hasattr(inc, "quarter") else 1,
                        "profitAfterTax": f(getattr(inc, "profitAfterTax", None)),
                        "netRevenue": f(getattr(inc, "netRevenue", None)),
                        "totalRevenue": f(getattr(inc, "totalRevenue", None)),
                        "netProfit": f(getattr(inc, "netProfit", None)),
                    }
                income_stmts.append(dto)

        # Process Balance Sheets
        if balances:
            for bal in balances:
                def f(v):
                    return float(v) if v is not None else None

                if is_bank:
                    dto = {
                        "companyId": symbol,
                        "year": bal.year if hasattr(bal, "year") else 2026,
                        "quarter": bal.quarter if hasattr(bal, "quarter") else 1,
                        "cashAndCashEquivalents": f(
                            getattr(bal, "cashAndCashEquivalents", None)
                        ),
                        "totalAssets": f(getattr(bal, "totalAssets", None)),
                        "equity": f(getattr(bal, "equity", None)),
                        "totalCapital": f(getattr(bal, "totalCapital", None)),
                        "totalLiabilities": f(getattr(bal, "totalLiabilities", None)),
                        "balancesWithSbv": f(getattr(bal, "balancesWithSbv", None)),
                        "interbankPlacementsAndLoans": f(
                            getattr(bal, "interbankPlacementsAndLoans", None)
                        ),
                        "tradingSecurities": f(getattr(bal, "tradingSecurities", None)),
                        "investmentSecurities": f(
                            getattr(bal, "investmentSecurities", None)
                        ),
                        "loansToCustomers": f(getattr(bal, "loansToCustomers", None)),
                        "govAndSbvDebt": f(getattr(bal, "govAndSbvDebt", None)),
                        "depositsBorrowingsOthers": f(
                            getattr(bal, "depositsBorrowingsOthers", None)
                        ),
                        "depositsFromCustomers": f(
                            getattr(bal, "depositsFromCustomers", None)
                        ),
                        "convertibleAndOtherPapers": f(
                            getattr(bal, "convertibleAndOtherPapers", None)
                        ),
                    }
                else:
                    dto = {
                        "companyId": symbol,
                        "year": bal.year if hasattr(bal, "year") else 2026,
                        "quarter": bal.quarter if hasattr(bal, "quarter") else 1,
                        "cashAndCashEquivalents": f(
                            getattr(bal, "cashAndCashEquivalents", None)
                        ),
                        "totalAssets": f(getattr(bal, "totalAssets", None)),
                        "equity": f(getattr(bal, "equity", None)),
                        "totalCapital": f(getattr(bal, "totalCapital", None)),
                        "totalLiabilities": f(getattr(bal, "totalLiabilities", None)),
                        "shortTermInvestments": f(
                            getattr(bal, "shortTermInvestments", None)
                        ),
                        "shortTermReceivables": f(
                            getattr(bal, "shortTermReceivables", None)
                        ),
                        "longTermReceivables": f(getattr(bal, "longTermReceivables", None)),
                        "inventories": f(getattr(bal, "inventories", None)),
                        "fixedAssets": f(getattr(bal, "fixedAssets", None)),
                        "shortTermBorrowings": f(getattr(bal, "shortTermBorrowings", None)),
                        "longTermBorrowings": f(getattr(bal, "longTermBorrowings", None)),
                        "advancesFromCustomers": f(
                            getattr(bal, "advancesFromCustomers", None)
                        ),
                    }
                balance_sheets.append(dto)

        # Push to backend
        if balance_sheets:
            endpoint = "bank-balance-sheets" if is_bank else "non-bank-balance-sheets"
            await client.push_data(endpoint, balance_sheets)

        if income_stmts:
            endpoint = "bank-income-statements" if is_bank else "non-bank-income-statements"
            await client.push_data(endpoint, income_stmts)

        if financial_inds:
            endpoint = (
                "bank-financial-indicators" if is_bank else "non-bank-financial-indicators"
            )
            await client.push_data(endpoint, financial_inds)

        # Push shareholders + dividends (previously crawled but not persisted).
        if shareholders:
            shareholders_payload = [
                s.model_dump()
                for s in shareholders
                if getattr(s, "shareholderName", "").strip()
            ]
            if shareholders_payload:
                await client.push_data(f"shareholders/{symbol}", shareholders_payload)

        if dividends:
            dividends_payload = [d.model_dump() for d in dividends]
            if dividends_payload:
                await client.push_data(f"dividends/{symbol}", dividends_payload)

    try:
        asyncio.run(push_all())
    except Exception as push_err:
        errors.append(f"Push to DB failed: {str(push_err)}")

    if not errors:
        log(f"✅ {symbol} CRAWL AND SYNC SUCCESS")
        return symbol, True, None
    else:
        log(f"❌ {symbol} FAILED: {errors}")
        return symbol, False, errors


from vnstock import Listing


def get_market_symbols() -> list[tuple[str, bool]]:
    """
    Fetch ALL symbols dynamically from HOSE, HNX, and UPCOM.
    """
    # 1. Danh sách tất cả các ngân hàng niêm yết tại VN (đã public, hiếm khi thêm mới)
    # Đây là cách hiệu quả nhất thay vì gọi API overview() 1600 lần chỉ để check "is_bank"
    bank_symbols = {
        "VCB",
        "BID",
        "CTG",
        "MBB",
        "TCB",
        "VPB",
        "ACB",
        "SSB",
        "SHB",
        "HDB",
        "TPB",
        "VIB",
        "MSB",
        "LPB",
        "EIB",
        "OCB",
        "SEB",
        "BAB",
        "KLB",
        "NVB",
        "VAB",
        "BVB",
        "SGB",
        "NAB",
        "PGB",
        "ABB",
        "TIN",
    }

    log("Đang tải danh sách 1600+ mã chứng khoán từ HOSE, HNX, UPCOM...")
    try:
        listing = Listing(source="VCI")

        # Lấy danh sách mã chứng khoán theo từng sàn
        df_hose = listing.symbols_by_group("HOSE")
        df_hnx = listing.symbols_by_group("HNX")
        df_upcom = listing.symbols_by_group("UPCOM")

        # Lấy cột symbol (tuỳ phiên bản vnstock mà có thể trả về DataFrame hoặc Series)
        def extract_symbols(df):
            if df is None or len(df) == 0:
                return []

            # Nếu là Series
            if hasattr(df, "tolist"):
                if not hasattr(df, "columns"):
                    return df.tolist()

            # Nếu là DataFrame
            col = (
                "ticker"
                if "ticker" in df.columns
                else "symbol" if "symbol" in df.columns else df.columns[0]
            )
            return df[col].astype(str).tolist()

        all_tickers = []
        for df, exc in [(df_hose, "HOSE"), (df_hnx, "HNX"), (df_upcom, "UPCOM")]:
            syms = extract_symbols(df)
            for s in syms:
                if 2 <= len(s.strip()) <= 4:
                    all_tickers.append((s.strip().upper(), exc))

        # Loại bỏ trùng lặp (nếu có, thường không bị trùng qua các sàn)
        unique_tickers = {}
        for s, exc in all_tickers:
            unique_tickers[s] = exc

        final_list = list(unique_tickers.items())
        log(f"✅ Tải thành công {len(final_list)} mã chứng khoán!")

        # Map thành tuple (symbol, is_bank, exchange)
        return [(t, t in bank_symbols, exc) for t, exc in final_list]

    except Exception as e:
        log(f"❌ Lỗi khi lấy danh sách mã toàn thị trường: {e}")
        log("⚠️ Đang fallback về danh sách mặc định (VN30)...")
        # Fallback list just in case standard API fails
        return [
            ("FPT", False, "HOSE"),
            ("HPG", False, "HOSE"),
            ("VNM", False, "HOSE"),
            ("VCB", True, "HOSE"),
            ("TCB", True, "HOSE"),
            ("MBB", True, "HOSE"),
        ]


def run_batch_crawl():
    symbols_to_crawl = get_market_symbols()

    # --- BATCH SYNC ALL COMPANIES UPFRONT ---
    client = JavaBackendClient()

    # Cây ngành (ICB) — chạy một lần trước khi gán FK công ty
    try:
        tree = build_industry_node_payloads()
        if tree:
            asyncio.run(client.push_data("industry-nodes", tree))
            log(f"✅ Đã đồng bộ {len(tree)} nút industry-nodes.")
    except Exception as e:
        log(f"⚠️ Không đồng bộ được industry-nodes (tiếp tục crawl): {e}")

    companies_payload = [
        CompanyModel(id=sym, exchange=exc, companyType="BANK" if is_bank else "NON_BANK").model_dump()
        for sym, is_bank, exc in symbols_to_crawl
    ]
    log(
        f"Đẩy toàn bộ {len(companies_payload)} mã Công ty (Master Data) lên Backend trước để tránh rác FK..."
    )
    try:
        asyncio.run(client.push_data("companies", companies_payload))
        log("✅ Đẩy Master Data Company thành công!")
    except Exception as e:
        log(f"❌ Lỗi khi đẩy Master Data Company. Dừng Crawler: {e}")
        return

    state = load_state()
    successful_list = state.get("successful", [])

    # Lọc bỏ các mã đã crawl thành công
    pending_symbols = [s for s in symbols_to_crawl if s[0] not in successful_list]
    log(
        f"Tổng số mã: {len(symbols_to_crawl)}. Đã xong: {len(successful_list)}. Còn lại: {len(pending_symbols)}"
    )

    if not pending_symbols:
        log("🎉 Tất cả các mã đã được crawl thành công!")
        return

    failed_report = {}

    with Manager() as manager:
        manager_dict = manager.dict()
        manager_dict["rate_limit_wait_until"] = 0.0
        api_lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(run_crawler_for_symbol, sym_data, manager_dict, api_lock): sym_data[0]
                for sym_data in pending_symbols
            }

            for future in as_completed(future_to_symbol):
                try:
                    sym, success, errors = future.result()
                    if success:
                        state["successful"].append(sym)
                        save_state(state)
                    else:
                        failed_report[sym] = errors
                except Exception as exc:
                    log(f"Tiến trình crash cho một mã: {exc}")

    if failed_report:
        log(f"⚠️ Có {len(failed_report)} mã thất bại. Danh sách: {list(failed_report.keys())}")
        with open("failed_report.json", "w", encoding="utf-8") as f:
            json.dump(failed_report, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_batch_crawl()
