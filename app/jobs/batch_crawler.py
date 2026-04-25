import logging
import json
import time
import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock
from app.core.config import settings
from app.models.investment import CompanyModel
from app.infrastructure.crawler.icb_sync import build_industry_node_payloads
from app.infrastructure.crawler.fireant_crawler_service import FireAntCrawlerService
from app.infrastructure.crawler.fireant_financial_client import fetch_all_symbols
from app.infrastructure.backend_client import JavaBackendClient
import app.core.http_client as _http_mod

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
log_lock = Lock()

STATE_DIR = Path(settings.CRAWLER_STATE_DIR)
STATE_FILE = STATE_DIR / "crawler_state.json"
FAILED_REPORT_FILE = STATE_DIR / "failed_report.json"
MAX_WORKERS = 8
RETRY_MAX = 3
SAFE_DELAY = 0.5


def load_state():
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"successful": []}


def save_state(state):
    with log_lock:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)


def run_crawler_for_symbol(sym_data: tuple):
    """Crawl + push financial data for a single symbol."""
    symbol, is_bank, exchange_group = sym_data
    crawler = FireAntCrawlerService()
    errors = []

    debug_symbols_env = os.getenv("DEBUG_COMPANY_META_SYMBOLS", "")
    debug_symbols = (
        {s.strip().upper() for s in debug_symbols_env.split(",") if s.strip()}
        if debug_symbols_env
        else set()
    )

    def safe_request(func, *args):
        retries = 0
        while retries < RETRY_MAX:
            try:
                time.sleep(SAFE_DELAY)
                return func(*args)
            except Exception as e:
                retries += 1
                wait = 2 ** retries
                logger.warning("[%s] Error: %s. Retry %d/%d in %ds", symbol, e, retries, RETRY_MAX, wait)
                time.sleep(wait)
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

    # 4. Cash Flow (optional — many small companies have none)
    cashflows, w_cf = safe_request(crawler.get_cash_flow_statements, symbol)

    # 5. Shareholders
    shareholders, w_sh = safe_request(crawler.get_company_shareholders, symbol)
    if not shareholders and w_sh:
        errors.append(f"Shareholders failed: {w_sh}")

    # 6. Dividends
    dividends, w_div = safe_request(crawler.get_company_dividends, symbol)
    if not dividends and w_div:
        errors.append(f"Dividends failed: {w_div}")

    # 7. Company meta
    company_meta, w_meta = safe_request(crawler.get_company_overview_meta, symbol)

    if symbol.upper() in debug_symbols:
        logger.info("[%s] company_meta=%s warnings=%s", symbol, company_meta, w_meta)

    # --- MAP TO DTOs AND PUSH TO JAVA BACKEND ---
    client = JavaBackendClient()

    async def push_all():
        def f(v):
            return float(v) if v is not None else None

        icb_code = (company_meta.get("icbCode") or "").strip() if company_meta else ""
        company_name = company_meta.get("companyName") if company_meta else None
        description = company_meta.get("description") if company_meta else None

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

        # Financial Indicators — model_dump gives camelCase keys matching Java DTOs
        if inds:
            financial_inds = []
            for ind in inds:
                dto = ind.model_dump()
                if is_bank:
                    dto.pop("grossMargin", None)
                    dto.pop("netMargin", None)
                else:
                    dto.pop("grossMargin", None)
                    dto.pop("netMargin", None)
                financial_inds.append(dto)

            endpoint = "bank-financial-indicators" if is_bank else "non-bank-financial-indicators"
            await client.push_data(endpoint, financial_inds)

        # Income Statements
        if incomes:
            income_stmts = []
            for inc in incomes:
                dto = inc.model_dump()
                if is_bank:
                    dto["interestExpense"] = dto.pop("interestAndSimilarExpenses", None)
                income_stmts.append(dto)

            endpoint = "bank-income-statements" if is_bank else "non-bank-income-statements"
            await client.push_data(endpoint, income_stmts)

        # Balance Sheets
        if balances:
            balance_sheets = [bal.model_dump() for bal in balances]
            endpoint = "bank-balance-sheets" if is_bank else "non-bank-balance-sheets"
            await client.push_data(endpoint, balance_sheets)

        # Cash Flow Statements
        if cashflows:
            cf_payload = [cf.model_dump() for cf in cashflows]
            await client.push_data("cash-flow-statements", cf_payload)

        # Shareholders
        if shareholders:
            shareholders_payload = [
                s.model_dump()
                for s in shareholders
                if getattr(s, "shareholderName", "").strip()
            ]
            if shareholders_payload:
                await client.push_data(f"shareholders/{symbol}", shareholders_payload)

        # Dividends
        if dividends:
            dividends_payload = [d.model_dump() for d in dividends]
            if dividends_payload:
                await client.push_data(f"dividends/{symbol}", dividends_payload)

    try:
        _http_mod._client = None
        asyncio.run(push_all())
    except Exception as push_err:
        errors.append(f"Push to DB failed: {str(push_err)}")

    if not errors:
        logger.info("✅ %s CRAWL AND SYNC SUCCESS", symbol)
        return symbol, True, None
    else:
        logger.info("❌ %s FAILED: %s", symbol, errors)
        return symbol, False, errors


def get_market_symbols() -> list[tuple[str, bool, str]]:
    """Fetch ALL symbols from FireAnt, auto-detect bank vs non-bank."""
    logger.info("Fetching symbol list from FireAnt...")

    debug_env = os.getenv("DEBUG_SYMBOLS", "")
    if debug_env:
        forced = [s.strip().upper() for s in debug_env.split(",") if s.strip()]
        logger.info("DEBUG_SYMBOLS override: %s", forced)
        crawler = FireAntCrawlerService()
        result = []
        for sym in forced:
            ct = crawler.detect_company_type(sym)
            is_bank = ct == "Bank" if ct else False
            result.append((sym, is_bank, "HOSE"))
        return result

    raw = fetch_all_symbols()
    if not raw:
        logger.error("FireAnt GET /symbols returned empty — using fallback")
        return [
            ("FPT", False, "HOSE"),
            ("HPG", False, "HOSE"),
            ("VNM", False, "HOSE"),
            ("VCB", True, "HOSE"),
            ("TCB", True, "HOSE"),
            ("MBB", True, "HOSE"),
        ]

    results = []
    for item in raw:
        sym = (item.get("symbol") or item.get("ticker") or "").strip().upper()
        if not sym or len(sym) < 2 or len(sym) > 4:
            continue

        exchange = (item.get("exchange") or item.get("floor") or "HOSE").strip().upper()
        icb = (item.get("icbCode") or "").strip()
        is_bank = icb.startswith("3010")
        results.append((sym, is_bank, exchange))

    logger.info("✅ Loaded %d symbols from FireAnt", len(results))
    return results


def run_batch_crawl():
    symbols_to_crawl = get_market_symbols()

    client = JavaBackendClient()

    async def push_master_data():
        # ICB tree first
        try:
            tree = build_industry_node_payloads()
            if tree:
                await client.push_data("industry-nodes", tree)
                logger.info("✅ Synced %d industry nodes.", len(tree))
        except Exception as e:
            logger.info("⚠️ Failed to sync industry-nodes (continuing): %s", e)

        # Push all companies master data upfront
        companies_payload = [
            CompanyModel(id=sym, exchange=exc, companyType="BANK" if is_bank else "NON_BANK").model_dump()
            for sym, is_bank, exc in symbols_to_crawl
        ]
        logger.info("Pushing %d companies (Master Data) to Backend...", len(companies_payload))
        await client.push_data("companies", companies_payload)
        logger.info("✅ Master Data push success!")

    try:
        _http_mod._client = None
        asyncio.run(push_master_data())
    except Exception as e:
        logger.info("❌ Failed to push Master Data. Stopping: %s", e)
        return

    state = load_state()
    successful_list = state.get("successful", [])

    pending_symbols = [s for s in symbols_to_crawl if s[0] not in successful_list]
    logger.info(
        "Total: %d. Done: %d. Pending: %d",
        len(symbols_to_crawl), len(successful_list), len(pending_symbols),
    )

    if not pending_symbols:
        logger.info("🎉 All symbols already crawled!")
        return

    failed_report = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(run_crawler_for_symbol, sym_data): sym_data[0]
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
                logger.info("Process crashed: %s", exc)

    if failed_report:
        logger.info("⚠️ %d symbols failed: %s", len(failed_report), list(failed_report.keys()))
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with FAILED_REPORT_FILE.open("w", encoding="utf-8") as f:
            json.dump(failed_report, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_batch_crawl()
