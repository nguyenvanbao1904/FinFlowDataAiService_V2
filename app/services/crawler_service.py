import logging
import os
import json
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from vnstock import Company, Finance
from app.services.fireant_profile import _empty_meta, fetch_fireant_company_meta
from app.services.icb_normalization import normalize_icb_code
from app.domain.entities.investment import (
    BankFinancialIndicator, NonBankFinancialIndicator,
    NonBankIncomeStatement, NonBankBalanceSheet,
    BankIncomeStatement, BankBalanceSheet,
    CompanyModel, CompanyShareholderModel, CompanyDividendModel
)

logger = logging.getLogger(__name__)


def _merge_company_meta(
    prefer: Dict[str, Optional[str]],
    fill: Dict[str, Optional[str]],
) -> Dict[str, Optional[str]]:
    """Prefer non-empty fields from ``prefer`` (e.g. FireAnt), then ``fill`` (vnstock VCI)."""

    def norm(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def pick(a: Optional[str], b: Optional[str]) -> Optional[str]:
        return norm(a) or norm(b)

    return {
        "companyName": pick(prefer.get("companyName"), fill.get("companyName")),
        "industryLabelFull": pick(
            prefer.get("industryLabelFull"), fill.get("industryLabelFull")
        ),
        "description": pick(prefer.get("description"), fill.get("description")),
        "icbCode": pick(prefer.get("icbCode"), fill.get("icbCode")),
    }


class AliasMapper:
    """Helper to extract values from pandas Series using multiple possible column names."""
    @staticmethod
    def get_value(row: pd.Series, aliases: list[str]) -> float | None:
        for alias in aliases:
            if alias in row.index and pd.notna(row[alias]):
                return float(row[alias])
        return None

class VnstockCrawlerService:
    """Service for crawling financial data using vnstock library."""

    # ``Company.overview()`` VCI không có icb_code1..4; mã số chỉ có trong ``Listing.symbols_by_industries()``.
    _listing_icb_by_symbol: Optional[Dict[str, str]] = None

    @classmethod
    def _load_vci_listing_icb_cache(cls) -> None:
        if cls._listing_icb_by_symbol is not None:
            return
        from vnstock import Listing

        logger.info("Đang tải VCI symbols_by_industries (mã ICB theo ticker) — cache một lần / process…")
        df = Listing(source="VCI").symbols_by_industries(show_log=False)
        m: Dict[str, str] = {}
        for _, row in df.iterrows():
            sym = str(row.get("symbol", "")).strip().upper()
            if not sym:
                continue
            code: Optional[str] = None
            for col in ("icb_code4", "icb_code3", "icb_code2", "icb_code1"):
                v = row.get(col)
                if v is None or pd.isna(v):
                    continue
                code = normalize_icb_code(v)
                if code:
                    break
            if code:
                m[sym] = code
        cls._listing_icb_by_symbol = m
        logger.info("Đã cache ICB cho %d mã từ VCI listing.", len(m))

    @classmethod
    def _get_icb_from_vci_listing(cls, symbol: str) -> Optional[str]:
        cls._load_vci_listing_icb_cache()
        assert cls._listing_icb_by_symbol is not None
        return cls._listing_icb_by_symbol.get(symbol.strip().upper())

    # Define aliases for fluctuating column names
    INCOME_PROFIT = ["Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)", "Lợi nhuận sau thuế (đồng)"]
    INCOME_NET_REVENUE = ["Doanh thu thuần về bán hàng và cung cấp dịch vụ (đồng)", "Doanh thu (đồng)", "Doanh thu thuần (đồng)", "Doanh thu thuần"]
    
    # Bank Income Aliases
    BANK_NET_INTEREST = ["Thu nhập lãi thuần"]
    BANK_NET_FEE = ["Lãi/lỗ thuần từ hoạt động dịch vụ", "Lãi thuần từ hoạt động dịch vụ"]
    BANK_NET_OTHER = ["Lãi/lỗ thuần từ hoạt động khác", "Lãi thuần từ hoạt động khác"]
    BANK_INTEREST_EXPENSE = ["Chi phí lãi và các khoản tương tự", "Chi phí trả lãi tiền gửi"]

    # Balance Sheet Common
    BAL_CASH = ["Tiền và tương đương tiền (đồng)"]
    BAL_TOTAL_ASSETS = ["TỔNG CỘNG TÀI SẢN (đồng)"]
    BAL_EQUITY = ["VỐN CHỦ SỞ HỮU (đồng)"]
    BAL_TOTAL_CAPITAL = ["TỔNG CỘNG NGUỒN VỐN (đồng)"]
    BAL_TOTAL_LIABILITIES = ["TỔNG CỘNG NỢ PHẢI TRẢ (đồng)", "NỢ PHẢI TRẢ (đồng)"]

    # Non-Bank Balance Sheet
    NONBANK_ST_INVEST = ["Giá trị thuần đầu tư ngắn hạn (đồng)", "Đầu tư tài chính ngắn hạn"]
    NONBANK_ST_RECV = ["Các khoản phải thu ngắn hạn (đồng)"]
    NONBANK_LT_RECV = ["Phải thu dài hạn (đồng)", "Các khoản phải thu dài hạn (đồng)"]
    NONBANK_INV = ["Hàng tồn kho, ròng (đồng)", "Hàng tồn kho ròng", "Hàng tồn kho (đồng)"]
    NONBANK_FIXED_ASSETS = ["Tài sản cố định (đồng)"]
    NONBANK_ST_BORROW = ["Vay và nợ thuê tài chính ngắn hạn (đồng)", "Vay và nợ thuê tài chính ngắn hạn"]
    NONBANK_LT_BORROW = ["Vay và nợ thuê tài chính dài hạn (đồng)", "Vay và nợ thuê tài chính dài hạn"]
    NONBANK_ADVANCES = ["Người mua trả tiền trước ngắn hạn (đồng)", "Người mua trả tiền trước ngắn hạn"]

    # Bank Balance Sheet
    BANK_BAL_SBV = ["Tiền gửi tại ngân hàng nhà nước Việt Nam", "Tiền gửi tại NHNN Việt Nam"]
    BANK_INTERBANK = ["Tiền gửi tại các TCTD khác và cho vay các TCTD khác"]
    BANK_TRADING_SEC = ["Chứng khoán kinh doanh"]
    BANK_INVEST_SEC = ["Chứng khoán đầu tư"]
    BANK_LOANS = ["Cho vay khách hàng"]
    BANK_DEBT_GOV = ["Các khoản nợ chính phủ và NHNN Việt Nam"]
    BANK_DEPOSITS_OTHER = ["Tiền gửi và vay các Tổ chức tín dụng khác"]
    BANK_DEPOSITS_CUST = ["Tiền gửi của khách hàng"]
    BANK_PAPERS = ["Phát hành giấy tờ có giá"]

    # Indicators Aliases (MultiIndex Tuples)
    IND_PE = [('Chỉ tiêu định giá', 'P/E'), ('Định giá', 'P/E')]
    IND_PB = [('Chỉ tiêu định giá', 'P/B'), ('Định giá', 'P/B')]
    IND_PS = [('Chỉ tiêu định giá', 'P/S'), ('Định giá', 'P/S')]
    IND_ROE = [('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'), ('Hiệu quả hoạt động', 'ROE (%)')]
    IND_ROA = [('Chỉ tiêu khả năng sinh lợi', 'ROA (%)'), ('Hiệu quả hoạt động', 'ROA (%)')]
    IND_EPS = [('Chỉ tiêu định giá', 'EPS (VND)'), ('Định giá', 'EPS (VND)')]
    IND_BVPS = [('Chỉ tiêu định giá', 'BVPS (VND)'), ('Định giá', 'BVPS (VND)')]
    IND_LNG = [('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận gộp (%)'), ('Hiệu quả hoạt động', 'Biên lợi nhuận gộp (%)')]
    IND_LNR = [('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'), ('Hiệu quả hoạt động', 'Biên lợi nhuận ròng (%)')]
    IND_CPLH = [('Chỉ tiêu định giá', 'Số CP lưu hành (Triệu CP)'), ('Định giá', 'Số CP lưu hành (Triệu CP)')]
    IND_YEAR = [('Meta', 'Năm'), 'Năm']
    IND_QUARTER = [('Meta', 'Kỳ'), 'Kỳ']

    def __init__(self):
        self.mapper = AliasMapper()

    def get_financial_indicators(self, symbol: str, is_bank: bool = False) -> Tuple[List[Any], List[str]]:
        """Returns (List of full historical quarters, List of missing fields if any)"""
        logger.info(f"Crawling indicators for {symbol}...")
        results = []
        warnings = []
        try:
            finance = Finance(symbol=symbol, source="VCI")
            ratio_df = finance.ratio(lang='vi')
            
            if ratio_df.empty:
                return [], [f"Empty ratio dataframe for {symbol}"]
                
            for _, row in ratio_df.iterrows():
                try:
                    base_kwargs = dict(
                        companyId=symbol,
                        year=int(self.mapper.get_value(row, self.IND_YEAR) or 0),
                        quarter=int(self.mapper.get_value(row, self.IND_QUARTER) or 0),
                        pe=self.mapper.get_value(row, self.IND_PE),
                        pb=self.mapper.get_value(row, self.IND_PB),
                        ps=self.mapper.get_value(row, self.IND_PS), 
                        roe=self.mapper.get_value(row, self.IND_ROE),
                        roa=self.mapper.get_value(row, self.IND_ROA),
                        eps=self.mapper.get_value(row, self.IND_EPS),
                        bvps=self.mapper.get_value(row, self.IND_BVPS),
                        cplh=self.mapper.get_value(row, self.IND_CPLH)
                    )
                    
                    if is_bank:
                        indicator = BankFinancialIndicator(
                            **base_kwargs
                        )
                    else:
                        indicator = NonBankFinancialIndicator(
                            **base_kwargs,
                            lng=self.mapper.get_value(row, self.IND_LNG),
                            lnr=self.mapper.get_value(row, self.IND_LNR)
                        )
                    results.append(indicator)
                except Exception as e:
                    warnings.append(f"Row mapping failed: {e}")
                    
            return results, warnings
        except Exception as e:
            return [], [f"Exception: {str(e)}"]

    def get_income_statement(self, symbol: str, is_bank: bool = False) -> Tuple[List[Any], List[str]]:
        logger.info(f"Crawling Income Statement for {symbol}...")
        results = []
        warnings = []
        try:
            finance = Finance(symbol=symbol, source="VCI")
            df = finance.income_statement(lang='vi')
            if df.empty:
                return [], [f"Empty income statement dataframe for {symbol}"]
                
            for _, row in df.iterrows():
                try:
                    profit = self.mapper.get_value(row, self.INCOME_PROFIT)
                    year = int(self.mapper.get_value(row, ['Năm']) or 0)
                    quarter = int(self.mapper.get_value(row, ['Kỳ']) or 0)
                    
                    if is_bank:
                        statement = BankIncomeStatement(
                            companyId=symbol, year=year, quarter=quarter,
                            profitAfterTax=profit,
                            netInterestIncome=self.mapper.get_value(row, self.BANK_NET_INTEREST),
                            netFeeAndCommissionIncome=self.mapper.get_value(row, self.BANK_NET_FEE),
                            netOtherIncomeOrExpenses=self.mapper.get_value(row, self.BANK_NET_OTHER),
                            interestAndSimilarExpenses=self.mapper.get_value(row, self.BANK_INTEREST_EXPENSE),
                            totalRevenue=self.mapper.get_value(row, self.INCOME_NET_REVENUE),
                            netProfit=profit
                        )
                    else:
                        statement = NonBankIncomeStatement(
                            companyId=symbol, year=year, quarter=quarter,
                            profitAfterTax=profit,
                            netRevenue=self.mapper.get_value(row, self.INCOME_NET_REVENUE),
                            totalRevenue=self.mapper.get_value(row, self.INCOME_NET_REVENUE),
                            netProfit=profit
                        )
                    results.append(statement)
                except Exception as e:
                    warnings.append(f"Row mapping failed: {e}")
            return results, warnings
        except Exception as e:
            return [], [f"Exception: {str(e)}"]

    def get_balance_sheet(self, symbol: str, is_bank: bool = False) -> Tuple[List[Any], List[str]]:
        logger.info(f"Crawling Balance Sheet for {symbol}...")
        results = []
        warnings = []
        try:
            finance = Finance(symbol=symbol, source="VCI")
            df = finance.balance_sheet(lang='vi')
            if df.empty:
                return [], [f"Empty balance sheet dataframe for {symbol}"]
                
            for _, row in df.iterrows():
                try:
                    year = int(self.mapper.get_value(row, ['Năm']) or 0)
                    quarter = int(self.mapper.get_value(row, ['Kỳ']) or 0)
                    cash = self.mapper.get_value(row, self.BAL_CASH)
                    total_assets = self.mapper.get_value(row, self.BAL_TOTAL_ASSETS)
                    equity = self.mapper.get_value(row, self.BAL_EQUITY)
                    total_cap = self.mapper.get_value(row, self.BAL_TOTAL_CAPITAL)
                    total_liab = self.mapper.get_value(row, self.BAL_TOTAL_LIABILITIES)
                    
                    if is_bank:
                        sheet = BankBalanceSheet(
                            companyId=symbol, year=year, quarter=quarter,
                            cashAndCashEquivalents=cash, totalAssets=total_assets,
                            equity=equity, totalCapital=total_cap, totalLiabilities=total_liab,
                            balancesWithSbv=self.mapper.get_value(row, self.BANK_BAL_SBV),
                            interbankPlacementsAndLoans=self.mapper.get_value(row, self.BANK_INTERBANK),
                            tradingSecurities=self.mapper.get_value(row, self.BANK_TRADING_SEC),
                            investmentSecurities=self.mapper.get_value(row, self.BANK_INVEST_SEC),
                            loansToCustomers=self.mapper.get_value(row, self.BANK_LOANS),
                            govAndSbvDebt=self.mapper.get_value(row, self.BANK_DEBT_GOV),
                            depositsBorrowingsOthers=self.mapper.get_value(row, self.BANK_DEPOSITS_OTHER),
                            depositsFromCustomers=self.mapper.get_value(row, self.BANK_DEPOSITS_CUST),
                            convertibleAndOtherPapers=self.mapper.get_value(row, self.BANK_PAPERS)
                        )
                    else:
                        sheet = NonBankBalanceSheet(
                            companyId=symbol, year=year, quarter=quarter,
                            cashAndCashEquivalents=cash, totalAssets=total_assets,
                            equity=equity, totalCapital=total_cap, totalLiabilities=total_liab,
                            shortTermInvestments=self.mapper.get_value(row, self.NONBANK_ST_INVEST),
                            shortTermReceivables=self.mapper.get_value(row, self.NONBANK_ST_RECV),
                            longTermReceivables=self.mapper.get_value(row, self.NONBANK_LT_RECV),
                            inventories=self.mapper.get_value(row, self.NONBANK_INV),
                            fixedAssets=self.mapper.get_value(row, self.NONBANK_FIXED_ASSETS),
                            shortTermBorrowings=self.mapper.get_value(row, self.NONBANK_ST_BORROW),
                            longTermBorrowings=self.mapper.get_value(row, self.NONBANK_LT_BORROW),
                            advancesFromCustomers=self.mapper.get_value(row, self.NONBANK_ADVANCES)
                        )
                    results.append(sheet)
                except Exception as e:
                    warnings.append(f"Row mapping failed: {e}")
            return results, warnings
        except Exception as e:
            return [], [f"Exception: {str(e)}"]

    def _clean_date_str(self, val: Any) -> Optional[str]:
        """Cleans pandas date columns to prevent empty string vs NaT bugs bridging to Java LocalDate"""
        if pd.isna(val) or val is None or str(val).strip() == "":
            return None
        # vnstock usually returns dates as "YYYY-MM-DD" strings in events
        return str(val).strip()

    def _company_overview_meta_vci(self, symbol: str) -> Tuple[dict[str, Optional[str]], List[str]]:
        """
        vnstock VCI `Company.overview()` → companyName / industry / synthetic description.
        """
        logger.debug(f"[{symbol}] Company overview meta via vnstock VCI...")
        try:
            company = Company(symbol=symbol, source="VCI")
            df = company.overview()
            if df is None or len(df) == 0:
                return _empty_meta(), []

            row = df.iloc[0].to_dict()

            def is_missing(v: Any) -> bool:
                return v is None or (isinstance(v, str) and v.strip() == "") or pd.isna(v)

            # Build a lowercase key index for resilient access.
            row_lc: dict[str, Any] = {str(k).lower(): v for k, v in row.items()}

            debug_symbol = os.getenv("DEBUG_OVERVIEW_SYMBOL", "").strip().upper()
            if debug_symbol and debug_symbol == symbol.upper():
                # Log a small subset to understand vnstock overview column naming.
                keys_industry = sorted([k for k in row_lc.keys() if "industry" in k and "industry_id" not in k])
                keys_name = sorted([k for k in row_lc.keys() if "name" in k and "share" not in k and "owner" not in k])
                keys_short = sorted([k for k in row_lc.keys() if "short" in k])
                keys_company_like = sorted([k for k in row_lc.keys() if any(x in k for x in ["company", "organ", "organization", "ticker"])])
                logger.warning(
                    f"[{symbol}] overview keys_industry={keys_industry} keys_short={keys_short} keys_company_like={keys_company_like} keys_name_sample={keys_name[:20]}"
                )
                for k in ["industry", "short_name", "company_name", "company_full_name", "organ_name", "organization_name", "ticker_name", "industry_label", "name"]:
                    if k in row_lc:
                        logger.warning(f"[{symbol}] overview {k}={row_lc.get(k)!r}")
                cp = row_lc.get("company_profile")
                logger.warning(f"[{symbol}] overview company_profile_type={type(cp).__name__} value_preview={str(cp)[:200]!r}")
                if isinstance(cp, str):
                    try:
                        cp_obj_dbg = json.loads(cp)
                        if isinstance(cp_obj_dbg, dict):
                            logger.warning(f"[{symbol}] overview company_profile_dbg_keys={list(cp_obj_dbg.keys())[:40]}")
                    except Exception:
                        pass
                elif isinstance(cp, dict):
                    logger.warning(f"[{symbol}] overview company_profile_dbg_keys={list(cp.keys())[:40]}")

            # 1) industry
            # Prefer keys that contain "industry" but exclude "industry_id*" helpers.
            industry: Optional[str] = None
            industry_keys = [k for k in row_lc.keys() if "industry" in k and "industry_id" not in k]
            # Keep stable preference order (most likely keys first).
            industry_preferred = [
                "industry",
                "industry_name",
                "industry_label",
                "industrytype",
                "sector",
            ]
            # Add whatever keys exist as fallback (but still filtered above).
            industry_keys_sorted = industry_preferred + [k for k in industry_keys if k not in industry_preferred]
            for k in industry_keys_sorted:
                val = row_lc.get(k)
                if not is_missing(val):
                    industry = str(val).strip()
                    break

            # Fallback: some vnstock versions don't expose an "industry" column in overview(),
            # but do expose ICB buckets (icb_name2/3/4). For UI we treat icb_name2 as industry.
            if is_missing(industry) or industry is None:
                for k in ["icb_name2", "icb_name3", "icb_name4"]:
                    val = row_lc.get(k)
                    if not is_missing(val):
                        industry = str(val).strip()
                        break

            # ICB code + full label path (backend resolve → industry_nodes / companies.industry_node_id)
            icb_code_v: Optional[str] = None
            for k in [
                "icb_code",
                "icbid",
                "ma_nganh_icb",
                "industry_code",
                "industrycode",
            ]:
                val = row_lc.get(k)
                if not is_missing(val):
                    icb_code_v = normalize_icb_code(val)
                    break

            # VCI CompaniesListingInfo / overview: mã số thường nằm ở icbCode1..4 (key sau lower()).
            if not icb_code_v:
                for level in (4, 3, 2, 1):
                    for k in (
                        f"icbcode{level}",
                        f"icb_code{level}",
                        f"icb_code_{level}",
                    ):
                        val = row_lc.get(k)
                        if not is_missing(val):
                            icb_code_v = normalize_icb_code(val)
                            if icb_code_v:
                                break
                    if icb_code_v:
                        break

            # Khi không dùng FireAnt: overview() thường không có cột mã — lấy từ CompaniesListingInfo (vnstock Listing).
            # Khi có FIREANT_ACCESS_TOKEN: không gán mã từ VCI listing (tránh lệch với cây ``GET /icb`` FireAnt).
            if not icb_code_v:
                if not (
                    (os.getenv("FIREANT_ACCESS_TOKEN") or os.getenv("FIREANT_BEARER_TOKEN") or "").strip()
                ):
                    icb_code_v = VnstockCrawlerService._get_icb_from_vci_listing(symbol)

            icb_parts: list[str] = []
            # ICB thường: cấp rộng → hẹp (vd. Ngân hàng > …)
            for k in ["icb_name2", "icb_name3", "icb_name4"]:
                val = row_lc.get(k)
                if not is_missing(val):
                    t = str(val).strip()
                    if t and t not in icb_parts:
                        icb_parts.append(t)
            icb_hierarchy = " > ".join(icb_parts) if icb_parts else None
            industry_label_full_v: Optional[str] = icb_hierarchy
            if not industry_label_full_v and industry:
                industry_label_full_v = industry
            if industry_label_full_v and len(industry_label_full_v) > 2000:
                industry_label_full_v = industry_label_full_v[:1997] + "..."

            # 2) company name
            # Prefer explicit company name keys first; do not use generic "name" unless we must.
            company_name: Optional[str] = None
            company_name_preferred = [
                "short_name",
                "company_name",
                "company_full_name",
                "organ_name",
                "organization_name",
                "ticker_name",
                "company_listing_name",
            ]
            for k in company_name_preferred:
                val = row_lc.get(k)
                if not is_missing(val):
                    company_name = str(val).strip()
                    break

            if company_name is None:
                # Fallback: generic "name" but only if it's not about industry/share/owner.
                name_keys = [
                    k for k in row_lc.keys()
                    if ("name" in k)
                    and ("industry" not in k)
                    and ("share" not in k)
                    and ("owner" not in k)
                    and not k.startswith("icb_")
                ]
                for k in name_keys:
                    val = row_lc.get(k)
                    if not is_missing(val):
                        company_name = str(val).strip()
                        break

            # Last-resort fallback: company_profile payload (often contains company name)
            if company_name is None:
                cp = row_lc.get("company_profile")
                cp_obj: Any = None
                if isinstance(cp, dict):
                    cp_obj = cp
                elif isinstance(cp, str):
                    try:
                        cp_obj = json.loads(cp)
                    except Exception:
                        cp_obj = None

                if isinstance(cp_obj, dict):
                    # Try common keys first (case-insensitive handled by listing both variants)
                    candidate_keys = [
                        "companyName",
                        "company_name",
                        "short_name",
                        "shortName",
                        "companyFullName",
                        "company_full_name",
                        "fullName",
                        "organ_name",
                        "organization_name",
                        "organizationName",
                        "ticker_name",
                        "company_listing_name",
                        "companyListingName",
                    ]

                    for cand in candidate_keys:
                        v = cp_obj.get(cand)
                        if not is_missing(v):
                            s = str(v).strip()
                            # Avoid accidentally picking the generic industry label.
                            if industry and s.lower() == str(industry).strip().lower():
                                continue
                            company_name = s
                            break

                    # If still missing, recursively scan nested dicts/lists for likely keys.
                    if company_name is None:
                        def find_in_obj(obj: Any, keys: list[str]) -> Optional[str]:
                            keys_lower = {k.lower() for k in keys}
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if str(k).lower() in keys_lower and not is_missing(v):
                                        s = str(v).strip()
                                        if industry and s.lower() == str(industry).strip().lower():
                                            continue
                                        return s
                                for v in obj.values():
                                    res = find_in_obj(v, keys)
                                    if res:
                                        return res
                            elif isinstance(obj, list):
                                for v in obj:
                                    res = find_in_obj(v, keys)
                                    if res:
                                        return res
                            return None

                        company_name = find_in_obj(cp_obj, candidate_keys)

            # 3) description (company intro):
            # vnstock `reports().description` thường là mô tả báo cáo phân tích/khuyến nghị,
            # không phải "giới thiệu công ty". Ở đây ta tạo "company bio" đơn giản từ overview.
            description: Optional[str] = None
            try:
                def pick_first(*cands: str) -> Optional[str]:
                    for cand in cands:
                        v = row_lc.get(cand)
                        if not is_missing(v):
                            return str(v).strip()
                    return None

                established_year = pick_first("established_year", "establishedyear", "established")
                website = pick_first("website", "web")
                short_name = pick_first("short_name", "company_name", "company_full_name")

                name_for_intro = company_name or short_name
                parts: list[str] = []
                if name_for_intro:
                    parts.append(name_for_intro)
                if industry:
                    parts.append(f"hoạt động trong lĩnh vực {industry}")
                if established_year:
                    parts.append(f"thành lập năm {established_year}")
                if website:
                    parts.append(f"Website: {website}")

                s = " ".join(parts).strip()
                if s:
                    description = s
            except Exception:
                description = None

            return {
                "companyName": company_name,
                "industryLabelFull": industry_label_full_v,
                "description": description,
                "icbCode": icb_code_v,
            }, []
        except Exception as e:
            m = _empty_meta()
            return m, [f"Exception: {str(e)}"]

    def get_company_overview_meta(self, symbol: str) -> Tuple[dict[str, Optional[str]], List[str]]:
        """
        Company metadata for the backend ``companies`` table.

        - Optional **FireAnt** REST ``/symbols/{symbol}/profile`` when ``FIREANT_ACCESS_TOKEN`` is set
          (same data as the FireAnt web “Hồ sơ” tab; needs OAuth Bearer, scope ``symbols-read``).
        - Luôn merge **vnstock VCI** overview để bù tên/mô tả khi thiếu.
        - Mã ICB: ưu tiên FireAnt khi có token; không dùng VCI listing cho mã khi có token (xem ``_company_overview_meta_vci``).
        """
        logger.info(f"Crawling Company overview meta for {symbol}...")
        fa, w_fa = fetch_fireant_company_meta(symbol)
        vci, w_vci = self._company_overview_meta_vci(symbol)
        merged = _merge_company_meta(fa, vci)
        return merged, w_fa + w_vci

    def get_company_shareholders(self, symbol: str) -> Tuple[List[CompanyShareholderModel], List[str]]:
        logger.info(f"Crawling Shareholders for {symbol}...")
        results = []
        warnings = []
        try:
            company = Company(symbol=symbol, source="VCI")
            df = company.shareholders()
            if df is None or len(df) == 0:
                return [], [] # Standard empty state
                
            for _, row in df.iterrows():
                try:
                    q_val = self.mapper.get_value(row, ['quantity'])
                    p_val = self.mapper.get_value(row, ['share_own_percent'])
                    
                    sh = CompanyShareholderModel(
                        companyId=symbol,
                        shareholderName=str(row.get('share_holder', '')).strip(),
                        quantity=int(q_val) if pd.notna(q_val) else None,
                        shareOwnPercent=float(p_val) if pd.notna(p_val) else None,
                        updateDate=self._clean_date_str(row.get('update_date'))
                    )
                    results.append(sh)
                except Exception as e:
                    warnings.append(f"Row mapping failed: {e}")
            return results, warnings
        except Exception as e:
            return [], [f"Exception: {str(e)}"]

    def get_company_dividends(self, symbol: str) -> Tuple[List[CompanyDividendModel], List[str]]:
        logger.info(f"Crawling Dividends for {symbol}...")
        results = []
        warnings = []
        try:
            company = Company(symbol=symbol, source="VCI")
            df = company.events()
            if df is None or len(df) == 0:
                return [], []
            
            # Filter only dividend events ("cổ tức" or "cổ phiếu thưởng")
            filter_mask = df['event_title'].str.contains('cổ tức|thưởng', case=False, na=False)
            df = df[filter_mask]
            
            for _, row in df.iterrows():
                try:
                    title = str(row.get('event_title', '')).strip()
                    event_list_name = str(row.get('event_list_name', '')).strip()
                    
                    # Deduce TYPE (CASH vs STOCK)
                    event_type = "STOCK"
                    if "tiền" in title.lower() or "tiền" in event_list_name.lower():
                        event_type = "CASH"
                        
                    # Value extraction (VND/share for cash)
                    val = self.mapper.get_value(row, ['value'])
                    # Ratio extraction 
                    ratio = str(row.get('ratio', '')) if pd.notna(row.get('ratio')) else None
                    if ratio == "": ratio = None
                    
                    div = CompanyDividendModel(
                        companyId=symbol,
                        eventTitle=title,
                        eventType=event_type,
                        ratio=ratio,
                        value=float(val) if pd.notna(val) else None,
                        recordDate=self._clean_date_str(row.get('record_date')),
                        exrightDate=self._clean_date_str(row.get('exright_date')),
                        issueDate=self._clean_date_str(row.get('issue_date'))
                    )
                    results.append(div)
                except Exception as e:
                    warnings.append(f"Row mapping failed: {e}")
            return results, warnings
        except Exception as e:
            return [], [f"Exception: {str(e)}"]
