"""
FireAntCrawlerService — drop-in replacement for VnstockCrawlerService.

Uses a single ``GET /symbols/{symbol}/financial-data?type=Q&count=100`` call
to fetch ALL quarterly data (up to 85 quarters), then splits the response
into indicators, income statements, balance sheets, and cash flow statements.
"""

from __future__ import annotations

import logging
from typing import Any

from app.infrastructure.crawler.fireant_adapter import fetch_fireant_company_meta
from app.infrastructure.crawler.fireant_financial_client import (
    fetch_financial_data,
    fetch_holders,
    fetch_dividends,
)
from app.models.investment import (
    BankBalanceSheet,
    BankFinancialIndicator,
    BankIncomeStatement,
    CashFlowStatement,
    CompanyDividendModel,
    CompanyShareholderModel,
    NonBankBalanceSheet,
    NonBankFinancialIndicator,
    NonBankIncomeStatement,
)

logger = logging.getLogger(__name__)


def _fv(quarter_data: dict[str, Any], key: str) -> float | None:
    """Extract a value from the nested ``financialValues`` dict."""
    fv = quarter_data.get("financialValues") or {}
    v = fv.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class FireAntCrawlerService:
    """
    Replaces VnstockCrawlerService.
    Each ``get_*`` method returns ``(list[Model], list[str_warnings])``.
    """

    def _fetch_quarters(self, symbol: str) -> list[dict[str, Any]]:
        data = fetch_financial_data(symbol, report_type="Q", count=100)
        if not data:
            return []
        # FireAnt occasionally emits stray legacy rows (e.g. ACB year=1990) that
        # predate the Vietnamese stock market (HOSE opened mid-2000). Drop any
        # row whose top-level year is implausibly old to keep charts clean.
        MIN_VALID_YEAR = 2000
        cleaned: list[dict[str, Any]] = []
        for d in data:
            if not isinstance(d, dict) or not d.get("financialValues"):
                continue
            y = d.get("year")
            try:
                y_int = int(y) if y is not None else 0
            except (TypeError, ValueError):
                y_int = 0
            if y_int and y_int < MIN_VALID_YEAR:
                continue
            cleaned.append(d)
        return cleaned

    # ------------------------------------------------------------------
    # Financial Indicators
    # ------------------------------------------------------------------

    def get_financial_indicators(
        self, symbol: str, is_bank: bool = False
    ) -> tuple[list[Any], list[str]]:
        quarters = self._fetch_quarters(symbol)
        if not quarters:
            return [], [f"No financial data from FireAnt for {symbol}"]

        results: list[Any] = []
        for q in quarters:
            year = q.get("year") or int((_fv(q, "Year") or 0))
            quarter = q.get("quarter") or int((_fv(q, "Quarter") or 0))
            if not year or not quarter:
                continue

            base = dict(
                companyId=symbol,
                year=year,
                quarter=quarter,
                pe=_fv(q, "PE"),
                pb=_fv(q, "PB"),
                ps=_fv(q, "PS"),
                roe=_fv(q, "ROE"),
                roa=_fv(q, "ROA"),
                eps=_fv(q, "BasicEPS"),
                bvps=_fv(q, "BookValuePerShare"),
                cplh=_fv(q, "ShareAtPeriodEnd"),
                grossMargin=_fv(q, "GrossMargin"),
                netMargin=_fv(q, "ROS"),
                saleGrowth=_fv(q, "SaleGrowth"),
                profitGrowth=_fv(q, "ProfitGrowth"),
                currentRatio=_fv(q, "CurrentRatio"),
                totalDebtOverEquity=_fv(q, "TotalDebtOverEquity"),
                evOverEbitda=_fv(q, "EVOverEBITDA"),
                inventoryTurnover=_fv(q, "InventoryTurnover"),
                payoutRatio=_fv(q, "PayoutRatio"),
                cashDividend=_fv(q, "CashDividend"),
                shareAtPeriodEnd=_fv(q, "ShareAtPeriodEnd"),
            )

            if is_bank:
                base.update(
                    nim=_fv(q, "NIM"),
                    yoea=_fv(q, "YOEA"),
                    cof=_fv(q, "COF"),
                    cir=_fv(q, "CIR"),
                    ldr=_fv(q, "LDR"),
                    nplToLoan=_fv(q, "NPLToLoan"),
                    loanlossReservesToNPL=_fv(q, "LoanlossReservesToNPL"),
                )
                results.append(BankFinancialIndicator(**base))
            else:
                base.update(
                    lng=_fv(q, "GrossMargin"),
                    lnr=_fv(q, "ROS"),
                )
                results.append(NonBankFinancialIndicator(**base))

        return results, []

    # ------------------------------------------------------------------
    # Income Statements
    # ------------------------------------------------------------------

    def get_income_statement(
        self, symbol: str, is_bank: bool = False
    ) -> tuple[list[Any], list[str]]:
        quarters = self._fetch_quarters(symbol)
        if not quarters:
            return [], [f"No financial data from FireAnt for {symbol}"]

        results: list[Any] = []
        for q in quarters:
            year = q.get("year") or int((_fv(q, "Year") or 0))
            quarter = q.get("quarter") or int((_fv(q, "Quarter") or 0))
            if not year or not quarter:
                continue

            if is_bank:
                results.append(
                    BankIncomeStatement(
                        companyId=symbol,
                        year=year,
                        quarter=quarter,
                        profitAfterTax=_fv(q, "ParentCompanyShareholderProfitAfterTax"),
                        totalRevenue=_fv(q, "TotalRevenue"),
                        netProfit=_fv(q, "ParentCompanyShareholderProfitAfterTax"),
                        netInterestIncome=_fv(q, "NetInterestIncome"),
                        netFeeAndCommissionIncome=_fv(q, "NetProfitFromServiceActivity"),
                        netOtherIncomeOrExpenses=_fv(q, "OtherNetProfit"),
                        interestAndSimilarExpenses=_fv(q, "InterestAndSimilarExpense"),
                        totalOperatingIncome=_fv(q, "TotalOperatingIncome"),
                        totalOperatingExpense=_fv(q, "TotalOperatingExpense"),
                        creditRiskProvisionsExpense=_fv(q, "CreditRiskProvisionsExpense"),
                        interestAndSimilarIncome=_fv(q, "InterestAndSimilarIncome"),
                    )
                )
            else:
                results.append(
                    NonBankIncomeStatement(
                        companyId=symbol,
                        year=year,
                        quarter=quarter,
                        profitAfterTax=_fv(q, "ParentCompanyShareholderProfitAfterTax"),
                        totalRevenue=_fv(q, "TotalRevenue"),
                        netRevenue=_fv(q, "NetSale"),
                        netProfit=_fv(q, "ParentCompanyShareholderProfitAfterTax"),
                        grossProfit=_fv(q, "GrossProfit"),
                        costOfGoodsSold=_fv(q, "CostOfGoodSold"),
                        sellingExpense=_fv(q, "SellingExpense"),
                        managingExpense=_fv(q, "ManagingExpense"),
                    )
                )

        return results, []

    # ------------------------------------------------------------------
    # Balance Sheets
    # ------------------------------------------------------------------

    def get_balance_sheet(
        self, symbol: str, is_bank: bool = False
    ) -> tuple[list[Any], list[str]]:
        quarters = self._fetch_quarters(symbol)
        if not quarters:
            return [], [f"No financial data from FireAnt for {symbol}"]

        results: list[Any] = []
        for q in quarters:
            year = q.get("year") or int((_fv(q, "Year") or 0))
            quarter = q.get("quarter") or int((_fv(q, "Quarter") or 0))
            if not year or not quarter:
                continue

            base = dict(
                companyId=symbol,
                year=year,
                quarter=quarter,
                cashAndCashEquivalents=_sum(
                    _fv(q, "Cash"), _fv(q, "CashEquivalent")
                ),
                totalAssets=_fv(q, "TotalAsset"),
                equity=_fv(q, "StockHolderEquity"),
                totalCapital=_fv(q, "TotalCapital"),
                totalLiabilities=_fv(q, "TotalDebt"),
            )

            if is_bank:
                base.update(
                    balancesWithSbv=_fv(q, "DepositAtStateBank"),
                    interbankPlacementsAndLoans=_fv(q, "DepositAtAndLoanToOtherCreditInstitution"),
                    tradingSecurities=_fv(q, "TradingSecurities"),
                    investmentSecurities=_fv(q, "InvestmentSecurities"),
                    loansToCustomers=_fv(q, "CustomerLoanAfterProvision"),
                    govAndSbvDebt=_fv(q, "DebtToGovernmentAndStateBank"),
                    depositsBorrowingsOthers=_fv(q, "DepositAndBorrowingFromOtherCreditInstitution"),
                    depositsFromCustomers=_fv(q, "DepositOfCustomer"),
                    convertibleAndOtherPapers=_fv(q, "IssuingValuablePaper"),
                    customerLoan=_fv(q, "CustomerLoan"),
                    standardDebt=_fv(q, "StandardDebt"),
                    watchlistDebt=_fv(q, "WatchlistDebt"),
                    substandardDebt=_fv(q, "SubstandardDebt"),
                    doubtfulDebt=_fv(q, "DoubtfulDebt"),
                    badDebt=_fv(q, "BadDebt"),
                    provisionForCustomerLoanLoss=_fv(q, "ProvisionForCustomerLoanLoss"),
                    issuingValuablePaper=_fv(q, "IssuingValuablePaper"),
                    totalEquity=_fv(q, "TotalEquity"),
                )
                results.append(BankBalanceSheet(**base))
            else:
                base.update(
                    shortTermInvestments=_fv(q, "ShortTermFinancialInvestment"),
                    shortTermReceivables=_fv(q, "TotalShortTermReceivable"),
                    longTermReceivables=_fv(q, "TotalLongTermReceivable"),
                    inventories=_fv(q, "TotalInventory"),
                    fixedAssets=_fv(q, "FixedAsset"),
                    shortTermBorrowings=_fv(q, "ShortTermInterestBearingDebt"),
                    longTermBorrowings=_fv(q, "LongTermInterestBearingDebt"),
                    advancesFromCustomers=_fv(q, "ShortTermAccountPayable"),
                    inProgressLongTermAsset=_fv(q, "InProgressLongTermAsset"),
                    convertibleBond=_fv(q, "ConvertibleBond"),
                )
                results.append(NonBankBalanceSheet(**base))

        return results, []

    # ------------------------------------------------------------------
    # Cash Flow Statements
    # ------------------------------------------------------------------

    def get_cash_flow_statements(
        self, symbol: str
    ) -> tuple[list[CashFlowStatement], list[str]]:
        quarters = self._fetch_quarters(symbol)
        if not quarters:
            return [], [f"No financial data from FireAnt for {symbol}"]

        results: list[CashFlowStatement] = []
        for q in quarters:
            year = q.get("year") or int((_fv(q, "Year") or 0))
            quarter = q.get("quarter") or int((_fv(q, "Quarter") or 0))
            if not year or not quarter:
                continue

            op = _fv(q, "CashflowFromOperatingActivity")
            inv = _fv(q, "CashflowFromInvestingActivity")
            fin = _fv(q, "CashflowFromFinancingActivity")
            if op is None and inv is None and fin is None:
                continue

            results.append(
                CashFlowStatement(
                    companyId=symbol,
                    year=year,
                    quarter=quarter,
                    operatingCashflow=op,
                    investingCashflow=inv,
                    financingCashflow=fin,
                )
            )

        return results, []

    # ------------------------------------------------------------------
    # Company Meta
    # ------------------------------------------------------------------

    def get_company_overview_meta(
        self, symbol: str
    ) -> tuple[dict[str, str | None], list[str]]:
        return fetch_fireant_company_meta(symbol)

    # ------------------------------------------------------------------
    # Shareholders
    # ------------------------------------------------------------------

    def get_company_shareholders(
        self, symbol: str
    ) -> tuple[list[CompanyShareholderModel], list[str]]:
        raw = fetch_holders(symbol)
        if not raw:
            return [], []

        results: list[CompanyShareholderModel] = []
        for h in raw:
            name = (h.get("name") or h.get("shareholderName") or "").strip()
            if not name:
                continue
            results.append(
                CompanyShareholderModel(
                    companyId=symbol,
                    shareholderName=name,
                    quantity=h.get("quantity"),
                    shareOwnPercent=h.get("ownership") or h.get("shareOwnPercent"),
                    updateDate=h.get("reportDate") or h.get("updateDate"),
                )
            )
        return results, []

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------

    def get_company_dividends(
        self, symbol: str
    ) -> tuple[list[CompanyDividendModel], list[str]]:
        raw = fetch_dividends(symbol)
        if not raw:
            return [], []

        results: list[CompanyDividendModel] = []
        for d in raw:
            year = d.get("year")
            cash_div = d.get("cashDividend") or 0
            stock_div = d.get("stockDividend") or 0
            if not year or (cash_div == 0 and stock_div == 0):
                continue

            parts: list[str] = []
            if cash_div:
                parts.append(f"Cổ tức tiền mặt {cash_div:,.0f} đ/cp")
            if stock_div:
                parts.append(f"Cổ tức cổ phiếu {stock_div}%")
            title = f"{symbol} {year}: {'; '.join(parts)}"

            if cash_div and stock_div:
                event_type = "CASH_AND_STOCK"
            elif stock_div:
                event_type = "STOCK"
            else:
                event_type = "CASH"

            results.append(
                CompanyDividendModel(
                    companyId=symbol,
                    eventTitle=title,
                    eventType=event_type,
                    value=cash_div if cash_div else None,
                    ratio=f"{stock_div}%" if stock_div else None,
                )
            )
        return results, []

    # ------------------------------------------------------------------
    # Detect company type from first quarter
    # ------------------------------------------------------------------

    def detect_company_type(self, symbol: str) -> str | None:
        """Returns 'Bank' or 'General' or None based on FireAnt companyType."""
        data = fetch_financial_data(symbol, report_type="Q", count=1)
        if data and isinstance(data, list) and data[0]:
            return data[0].get("companyType")
        return None


def _sum(*values: float | None) -> float | None:
    """Sum non-None values; return None if all None."""
    nums = [v for v in values if v is not None]
    return sum(nums) if nums else None
