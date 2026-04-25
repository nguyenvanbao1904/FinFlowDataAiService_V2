from pydantic import BaseModel
from typing import Optional


class IncomeStatementBase(BaseModel):
    companyId: str
    year: int
    quarter: int
    profitAfterTax: Optional[float] = None


class BalanceSheetBase(BaseModel):
    companyId: str
    year: int
    quarter: int
    cashAndCashEquivalents: Optional[float] = None
    totalAssets: Optional[float] = None
    equity: Optional[float] = None
    totalCapital: Optional[float] = None
    totalLiabilities: Optional[float] = None


class BankIncomeStatement(IncomeStatementBase):
    totalRevenue: Optional[float] = None
    netProfit: Optional[float] = None
    netInterestIncome: Optional[float] = None
    netFeeAndCommissionIncome: Optional[float] = None
    netOtherIncomeOrExpenses: Optional[float] = None
    interestAndSimilarExpenses: Optional[float] = None
    # New fields for FireAnt
    totalOperatingIncome: Optional[float] = None
    totalOperatingExpense: Optional[float] = None
    creditRiskProvisionsExpense: Optional[float] = None
    interestAndSimilarIncome: Optional[float] = None


class NonBankIncomeStatement(IncomeStatementBase):
    totalRevenue: Optional[float] = None
    netRevenue: Optional[float] = None
    netProfit: Optional[float] = None
    # New fields for FireAnt
    grossProfit: Optional[float] = None
    costOfGoodsSold: Optional[float] = None
    sellingExpense: Optional[float] = None
    managingExpense: Optional[float] = None


class BankBalanceSheet(BalanceSheetBase):
    balancesWithSbv: Optional[float] = None
    interbankPlacementsAndLoans: Optional[float] = None
    tradingSecurities: Optional[float] = None
    investmentSecurities: Optional[float] = None
    loansToCustomers: Optional[float] = None
    govAndSbvDebt: Optional[float] = None
    depositsBorrowingsOthers: Optional[float] = None
    depositsFromCustomers: Optional[float] = None
    convertibleAndOtherPapers: Optional[float] = None
    # New fields for FireAnt
    customerLoan: Optional[float] = None
    standardDebt: Optional[float] = None
    watchlistDebt: Optional[float] = None
    substandardDebt: Optional[float] = None
    doubtfulDebt: Optional[float] = None
    badDebt: Optional[float] = None
    provisionForCustomerLoanLoss: Optional[float] = None
    issuingValuablePaper: Optional[float] = None
    totalEquity: Optional[float] = None


class NonBankBalanceSheet(BalanceSheetBase):
    shortTermInvestments: Optional[float] = None
    shortTermReceivables: Optional[float] = None
    longTermReceivables: Optional[float] = None
    inventories: Optional[float] = None
    fixedAssets: Optional[float] = None
    shortTermBorrowings: Optional[float] = None
    longTermBorrowings: Optional[float] = None
    advancesFromCustomers: Optional[float] = None
    # New fields for FireAnt
    inProgressLongTermAsset: Optional[float] = None
    convertibleBond: Optional[float] = None


class FinancialIndicatorBase(BaseModel):
    companyId: str
    year: int
    quarter: int
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    eps: Optional[float] = None
    bvps: Optional[float] = None
    cplh: Optional[float] = None
    # New common fields for FireAnt
    grossMargin: Optional[float] = None
    netMargin: Optional[float] = None
    saleGrowth: Optional[float] = None
    profitGrowth: Optional[float] = None
    currentRatio: Optional[float] = None
    totalDebtOverEquity: Optional[float] = None
    evOverEbitda: Optional[float] = None
    inventoryTurnover: Optional[float] = None
    payoutRatio: Optional[float] = None
    cashDividend: Optional[float] = None
    shareAtPeriodEnd: Optional[float] = None


class BankFinancialIndicator(FinancialIndicatorBase):
    nim: Optional[float] = None
    yoea: Optional[float] = None
    cof: Optional[float] = None
    cir: Optional[float] = None
    ldr: Optional[float] = None
    nplToLoan: Optional[float] = None
    loanlossReservesToNPL: Optional[float] = None


class NonBankFinancialIndicator(FinancialIndicatorBase):
    lng: Optional[float] = None
    lnr: Optional[float] = None


class CashFlowStatement(BaseModel):
    companyId: str
    year: int
    quarter: int
    operatingCashflow: Optional[float] = None
    investingCashflow: Optional[float] = None
    financingCashflow: Optional[float] = None


class IndustryNodeModel(BaseModel):
    id: str
    parentId: Optional[str] = None
    nameVi: str
    level: int
    icbCode: Optional[str] = None
    detailLabel: Optional[str] = None


class CompanyModel(BaseModel):
    id: str
    exchange: str
    industryNodeId: Optional[str] = None
    industryIcbCode: Optional[str] = None
    companyName: Optional[str] = None
    description: Optional[str] = None
    companyType: str


class CompanyShareholderModel(BaseModel):
    companyId: str
    shareholderName: str
    quantity: Optional[int] = None
    shareOwnPercent: Optional[float] = None
    updateDate: Optional[str] = None


class CompanyDividendModel(BaseModel):
    companyId: str
    eventTitle: str
    eventType: str
    ratio: Optional[str] = None
    value: Optional[float] = None
    recordDate: Optional[str] = None
    exrightDate: Optional[str] = None
    issueDate: Optional[str] = None
