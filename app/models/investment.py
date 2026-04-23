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


class NonBankIncomeStatement(IncomeStatementBase):
    totalRevenue: Optional[float] = None
    netRevenue: Optional[float] = None
    netProfit: Optional[float] = None


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


class NonBankBalanceSheet(BalanceSheetBase):
    shortTermInvestments: Optional[float] = None
    shortTermReceivables: Optional[float] = None
    longTermReceivables: Optional[float] = None
    inventories: Optional[float] = None
    fixedAssets: Optional[float] = None
    shortTermBorrowings: Optional[float] = None
    longTermBorrowings: Optional[float] = None
    advancesFromCustomers: Optional[float] = None


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


class BankFinancialIndicator(FinancialIndicatorBase):
    pass


class NonBankFinancialIndicator(FinancialIndicatorBase):
    lng: Optional[float] = None
    lnr: Optional[float] = None


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
