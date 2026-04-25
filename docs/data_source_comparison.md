# So sanh Data Sources: FireAnt vs vnstock (free) vs vnstock_data (paid)

> **Ngay tao:** 2026-04-23
> **Muc dich:** So sanh chi tiet 3 nguon du lieu tai chinh VN de chon thay the vnstock (da hong) cho FinFlow.

---

## 1. Tong quan

| Tieu chi | FireAnt REST v2 | vnstock (free) | vnstock_data (paid) |
|----------|-----------------|----------------|---------------------|
| **Gia** | Mien phi | Mien phi | ~189k/thang |
| **Trang thai (04/2026)** | Hoat dong tot | **HONG** (VCI GraphQL tra ve rong, KBS SAS 404) | Hoat dong (package rieng, khong tren PyPI) |
| **Phu thuoc** | HTTP REST thuan — tu viet adapter | vnstock library (2,300+ lines) | vnstock_data library (closed source) |
| **Xac thuc** | Bearer token (het han 2074) | API key (chi tang rate limit, khong mo data) | License key (189k/thang) |
| **Rate limit** | Khong ro rang, da test burst 50 req ok | Guest: 20 req/min, Free: 60 req/min | Khong ro |
| **Format** | JSON REST | pandas DataFrame | pandas DataFrame |
| **Lich su** | **85 quy** (VNM tu Q4/2003) | 4-8 quy (free tier) | Nhieu nam (theo quang cao) |
| **Bulk download** | Co (`/symbols/all-financial-data`) | Khong | Khong ro |
| **Bank vs Non-bank** | Tu dong phan biet (CompanyType field) | Can truyen tham so `is_bank` | Tuong tu vnstock |

---

## 2. Do sau lich su (Historical Depth)

### FireAnt
- **VNM (non-bank):** 85 quy, tu ~Q4/2004 den Q4/2025
- **ACB (bank):** 82 quy, tu ~Q2/2005 den Q4/2025
- Endpoint: `GET /symbols/{symbol}/financial-data?type=Q&count=100`

### vnstock (free)
- VCI GraphQL: tra ve `{}` (da hong tu ~2026)
- KBS SAS: 404 Not Found
- Thuc te: **0 quy** (khong hoat dong)

### vnstock_data (paid)
- Quang cao "du lieu day du", nhung phu thuoc package closed-source
- Khong kiem chung duoc do khong co license

---

## 3. So sanh chi tiet Field Mapping

### 3.1 Income Statement — Non-bank

| Field trong FinFlow DB | FireAnt field | vnstock field | Ghi chu |
|------------------------|---------------|---------------|---------|
| `totalRevenue` | `TotalRevenue` | ISA1 (mapped) | 1:1 |
| `netRevenue` | `NetSale` | ISA2 (mapped) | 1:1 |
| `costOfGoodsSold` | `CostOfGoodSold` | ISA3 (mapped) | 1:1 |
| `grossProfit` | `GrossProfit` | ISA4 (mapped) | 1:1 |
| `sellingExpenses` | `SellingExpense` | ISA7 (mapped) | 1:1 |
| `managementExpenses` | `ManagingExpense` | ISA8 (mapped) | 1:1 |
| `operatingProfit` | `NetProfitFromOperatingActivity` | ISA17 (mapped) | 1:1 |
| `profitBeforeTax` | `ProfitBeforeTax` | ISA20 (mapped) | 1:1 |
| `profitAfterTax` | `ProfitAfterTax` | ISA22 (mapped) | 1:1 |
| `netProfit` (LNST co dong cong ty me) | `ParentCompanyShareholderProfitAfterTax` | ISA23 (mapped) | 1:1 |
| — | `FinancialExpense` | ISA9 (mapped) | **MOI** — FireAnt co |
| — | `InterestExpense` | ISA10 (mapped) | **MOI** — FireAnt co |
| — | `OtherRevenue` / `OtherExpense` | ISA18/ISA19 | **MOI** — FireAnt co |

### 3.2 Income Statement — Bank

| Field trong FinFlow DB | FireAnt field | vnstock field |
|------------------------|---------------|---------------|
| `totalRevenue` | `TotalRevenue` | ISB25 |
| `netInterestIncome` | `NetInterestIncome` | ISB26 - ISB27 |
| `netFeeAndCommissionIncome` | `NetProfitFromServiceActivity` | ISB29 |
| `interestAndSimilarExpenses` | `InterestAndSimilarExpense` | ISB27 |
| `profitBeforeTax` | `ProfitBeforeTax` | ISB38 |
| `profitAfterTax` | `ProfitAfterTax` | ISB41 |
| `netProfit` | `ParentCompanyShareholderProfitAfterTax` | ISB41 |
| `netOtherIncomeOrExpenses` | `OtherNetProfit` | ISB34 |
| — | `TotalOperatingIncome` | ISB35 | **MOI** |
| — | `TotalOperatingExpense` | ISB36 | **MOI** |
| — | `CreditRiskProvisionsExpense` | ISB37 | **MOI** |
| — | `NetProfitFromTradingOfTradingSecurities` | ISB30 | **MOI** |
| — | `NetProfitFromTradingOfInvestmentSecurities` | ISB31 | **MOI** |

### 3.3 Balance Sheet — Non-bank

| Field trong FinFlow DB | FireAnt field | vnstock field |
|------------------------|---------------|---------------|
| `totalAssets` | `TotalAsset` | BSA1 |
| `totalLiabilities` | `TotalDebt` | BSA50 |
| `equity` | `StockHolderEquity` | BSA78 |
| `totalCapital` | `TotalCapital` | BSA96 |
| `cashAndCashEquivalents` | `Cash` + `CashEquivalent` | BSA2 + BSA5 |
| `shortTermInvestments` | `ShortTermFinancialInvestment` | BSA8 |
| `shortTermReceivables` | `TotalShortTermReceivable` | BSA10 |
| `inventories` | `TotalInventory` | BSA22 |
| `fixedAssets` | `FixedAsset` | BSA43 |
| `longTermReceivables` | `TotalLongTermReceivable` | BSA29 |
| `shortTermBorrowings` | `ShortTermInterestBearingDebt` | BSA53 |
| `longTermBorrowings` | `LongTermInterestBearingDebt` | BSA67 |
| — | `TangibleAsset` | BSA44 | **MOI** |
| — | `IntangibleAsset` | BSA46 | **MOI** |
| — | `GoodWill` | BSA48 | **MOI** |
| — | `RealEstateInvest` | BSA50 | **MOI** |
| — | `WorkingCapital` | computed | **MOI** |
| — | `TreasuryStock` | BSA85 | **MOI** |
| — | `PaidInCapital` | BSA79 | **MOI** |
| — | `RetainedProfit` | BSA90 | **MOI** |

### 3.4 Balance Sheet — Bank

| Field trong FinFlow DB | FireAnt field | vnstock field |
|------------------------|---------------|---------------|
| `totalAssets` | `TotalAsset` | BSB97 |
| `balancesWithSbv` | `DepositAtStateBank` | BSB98 |
| `interbankPlacementsAndLoans` | `DepositAtAndLoanToOtherCreditInstitution` | BSB99 |
| `loansToCustomers` | `CustomerLoanAfterProvision` | BSB103 |
| `tradingSecurities` | `TradingSecurities` | BSB101 |
| `investmentSecurities` | `InvestmentSecurities` | BSB105 |
| `depositsFromCustomers` | `DepositOfCustomer` | BSB110 |
| `depositsBorrowingsOthers` | `DepositAndBorrowingFromOtherCreditInstitution` | BSB108 |
| `govAndSbvDebt` | `DebtToGovernmentAndStateBank` | BSB109 |
| `convertibleAndOtherPapers` | `IssuingValuablePaper` | BSB112 |
| — | `TotalInterestEarningAsset` | computed | **MOI** |
| — | `TotalInterestBearingDebt` | computed | **MOI** |
| — | `StandardDebt` / `WatchlistDebt` / `SubstandardDebt` / `DoubtfulDebt` / `BadDebt` | — | **MOI** — NPL breakdown |
| — | `ProvisionForCustomerLoanLoss` | BSB104 | **MOI** |
| — | `CharterCapital` | BSB117 | **MOI** |

### 3.5 Financial Indicators

| Field trong FinFlow DB | FireAnt field | vnstock field |
|------------------------|---------------|---------------|
| `pe` | `PE` | pe |
| `pb` | `PB` | pb |
| `ps` | `PS` | ps |
| `roe` | `ROE` | roe |
| `roa` | `ROA` | roa |
| `eps` | `BasicEPS` | eps |
| `bvps` | `BookValuePerShare` | bvps |
| `grossMargin` (lng) | `GrossMargin` | grossMargin |
| `netMargin` (lnr) | `ROS` | netProfitMargin |
| — | `EBIT` / `EBITDA` | ebit / ebitda | **MOI** |
| — | `CurrentRatio` / `QuickRatio` / `CashRatio` | currentRatio / quickRatio / cashRatio | **MOI** |
| — | `InterestCoverageRatio` | interestCoverage | **MOI** |
| — | `TotalDebtOverEquity` / `TotalDebtOverAsset` | de / — | **MOI** |
| — | `ROIC` / `ROCE` | — / — | **MOI** |
| — | `DividendYield` | dividend | **MOI** |
| — | `EVOverEBITDA` | evPerEbitda | **MOI** |

### 3.6 Bank-specific Indicators (chi FireAnt)

| FireAnt field | Mo ta | Y nghia |
|---------------|-------|---------|
| `NIM` | Net Interest Margin | Bien lai rong |
| `YOEA` | Yield on Earning Assets | Loi suat tai san sinh loi |
| `COF` | Cost of Funds | Chi phi von |
| `CIR` | Cost-to-Income Ratio | Ty le chi phi/thu nhap |
| `CLR` | Credit Loss Rate | Ty le mat von tin dung |
| `LDR` | Loan-to-Deposit Ratio | Ty le cho vay/huy dong |
| `LAR` | Loan-to-Asset Ratio | Ty le cho vay/tong tai san |
| `NPLToLoan` | Non-Performing Loan ratio | Ty le no xau |
| `LoanlossReservesToNPL` | Provision coverage | Ty le bao phu du phong |
| `EquityToLoan` | Equity/Loan ratio | Von chu so huu/cho vay |
| `EquityToDeposit` | Equity/Deposit ratio | Von CSH/tien gui |

---

## 4. FireAnt Exclusive APIs (vnstock KHONG co)

### 4.1 Estimated Price (Dinh gia tu dong)
```
GET /symbols/{symbol}/estimated-price
```
Tra ve: **DCF, Graham Number, PE-based, PB-based** estimated fair values.
- **Use case:** Tinh nang dinh gia tu dong trong chatbot, hien thi tren UI.

### 4.2 Dynamic Financial Data
```
GET /symbols/{symbol}/dynamic-financial-data?type=Q&year=2025&quarter=4
```
Tra ve: Gom nhieu du lieu khac nhau theo ky (quarterly/annual).

### 4.3 All Financial Data (Bulk)
```
GET /symbols/all-financial-data?type=Q&year=2025&quarter=4
```
Tra ve: **TAT CA cong ty** trong 1 API call duy nhat.
- **Use case:** Batch crawler chi can 1 call thay vi N calls (1 per symbol). Toc do crawl tang gap 100-1000x.

### 4.4 ICB Industry Classification
```
GET /symbols/{symbol}/icb
```
Tra ve: Phan nganh ICB day du (level 1-4).

### 4.5 Holder Transactions
```
GET /symbols/{symbol}/holder-transactions
```
Tra ve: Giao dich mua/ban cua co dong lon, noi bo.
- **Use case:** Phan tich insider trading, tin hieu mua/ban.

### 4.6 Timescale Marks
```
GET /symbols/{symbol}/timescale-marks
```
Tra ve: Su kien quan trong (chia co tuc, phat hanh them, DHDCD...).

### 4.7 Fundamental
```
GET /symbols/{symbol}/fundamental
```
Tra ve: Du lieu co ban tong hop.

### 4.8 Sector Benchmarks (co san trong financial-data)
FireAnt tra ve **50+ sector benchmarks** trong cung 1 API call:
- `SectorPE`, `SectorPB`, `SectorPS`, `SectorEPS`
- `SectorROE`, `SectorROA`, `SectorROCE`, `SectorROIC`
- `SectorGrossMargin`, `SectorEBITMargin`, `SectorOperatingMargin`
- `SectorCurrentRatio`, `SectorInterestCoverageRatio`
- `SectorTotalDebtOverEquity`, `SectorTotalAssetTurnover`
- `SectorInventoryTurnover`, `SectorReceivableTurnover`
- `SectorNetSale`, `SectorTotalAsset`, `SectorTotalDebt`...

---

## 5. Du lieu moi co the dua vao FinFlow

### 5.1 HIGH Priority (anh huong truc tiep den phan tich)

| Du lieu | FireAnt field | Hien trang FinFlow | De xuat |
|---------|---------------|--------------------|---------|
| **EBIT / EBITDA** | `EBIT`, `EBITDA`, `CoreEBIT`, `CoreEBITDA` | Chua co | Them vao `financial_indicators` table |
| **EV/EBITDA** | `EVOverEBITDA` | Chua co | Them vao indicators — chi so dinh gia quan trong |
| **Sector benchmarks** | `SectorPE`, `SectorPB`, `SectorROE`... | Chua co | Tao bang `sector_benchmarks` hoac embed vao indicators |
| **Growth metrics** | `SaleGrowth`, `ProfitGrowth`, `ProfitGrowth_TTM`... | Chua co | Them vao indicators — du lieu tang truong |
| **Liquidity ratios** | `CurrentRatio`, `QuickRatio`, `CashRatio` | Chua co | Them vao indicators |
| **Leverage ratios** | `TotalDebtOverEquity`, `TotalDebtOverAsset`, `InterestCoverageRatio` | Chua co | Them vao indicators |
| **NPL breakdown (bank)** | `StandardDebt`, `WatchlistDebt`, `SubstandardDebt`, `DoubtfulDebt`, `BadDebt` | Chua co | Them vao bank_balance_sheets |
| **Bank-specific ratios** | `NIM`, `CIR`, `LDR`, `NPLToLoan`, `LoanlossReservesToNPL` | Chua co | Them vao bank financial indicators |

### 5.2 MEDIUM Priority (nang cao trai nghiem)

| Du lieu | FireAnt field | De xuat |
|---------|---------------|---------|
| **Estimated fair value** | `/estimated-price` (DCF, Graham, PE, PB) | Hien thi tren UI + dung trong chatbot valuation |
| **TTM metrics** | `*_TTM` (trailing twelve months) | Phan tich xu huong 12 thang |
| **Cash flow data** | `CashflowFromOperatingActivity`, `CashflowFromInvestingActivity`, `CashflowFromFinancingActivity` | Tao bang `cash_flow_statements` |
| **CAPEX** | `CAPEX` | Them vao cash flow |
| **Dividend data** | `CashDividend`, `StockDividend`, `DividendYield`, `PayoutRatio` | Mo rong bang dividends |
| **Z-Score / Credit Rating** | `ManufacturingZScore`, `ManufacturingSPRating`, `ManufacturingMoodyRating` | Them vao indicators — danh gia rui ro |

### 5.3 LOW Priority (nice-to-have)

| Du lieu | FireAnt field | De xuat |
|---------|---------------|---------|
| **Holder transactions** | `/holder-transactions` | Tinh nang insider tracking |
| **Timescale marks** | `/timescale-marks` | Su kien cong ty tren timeline |
| **Planning data** | `PlanningRevenue`, `PlanningProfitBeforeTax`, `PlanningEPS` | So sanh ke hoach vs thuc te |
| **Average metrics** | `Avg*` (AvgTotalAsset, AvgROE...) | Phan tich xu huong trung binh |
| **ICB classification** | `/icb` | Phan nganh chi tiet |
| **Market cap** | `MarketCapAtPeriodEnd`, `AvgMarketCapInPeriod` | Hien thi tren UI |

---

## 6. Tong hop so luong fields

| Nguon | Non-bank | Bank | Ghi chu |
|-------|----------|------|---------|
| **FireAnt financial-data** | **340 fields** | **298 fields** | 1 API call, bao gom income + balance + indicators + growth + sector |
| vnstock VCI (khi con hoat dong) | ~80 fields/bao cao | ~80 fields/bao cao | 3 API calls rieng cho income, balance, ratio |
| vnstock KBS (khi con hoat dong) | ~60 fields/bao cao | ~60 fields/bao cao | 4 API calls rieng |
| **FinFlow DB hien tai** | ~25 fields | ~30 fields | Chi luu cac truong chinh |
| **FinFlow DB tiem nang (voi FireAnt)** | **~80+ fields** | **~90+ fields** | Tang 3x du lieu |

---

## 7. Ket luan va Khuyen nghi

### Ket luan
1. **vnstock (free) da CHET** — VCI GraphQL tra ve rong, KBS SAS 404. Khong the phuc hoi.
2. **vnstock_data (paid)** — Hoat dong nhung phu thuoc closed-source package, 189k/thang, van la vnstock ecosystem.
3. **FireAnt REST v2** — **Lua chon tot nhat**:
   - Mien phi, token het han 2074
   - Du lieu phong phu nhat (340 fields vs ~80 fields)
   - Lich su sau nhat (85 quy vs 4-8 quy)
   - Bulk API (`all-financial-data`) cho batch crawl cuc nhanh
   - Khong phu thuoc thu vien ben ngoai — tu viet HTTP adapter
   - Da co token trong `.env` va `fetch_fireant_company_meta` dang hoat dong

### Khuyen nghi hanh dong
1. **Viet `FireAntCrawlerService`** thay the `VnstockCrawlerService`
   - Map 340 fields vao cac entity hien tai
   - Su dung bulk API cho batch crawl
2. **Mo rong DB schema** de luu them du lieu moi (EBITDA, sector benchmarks, growth metrics, bank ratios)
3. **Loai bo dependency vnstock** hoan toan khoi project
4. **Tich hop estimated-price API** vao chatbot valuation engine
