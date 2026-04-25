# FinFlow Chart Requirements — FireAnt Data Mapping

> **Ngay tao:** 2026-04-23
> **Nguon tham khao:** Finbook (finbook.vn)
> **Data source:** FireAnt REST v2 (`/symbols/{symbol}/financial-data`)

---

## A. NON-BANK: 12 bieu do

### 1. TAI SAN (stacked bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Tien & tuong duong, DTTCNH thuan | `Cash` + `CashEquivalent` + `ShortTermFinancialInvestment` | Cong 3 truong |
| Cac khoan phai thu | `TotalShortTermReceivable` + `TotalLongTermReceivable` | Cong 2 truong |
| Hang ton kho | `TotalInventory` | Truc tiep |
| Tai san co dinh | `FixedAsset` | Truc tiep |
| TS do dang dai han | `InProgressLongTermAsset` | Truc tiep |
| Khac | `TotalAsset` - tong cac muc tren | Computed |
| **Line:** Khoan phai thu/TS (%) | (`TotalShortTermReceivable` + `TotalLongTermReceivable`) / `TotalAsset` | Computed |

**Trang thai:** ✅ 100%

---

### 2. NGUON VON (stacked bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Von chu so huu | `StockHolderEquity` | Truc tiep |
| Nguoi mua tra truoc + DT chua thuc hien | `ShortTermAccountPayable` | Gan dung (bao gom advances + unearned) |
| Vay ngan han | `ShortTermInterestBearingDebt` | Truc tiep |
| Vay dai han + TP chuyen doi | `LongTermInterestBearingDebt` + `ConvertibleBond` | Cong 2 truong |
| Khac | `TotalCapital` - tong cac muc tren | Computed |
| **Line:** No vay rong/VCSH (%) | (`ShortTermInterestBearingDebt` + `LongTermInterestBearingDebt` - `Cash` - `CashEquivalent`) / `StockHolderEquity` | Computed |

**Trang thai:** ✅ 100%

---

### 3. ROE VA ROA (dual line)
| Thanh phan | FireAnt field |
|------------|---------------|
| ROE cuoi ky (%) | `ROE` |
| ROA cuoi ky (%) | `ROA` |

**Trang thai:** ✅ 100%

---

### 4. DOANH THU THUAN (bar + line)
| Thanh phan | FireAnt field |
|------------|---------------|
| Doanh so (bar) | `NetSale` |
| Tang truong DT thuan YoY (%) | `SaleGrowth` |

**Trang thai:** ✅ 100%

---

### 5. LNST (bar + line)
| Thanh phan | FireAnt field |
|------------|---------------|
| LNST co dong cty me (bar) | `ParentCompanyShareholderProfitAfterTax` |
| Tang truong LNST YoY (%) | `ProfitGrowth` |

**Trang thai:** ✅ 100%

---

### 6. BIEN LNG VA BIEN LNR (dual line)
| Thanh phan | FireAnt field |
|------------|---------------|
| Bien loi nhuan gop (%) | `GrossMargin` |
| Bien loi nhuan rong (%) | `ROS` |

**Trang thai:** ✅ 100%

---

### 7. LUU CHUYEN TIEN TE (stacked bar + line)
| Thanh phan | FireAnt field |
|------------|---------------|
| LCTT hoat dong kinh doanh | `CashflowFromOperatingActivity` |
| LCTT hoat dong dau tu | `CashflowFromInvestingActivity` |
| LCTT hoat dong tai chinh | `CashflowFromFinancingActivity` |
| **Line:** Luu chuyen tien thuan | Tong 3 dong tren | Computed |

**Trang thai:** ✅ 100%

---

### 8. VONG QUAY HANG TON KHO (bar + line)
| Thanh phan | FireAnt field |
|------------|---------------|
| Hang ton kho (bar) | `TotalInventory` |
| Vong quay HTK (lan) | `InventoryTurnover` |

**Trang thai:** ✅ 100%

---

### 9. CO TUC (bar + line)
| Thanh phan | FireAnt field |
|------------|---------------|
| LNST co dong cty me (bar) | `ParentCompanyShareholderProfitAfterTax` |
| Co tuc da tra (bar) | `CashDividend` * `ShareAtPeriodEnd` |
| **Line:** Payout ratio (%) | `PayoutRatio` |

**Trang thai:** ✅ 100%

---

### 10. PE LICH SU (line)
| Thanh phan | FireAnt field |
|------------|---------------|
| PE theo quy | `PE` |

**Trang thai:** ✅ 100%

---

### 11. PB LICH SU (line)
| Thanh phan | FireAnt field |
|------------|---------------|
| PB theo quy | `PB` |

**Trang thai:** ✅ 100%

---

### 12. PS LICH SU (line)
| Thanh phan | FireAnt field |
|------------|---------------|
| PS theo quy | `PS` |

**Trang thai:** ✅ 100%

---

### KET QUA NON-BANK: 12/12 ✅ (100%)

---

## B. BANK: 15 bieu do

> **Nguyen tac:** Chi giu nhung chart co du data tu FireAnt hoac tinh duoc.
> Nhung chart can data chi tiet tu thuyet minh BCTC (khong co trong bat ky API free nao) → BO.

### Cac chart BI BO va ly do:

| Chart goc (Finbook) | Ly do bo |
|----------------------|----------|
| **Co cau cho vay theo ky han** (ngan han vs trung dai han) | FireAnt khong chia cho vay theo ky han. Day la du lieu thuyet minh BCTC, khong API nao co. Rui ro ky han da duoc phan anh qua **LDR** (Loan-to-Deposit Ratio) — FireAnt co san. |
| **Co cau cho vay theo nhom KH** (ca nhan vs to chuc) | Tuong tu — chi co trong thuyet minh BCTC. Chat luong tin dung da duoc the hien qua **NPL breakdown** (nhom 3/4/5) — FireAnt co san. |
| **CASA** | FireAnt khong tach tien gui khong ky han vs co ky han. Tuy nhien **COF** (Cost of Funds) la proxy truc tiep: CASA cao → COF thap. FireAnt co COF san, nen CASA chart thua. |

---

### 1. SO SANH TRONG NGANH NGAN HANG (radar chart)
| Truc radar | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Tang truong cho vay KH | `CustomerLoan` so voi ky truoc | Computed |
| Tong TS/VCSH (lan) | `TotalAsset` / `TotalEquity` | Computed |
| Ty le No (2->5) | (`WatchlistDebt` + `SubstandardDebt` + `DoubtfulDebt` + `BadDebt`) / `CustomerLoan` | Computed |
| Ty le No xau (3->5) = NPL | `NPLToLoan` | Truc tiep |
| Du phong RRTD/No xau | `LoanlossReservesToNPL` | Truc tiep |
| NIM | `NIM` | Truc tiep |
| COF | `COF` | Truc tiep |
| ROE (cuoi ky) | `ROE` | Truc tiep |

> So voi Finbook goc: bo CASA (da co COF), bo "Lai phi phai thu/Tong TS" (khong co truc tiep),
> bo "Du phong RRTD/No(2->5)" (trung voi Du phong/No xau). Giam tu 11 → 8 truc, radar van du y nghia.

**Trang thai:** ✅ 100%

---

### 2. NO XAU — NHOM 3->5 (bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Ti le No xau (3->5) (%) — bar | `NPLToLoan` | Truc tiep |
| Du phong RRTD/No xau (%) — line | `LoanlossReservesToNPL` | Truc tiep |

> Bo: du phong chung vs cu the (khong API nao co). Giu tong du phong coverage — du de danh gia suc khoe tin dung.

**Trang thai:** ✅ 100%

---

### 3. NO NHOM 2->5 (bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Ti le No (2->5) (%) — bar | (`WatchlistDebt` + NPL) / `CustomerLoan` | Computed |
| Du phong RRTD/No (2->5) (%) — line | `ProvisionForCustomerLoanLoss` / (WatchlistDebt + NPL) | Computed |

> Tuong tu chart 2, bo chung/cu the, giu tong coverage.

**Trang thai:** ✅ 100%

---

### 4. CO CAU NO XAU (multi-line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Ty le No nhom 3 (%) | `SubstandardDebt` / `CustomerLoan` | Computed |
| Ty le No nhom 4 (%) | `DoubtfulDebt` / `CustomerLoan` | Computed |
| Ty le No nhom 5 (%) | `BadDebt` / `CustomerLoan` | Computed |

**Trang thai:** ✅ 100%

---

### 5. TAI SAN BANK (stacked bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Cho vay khach hang | `CustomerLoanAfterProvision` | Truc tiep |
| Chung khoan dau tu | `InvestmentSecurities` | Truc tiep |
| Khac | `TotalAsset` - tren | Computed |
| **Line:** Lai phi phai thu/Tong TS (%) | `InterestAndSimilarIncome` / `TotalAsset` | Computed (proxy) |

> Lai phi phai thu/Tong TS: dung InterestAndSimilarIncome/TotalAsset lam proxy.
> Khong chinh xac 100% (income vs receivable) nhung the hien cung xu huong.

**Trang thai:** ✅ 100% (line la proxy)

---

### 6. NGUON VON BANK (stacked bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Tien gui cua KH | `DepositOfCustomer` | Truc tiep |
| Phat hanh GTCG | `IssuingValuablePaper` | Truc tiep |
| Khac | `TotalDebt` - tren | Computed |
| **Line:** TS/VCSH (lan) | `TotalAsset` / `TotalEquity` | Computed |

**Trang thai:** ✅ 100%

---

### 7. ROE VA ROA (dual line)
| Thanh phan | FireAnt field |
|------------|---------------|
| ROE cuoi ky (%) | `ROE` |
| ROA cuoi ky (%) | `ROA` |

**Trang thai:** ✅ 100%

---

### 8. CO CAU TOI (stacked bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Thu nhap lai thuan | `NetInterestIncome` | Truc tiep |
| Lai thuan HD dich vu | `NetProfitFromServiceActivity` | Truc tiep |
| Thu nhap HD khac | `TotalOperatingIncome` - 2 muc tren | Computed |
| **Line:** Tang truong TOI YoY (%) | `TotalOperatingIncome` so voi ky truoc | Computed |

**Trang thai:** ✅ 100%

---

### 9. LNST (bar + line)
| Thanh phan | FireAnt field |
|------------|---------------|
| LNST co dong cty me (bar) | `ParentCompanyShareholderProfitAfterTax` |
| Tang truong LNST YoY (%) | `ProfitGrowth` |

**Trang thai:** ✅ 100%

---

### 10. CHI SO SINH LOI (triple line)
| Thanh phan | FireAnt field |
|------------|---------------|
| NIM (%) | `NIM` |
| Lai cho vay YOEA (%) | `YOEA` |
| Chi phi von COF (%) | `COF` |

**Trang thai:** ✅ 100%

---

### 11. CHO VAY KHACH HANG (bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| Cho vay KH (bar) | `CustomerLoan` | Truc tiep |
| Tang truong cho vay YoY (%) | `CustomerLoan` so voi ky truoc | Computed |

**Trang thai:** ✅ 100%

---

### 12. CO TUC (bar + line)
| Thanh phan | FireAnt field | Cach tinh |
|------------|---------------|-----------|
| LNST co dong cty me (bar) | `ParentCompanyShareholderProfitAfterTax` | Truc tiep |
| Co tuc da tra (bar) | `CashDividend` * `ShareAtPeriodEnd` | Computed |
| **Line:** Payout ratio (%) | `PayoutRatio` | Truc tiep |

**Trang thai:** ✅ 100%

---

### 13. PE LICH SU (line)
| Thanh phan | FireAnt field |
|------------|---------------|
| PE theo quy | `PE` |

**Trang thai:** ✅ 100%

---

### 14. PB LICH SU (line)
| Thanh phan | FireAnt field |
|------------|---------------|
| PB theo quy | `PB` |

**Trang thai:** ✅ 100%

---

### 15. PS LICH SU (line)
| Thanh phan | FireAnt field |
|------------|---------------|
| PS theo quy | `PS` |

**Trang thai:** ✅ 100%

---

### KET QUA BANK: 15/15 ✅ (100%)

> Goc Finbook co 18 chart. Bo 3 chart can thuyet minh BCTC (cho vay theo ky han, cho vay theo nhom KH, CASA).
> Radar chart giam tu 11 → 8 truc (bo CASA — da co COF, bo 2 truc trung lap).
> Tat ca 15 chart con lai deu co du data tu FireAnt.

---

## C. TONG KET

| Nhom | So bieu do | FireAnt dap ung | Ghi chu |
|------|------------|-----------------|---------|
| Non-bank | 12 | **12/12 (100%)** | |
| Bank | 15 (goc 18, bo 3) | **15/15 (100%)** | Bo 3 chart can thuyet minh BCTC |
| **Tong** | **27** | **27/27 (100%)** | |

### 3 chart bank bi bo:
1. **Co cau cho vay theo ky han** → LDR da phan anh rui ro ky han
2. **Co cau cho vay theo nhom KH** → NPL breakdown da phan anh chat luong tin dung
3. **CASA** → COF la proxy truc tiep (CASA cao = COF thap)

### Fields can tinh (computed) — khong co truc tiep tu FireAnt:
- Tang truong YoY cho cac metric (so sanh `field` quy nay vs cung ky nam truoc)
- No nhom 2->5 ratio (cong WatchlistDebt + NPL)
- Net debt / Equity (non-bank)
- Khoan phai thu / Tong TS (non-bank)
- Cac muc "Khac" trong stacked bar (TotalAsset - cac thanh phan cu the)

> Tat ca computed fields deu don gian (cong/tru/chia), khong can du lieu ngoai FireAnt.

---

## D. TINH NANG FINBOOK CON LAI — CO THE HOC HOI

> Ket qua tu viec phan tich source code frontend cua finbook.vn (build manifest, JS chunks, route structure).

### D.1 DA CO TRONG FINFLOW (trung lap)

| Tinh nang Finbook | FinFlow tuong duong | Ghi chu |
|-------------------|---------------------|---------|
| 27 bieu do phan tich (non-bank + bank) | Phan B & C o tren | Da mapping 100% |
| Dinh gia PE/PB/PS | Chatbot valuation_engine (PE/PB + industry playbook) | FinFlow da co, co the mo rong |
| Danh muc dau tu | `investment.portfolio` module (BE + iOS) | Da co |
| Download BCTC (link CafeF) | Khong can — FinFlow crawl data tu FireAnt | Khong can |

### D.2 TINH NANG MOI CO THE HOC HOI

#### 1. BO LOC CO PHIEU (Stock Screener) — ⭐ HIGH
Finbook co 2 bo loc rieng biet:
- **Bo loc doanh nghiep** (non-bank): loc theo ~40 tieu chi
- **Bo loc ngan hang** (bank): loc theo ~25 tieu chi bank-specific

**Cac nhom tieu chi loc:**

| Nhom | Tieu chi | FireAnt co? |
|------|----------|-------------|
| **Dinh gia** | PE, PB, PS, Von hoa | ✅ |
| **Kha nang sinh loi** | ROE, ROA, Bien LNG, Bien LNR, NIM, YOEA, COF (bank) | ✅ |
| **Tang truong** | DT thuan YoY, LNST YoY (quy, 3 nam, 5 nam), Tang truong TOI (bank), Tang truong cho vay (bank), Tang truong VCSH, Tang truong Tong TS | ✅ |
| **Can doi ke toan** | No vay rong/VCSH, Khoan phai thu/Tong TS, Hang ton kho/Tong TS, TSCD/Tong TS, TS do dang/TSCD, No chiem dung/VCSH | ✅ |
| **Hieu qua hoat dong** | Vong quay HTK, Vong quay phai thu | ✅ |
| **Chat luong tai san (bank)** | No xau (3->5), No (2->5), Du phong RR/No xau, Du phong RR/No(2->5), Tong TS/VCSH, Quy mo tin dung | ✅ |
| **Co tuc** | Co tuc/Gia CP, Co tuc TB 3 nam, Co tuc TB 5 nam | ✅ (tinh duoc) |
| **Thong tin co ban** | Nganh, San giao dich | ✅ |

> **Kha thi voi FireAnt:** ~95%. Chi thieu CASA va cho vay theo ky han/nhom KH (da bo o tren).
> **De xuat:** Xay dung screener API tren backend, dung bulk API `all-financial-data` de lay data toan thi truong.

#### 2. DU LIEU NGANH (Sector Analytics) — ⭐ HIGH
Finbook co trang `/du-lieu-nganh` hien thi:
- Trung vi nganh cho cac chi so: ROE, Tong TS/VCSH, Ty le No, Tang truong cho vay...
- So sanh cong ty vs trung vi nganh

| Du lieu | FireAnt co? | Ghi chu |
|---------|-------------|---------|
| Sector PE, PB, PS median | ✅ `SectorPE`, `SectorPB`, `SectorPS` | Co san |
| Sector ROE, ROA | ✅ `SectorROE`, `SectorROA` | Co san |
| Sector GrossMargin, EBITMargin | ✅ `SectorGrossMargin`, `SectorEBITMargin` | Co san |
| Sector ratios (CurrentRatio, D/E...) | ✅ 50+ sector fields | Co san |
| ICB industry classification | ✅ `ICBCode`, `ICBName` | Co san |

> **Kha thi voi FireAnt:** 100%. FireAnt tra 50+ sector benchmarks san trong moi API call.

#### 3. DU LIEU THI TRUONG CHUNG (VNINDEX Analytics) — MEDIUM
Finbook co trang `/du-lieu-ttck`:
- Bieu do PE/PB lich su cua VNINDEX (tu 2008)
- So sanh PE hien tai vs trung vi lich su

| Du lieu | FireAnt co? | Ghi chu |
|---------|-------------|---------|
| VNINDEX PE/PB lich su | ❌ Khong co truc tiep | Da kiem tra tat ca endpoint: financial-data, historical-price, fundamental, profile, market/overview → tat ca 404 hoac error |

> **Ket qua kiem tra (2026-04-23):**
> - FireAnt **KHONG** co endpoint cho index-level data (VNINDEX, VN30...).
> - Da test: `/symbols/VNINDEX/financial-data`, `/symbols/VNINDEX/historical-price`, `/symbols/VNINDEX/fundamental`, `/symbols/VNINDEX/profile`, `/market/overview` → tat ca fail.
>
> **Giai phap: Tu tinh (self-compute) tu du lieu tung co phieu**
> - Cong thuc chuan market-cap-weighted:
>   - `VNINDEX PE = Σ(MarketCap) / Σ(PAT_TTM)` (cho tat ca co phieu tren HOSE)
>   - `VNINDEX PB = Σ(MarketCap) / Σ(Equity)` (cho tat ca co phieu tren HOSE)
> - Test thu voi 5 co phieu (VNM, VCB, HPG, FPT, MWG): **PE = 26.75, PB = 3.33** — hop ly.
> - Luu y: Mot so bank (VD: VCB) co `PAT_TTM = 0` trong FireAnt → can dung `ProfitAfterTax_TTM` hoac `ProfitAfterTax_CUM` thay the.
>
> **De xuat:** Xay dung scheduled job (monthly/quarterly):
> 1. Goi `/symbols/all-financial-data` lay data toan thi truong (⚠️ endpoint nay dang tra ve `null` — can dieu tra them)
> 2. Neu bulk API khong kha dung: crawl tung co phieu HOSE (~400 ma), tinh aggregate
> 3. Luu ket qua vao bang `market_index_metrics` (date, pe, pb, market_cap_total)
> 4. Hien thi bieu do PE/PB lich su cua VNINDEX, so sanh voi trung vi

#### 4. KHAU VI RUI RO (Risk Profile Questionnaire) — MEDIUM
Finbook co `/khau-vi-rui-ro`:
- Khao sat xac dinh khau vi rui ro nha dau tu
- Ket qua: E ngai rui ro / Phong thu / Trung tinh / Ua thich rui ro / Tan cong

> **Kha thi:** 100% — khong can data ngoai, chi can logic questionnaire.
> **De xuat:** Them vao iOS app Profile module. Ket qua dung de ca nhan hoa goi y dau tu trong chatbot.

#### 5. TAI CHINH CA NHAN (Personal Finance Dashboard) — MEDIUM
Finbook co `/ts-tai-chinh-ca-nhan` — **rat giong FinFlow da co!**
- Bang can doi ke toan ca nhan (tai san vs no)
- Bao cao KQKD ca nhan (thu nhap vs chi phi)
- ROA, ROE ca nhan
- Co cau tai san: thanh khoan, co dinh, dau tu
- Co cau no: ngan han, dai han, chiem dung

> **FinFlow da co:** Transaction module + Wealth module + Budget module.
> **Co the hoc hoi:** Cach Finbook **trinh bay** — bang can doi ke toan ca nhan, tinh ROA/ROE ca nhan.

#### 6. MUC TIEU TAI CHINH (Financial Goal Planner) — LOW
Finbook co `/ts-muc-tieu-tai-chinh`:
- Dat muc tieu tiet kiem
- Tinh toan so tien can gop moi thang

> **De xuat:** Nice-to-have, co the them vao Budget module sau.

#### 7. TINH LAI KEP (Compound Interest Calculator) — LOW
Finbook co `/ts-tinh-lai-kep`:
- May tinh lai kep
- Mo phong dau tu DCA (Dollar Cost Averaging)
- So sanh voi VNINDEX

> **De xuat:** Cong cu tien ich, co the them vao iOS app hoac chatbot.

#### 8. BANG GIA + QUAN TAM (Watchlist) — LOW
Finbook co `/ts-bang-gia`:
- Tao danh sach quan tam
- Gia can nhac ban, gia ngung lo
- Link BCTC
- Phan loai theo khau vi rui ro

> **FinFlow da co:** Portfolio module co watchlist tuong tu.
> **Co the hoc hoi:** Them truong "gia muc tieu", "gia ngung lo" vao portfolio_assets.

#### 9. XEP HANG DANH GIA TU DONG — CO SAN TU FIREANT
Finbook hien thi danh gia tu dong tren moi chart (VD: "CAO", "AN TOAN", "TRUNG TINH", "TICH CUC SO VOI NGANH").

| Danh gia | Logic | FireAnt co? |
|----------|-------|-------------|
| ROE/ROA: Cao/Trung tinh/Thap | So voi SectorROE/SectorROA | ✅ |
| Nguon von: An toan/Rui ro | Dua tren No vay rong/VCSH | ✅ |
| No xau: Tich cuc/Tieu cuc so voi nganh | So voi SectorNPLToLoan | ✅ |
| Tang truong: % kep 5 nam | Tinh tu du lieu lich su | ✅ |

> **Kha thi:** 100%. Tat ca danh gia deu la so sanh don gian giua chi so cong ty vs sector benchmark.
> FireAnt tra ca 2 (company value + sector value) trong 1 API call.

---

## E. TONG KET CUOI CUNG

### Nhung gi FinFlow da co:
- 27 bieu do phan tich (Section B & C)
- Dinh gia PE/PB/PS
- Danh muc dau tu
- Tai chinh ca nhan (thu chi, ngan sach, tai san)
- AI Chatbot phan tich

### Nhung gi co the hoc hoi tu Finbook:

| STT | Tinh nang | Muc do uu tien | Kha thi | Ghi chu |
|-----|-----------|----------------|---------|---------|
| 1 | **Bo loc co phieu** (2 loai: DN + NH) | ⭐ HIGH | 95% | Dung bulk API FireAnt |
| 2 | **Du lieu nganh** (sector benchmarks) | ⭐ HIGH | 100% | FireAnt co 50+ sector fields |
| 3 | **Xep hang tu dong** tren chart | ⭐ HIGH | 100% | So sanh company vs sector |
| 4 | **VNINDEX PE/PB lich su** | MEDIUM | ✅ Tu tinh duoc | FireAnt khong co truc tiep, nhung tinh duoc tu du lieu tung co phieu (market-cap-weighted) |
| 5 | **Khau vi rui ro** questionnaire | MEDIUM | 100% | Khong can data ngoai |
| 6 | **BCTC ca nhan** (can doi KT ca nhan) | MEDIUM | 100% | Mo rong tren Wealth + Transaction |
| 7 | Muc tieu tai chinh | LOW | 100% | Mo rong Budget module |
| 8 | May tinh lai kep / DCA | LOW | 100% | Cong cu tien ich |
| 9 | Gia muc tieu / ngung lo trong watchlist | LOW | 100% | Them field vao portfolio_assets |
