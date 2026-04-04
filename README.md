# FinFlow Data & AI Service

This is a standalone Python microservice that runs alongside the Java Spring Boot Backend and SwiftUI iOS App.

## Responsibilities

1. **Data Crawling**: Fetching financial statements, price histories, and market indicators from external sources.
2. **AI & Analysis**: Running predictions, calculating technical indicators, and providing intelligent portfolio insights using LangChain / OpenAI.
3. **Data Sync**: Pushing processed data to the Java Backend via internal HTTP APIs (REST).

## Architecture (Cleanup for AI phase)

Current structure keeps runtime stable while aligning to Clean Architecture boundaries:

- `app/domain/entities/`: domain DTO/entities used by use cases and crawlers.
- `app/domain/ports/`: abstraction ports (e.g. backend sync contract).
- `app/services/`: crawler-focused application logic (to be split into use cases gradually).
- `app/clients/`: outbound adapters (HTTP client to Java backend).
- `app/core/`: config and framework wiring.

Notes:

- `app/models/investment.py` is kept as a backward-compatible import path and now re-exports from `app/domain/entities/investment.py`.
- Runtime artifacts (`crawler_state.json`, `failed_report.json`) are intentionally ignored via `.gitignore`.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn app.main:app --reload --port 8001
```

## Running Batch Crawler

```bash
python -m app.jobs.batch_crawler
```

## Final Model Pipeline (Single Command)

Single script to run the finalized model flow end-to-end for the selected setup:

- signed-log-growth target transform
- train up to target year 2025
- predict target year 2026
- auto export 2 report tables (Revenue and Profit) for selected symbols

Run:

```bash
python scripts/financial_training/run_final_model_pipeline.py
```

Default report symbols:

- `ACB, VEA, NLG, DGC, PNJ, MWG, VIB, VPB`

Main outputs:

- `artifacts/models/final_model_pipeline/bank_revenue_next.joblib`
- `artifacts/models/final_model_pipeline/bank_profit_after_tax_next.joblib`
- `artifacts/models/final_model_pipeline/nonbank_revenue_next.joblib`
- `artifacts/models/final_model_pipeline/nonbank_profit_after_tax_next.joblib`
- `artifacts/models/final_model_pipeline/report_table.csv`
- `artifacts/models/final_model_pipeline/report_style.md`
- `artifacts/models/final_model_pipeline/summary.json`

Useful flags:

- `--train-target-year-max 2025`: train cutoff by target year
- `--predict-target-year 2026`: prediction target year
- `--symbols ACB,VEA,NLG,DGC,PNJ,MWG,VIB,VPB`: symbols included in report
- `--nonbank-feature-budget 50`: max selected non-bank features via RFE
- `--steel-boost 3.0`: weight for steel spread interaction
- `--out-dir artifacts/models/final_model_pipeline`: output directory

## Test Forecast Script (One Symbol, Multi-Year)

Use this script after the final pipeline has generated 4 joblib models. It forecasts one symbol recursively from base year (default 2025) to your target year.

Run:

```bash
python scripts/financial_training/test_final_models_forecast.py --symbol ACB --to-year 2030
```

Useful flags:

- `--base-year 2025`: starting year for recursive forecast
- `--to-year 2030`: forecast output until this year
- `--model-dir artifacts/models/final_model_pipeline`: folder containing 4 model files
- `--bank-revenue-model ...`, `--bank-profit-model ...`, `--nonbank-revenue-model ...`, `--nonbank-profit-model ...`: manually choose each model path
- `--out-csv artifacts/models/final_model_pipeline/forecast_ACB_2025_2030.csv`: custom output path

## Annual Report RAG Chunking

Chunk annual report PDFs into JSON chunks with metadata (`stock_code`, `year`, `category`, `section`) for RAG.

1. Put PDF files into:

- `artifacts/rag/annual_reports/raw_pdfs`

2. Run chunking script:

```bash
python scripts/financial_training/chunk_annual_reports.py
```

3. Output JSON:

- `artifacts/rag/annual_reports/chunks/annual_reports_chunks.json`

Optional flags:

- `--pdf-files <path1> <path2>`: process explicit PDF files
- `--min-chars 300`: minimum chunk size
- `--max-chars 2400`: split very long text into child chunks
- `--ocr-image-only`: OCR pages without text layer (for scanned PDFs)
- `--ocr-language vie+eng`: OCR language pack
- `--ocr-dpi 200`: OCR render DPI

## AI Transaction Prefill API

- Endpoint: `POST /api/v1/ai/transaction-prefill`
- Purpose: receive OCR/manual text from FE, call Gemini (JSON mode), return structured prefill JSON. This endpoint does not save data to backend.
- Security: when `INTERNAL_API_KEY` is configured, caller must provide header `X-Internal-Api-Key`.

### Required env

- `GEMINI_API_KEY`
- `INTERNAL_API_KEY` (must match backend `INTERNAL_API_KEY` for `/api/internal/**`)
- Optional:
  - `GEMINI_MODEL` (default: `gemini-1.5-flash`)
  - `GEMINI_TIMEOUT_SECONDS` (default: `20`)

### Request example

```json
{
  "rawText": "ăn trưa bún bò 65000 bằng ví momo",
  "categories": [{ "id": "cat_food", "name": "Ăn uống", "type": "EXPENSE" }],
  "accounts": [
    { "id": "acc_wallet", "name": "Ví Momo", "transactionEligible": true }
  ],
  "recentHistory": [],
  "locale": "vi-VN",
  "timezone": "Asia/Ho_Chi_Minh",
  "source": "text"
}
```

### Response example

```json
{
  "amount": 65000,
  "type": "EXPENSE",
  "categoryId": "cat_food",
  "accountId": "acc_wallet",
  "note": "ăn trưa bún bò",
  "transactionDate": "2026-03-31T10:00:00+07:00",
  "confidence": 0.89,
  "missingFields": [],
  "warnings": []
}
```

## Analytics insights (caching)

Response caching for `POST /api/v1/ai/analytics-insights` is implemented in the **Spring Boot backend** (Redis), not in this service. Only the backend calls this endpoint.
