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
  "categories": [
    {"id": "cat_food", "name": "Ăn uống", "type": "EXPENSE"}
  ],
  "accounts": [
    {"id": "acc_wallet", "name": "Ví Momo", "transactionEligible": true}
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
