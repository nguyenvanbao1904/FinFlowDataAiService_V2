# FinFlow Data & AI Service

Python microservice chạy song song với Spring Boot Backend, cung cấp AI analysis và data crawling cho hệ sinh thái FinFlow.

## Responsibilities

1. **AI Chat Agent** — ReAct agent loop (DeepSeek LLM) với 19 tools: truy vấn dữ liệu thị trường, RAG annual reports, định giá cổ phiếu, quản lý tài chính cá nhân.
2. **Transaction Prefill** — Trích xuất giao dịch từ văn bản tự nhiên (OCR/speech) bằng LLM structured output.
3. **Analytics Insights** — Sinh insights tài chính cá nhân (cảnh báo + mẹo) từ dữ liệu chi tiêu.
4. **Data Crawling** — Thu thập báo cáo tài chính, giá cổ phiếu, chỉ số thị trường từ vnstock & FireAnt.
5. **ML Forecast** — Dự báo doanh thu/lợi nhuận doanh nghiệp bằng 2 ML pipelines: eval (đánh giá model) + production (chatbot dùng), mỗi pipeline 4 XGBoost models.
6. **Data Sync** — Đẩy dữ liệu đã xử lý về Java Backend qua internal HTTP APIs.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI, Pydantic v2, pydantic-settings |
| LLM | DeepSeek (OpenAI-compatible API, native tool calling) |
| Vector DB | Qdrant (semantic search) + SQLite (BM25 keyword search) |
| HTTP Client | httpx.AsyncClient (shared singleton) |
| Market Data | vnstock, FireAnt REST v2 |
| ML | joblib models, on-demand subprocess forecast |
| Embeddings | MLX BGE-M3 (Apple Silicon local) |

## Architecture

```
app/
├── main.py                         # FastAPI entry, lifespan, CORS, exception handlers
├── core/
│   ├── config.py                   # pydantic-settings (reads .env)
│   ├── dependencies.py             # Lazy singleton factories, API key guard
│   └── http_client.py              # Shared httpx.AsyncClient singleton
├── models/
│   ├── analytics.py                # Analytics insights DTOs
│   ├── chat.py                     # Chat orchestration DTOs
│   ├── investment.py               # Investment/market data DTOs
│   └── transaction.py              # Transaction prefill DTOs
├── routers/
│   ├── ai.py                       # /transaction-prefill, /analytics-insights
│   └── chat.py                     # /chat/orchestrate, /chat/thread-summary
├── services/
│   ├── analytics_service.py        # Analytics insights generation
│   ├── prefill_service.py          # Transaction prefill via LLM
│   ├── forecast_service.py         # ML forecast per request (no persistent cache)
│   └── chat/
│       ├── orchestrator.py         # ReAct agent loop (max 6 iterations)
│       ├── tool_registry.py        # 19 tool definitions (single source of truth)
│       ├── valuation_engine.py     # Deterministic fair-value (PE/PB + industry playbook)
│       ├── tracing.py              # Request tracing
│       ├── prompts/
│       │   └── agent_prompt.py     # Agent system prompt
│       └── utils/
│           ├── math_helpers.py
│           └── vietnamese_text.py
├── infrastructure/
│   ├── llm_client.py               # DeepSeek client: call_structured, call_json, call_with_tools
│   ├── market_data_client.py       # Routes tool calls to Java backend APIs
│   ├── rag_client.py               # Hybrid vector+keyword RAG (Qdrant + SQLite)
│   ├── backend_client.py           # Java backend HTTP sync client
│   ├── ports/
│   │   └── backend_sync_port.py    # Abstract sync contract
│   └── crawler/
│       ├── vnstock_adapter.py      # vnstock data crawling with alias mapping
│       ├── fireant_adapter.py      # FireAnt company profile + ICB data
│       ├── icb_normalization.py    # ICB code normalization
│       └── icb_sync.py             # ICB industry tree sync
└── jobs/
    └── batch_crawler.py            # Multi-process market data crawler

scripts/financial_training/         # Offline pipelines (model training, RAG indexing, embeddings)
```

30 Python source files (excluding `__init__.py`).

## API Endpoints

All endpoints (except `/` and `/health`) require header `X-Internal-Api-Key`.
In production, the service fails closed when `INTERNAL_API_KEY` is missing; only `ENVIRONMENT=local|development|test` may bypass this for local development.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/ai/transaction-prefill` | Trích xuất giao dịch từ text |
| `POST` | `/api/v1/ai/analytics-insights` | Sinh insights tài chính cá nhân |
| `POST` | `/api/v1/ai/chat/orchestrate` | ReAct agent chat |
| `POST` | `/api/v1/ai/chat/thread-summary` | Tóm tắt context thread |

### Transaction Prefill

Nhận text từ OCR/speech, gọi DeepSeek LLM (structured output + Pydantic validation + retry loop), trả về giao dịch đã trích xuất.

**Request:**

```json
{
  "rawText": "ăn trưa bún bò 65k bằng ví momo",
  "categories": [{ "id": "cat_food", "name": "Ăn uống", "type": "EXPENSE" }],
  "accounts": [{ "id": "acc_wallet", "name": "Ví Momo", "transactionEligible": true }],
  "recentHistory": [],
  "locale": "vi-VN",
  "timezone": "Asia/Ho_Chi_Minh",
  "source": "text"
}
```

**Response:**

```json
{
  "amount": 65000,
  "type": "EXPENSE",
  "categoryId": "cat_food",
  "accountId": "acc_wallet",
  "note": "ăn trưa bún bò",
  "transactionDate": "2026-04-22T10:00:00+07:00",
  "confidence": 0.89,
  "missingFields": [],
  "warnings": []
}
```

### Analytics Insights

Sinh 2 insights (WARNING + TIP) từ dữ liệu chi tiêu. Hỗ trợ 2 tier: `FULL` (có trend data) và `SPARSE` (ít dữ liệu).

### Chat Orchestrate

ReAct agent loop: LLM gọi tools → nhận kết quả → quyết định gọi thêm hoặc trả lời. Tối đa 6 iterations.

**19 tools available:**

| Category | Tools |
|----------|-------|
| Market Data (12) | `get_company_financial_series`, `get_company_metrics`, `get_company_daily_valuations`, `get_company_live_valuation_snapshot`, `get_company_forecast`, `get_company_dividends`, `get_company_valuations`, `get_company_analysis`, `get_company_market_data`, `get_industry_nodes`, `suggest_companies`, `get_company_industries` |
| RAG (1) | `search_annual_reports` — Hybrid vector+keyword search trên annual reports |
| Valuation (1) | `compute_fair_value` — Deterministic PE/PB fair-value với industry playbook |
| Personal Finance (5) | `get_personal_finance_report`, `get_user_transaction_context`, `add_transaction`, `get_user_budgets`, `add_budget` |

### Thread Summary

Tóm tắt context cuộc hội thoại để duy trì continuity giữa các lượt chat.

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `INTERNAL_API_KEY` | Shared key với Spring Boot backend |
| `ENVIRONMENT` | `production` by default; set `local` for local dev without internal auth |

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model ID |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `LLM_TIMEOUT_SECONDS` | `60` | HTTP timeout cho LLM calls |
| `JAVA_BACKEND_URL` | `http://localhost:8080/api/internal` | Spring Boot internal API |

### Embeddings (offline scripts)

| Variable | Default | Description |
|----------|---------|-------------|
| `VOYAGE_API_KEY` | | Voyage API key for embedding/rerank |
| `VOYAGE_EMBED_BASE_URL` | `https://api.voyageai.com/v1` | Voyage embedding API URL |
| `VOYAGE_EMBED_MODEL` | `voyage-3.5-lite` | Embedding model name |

### Prefill Behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFILL_ENABLE_CATEGORY_HEURISTIC` | `false` | Keyword-based category correction sau LLM |
| `PREFILL_ENFORCE_TYPE_CATEGORY_CONSISTENCY` | `true` | Auto-align tx type với category type |

### Chat & RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_RAG_ENABLED` | `true` | Bật/tắt RAG retrieval |
| `CHAT_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant vector DB URL |
| `CHAT_QDRANT_API_KEY` | | Qdrant API key (nếu có auth) |
| `CHAT_QDRANT_COLLECTION` | `annual_report_chunks_voyage_3_5_lite` | Qdrant collection name |
| `CHAT_RAG_CHUNKS_DB` | `artifacts/.../chunks.sqlite` | SQLite chunks DB path |
| `CHAT_RAG_TOPK_VECTOR` | `50` | Top-K kết quả vector search |
| `CHAT_RAG_TOPK_KEYWORD` | `50` | Top-K kết quả keyword search |
| `CHAT_RAG_TOPK_FINAL` | `5` | Top-K kết quả cuối cùng (sau rerank) |
| `CHAT_TOOL_TIMEOUT_SECONDS` | `30` | Timeout cho tool calls |
| `CHAT_FORECAST_ENABLED` | `true` | Bật/tắt ML forecast tool |
| `CHAT_FORECAST_SUMMARY_JSON` | `artifacts/.../production_pipeline/summary.json` | Forecast summary metadata |
| `CHAT_FORECAST_SCRIPT` | `scripts/.../test_final_models_forecast.py` | Path tới forecast script |
| `CHAT_FORECAST_TIMEOUT_SECONDS` | `180` | Timeout cho mỗi request forecast |
| `CHAT_FORECAST_TOP_FACTORS` | `5` | Số top factors hiển thị trong forecast |
| `CHAT_DEBUG_LOG_PROMPTS` | `false` | Log prompts/responses (dev only) |
| `CHAT_DEBUG_LOG_MAX_CHARS` | `8000` | Max chars khi log prompt/response |

### Market Data Crawling

| Variable | Default | Description |
|----------|---------|-------------|
| `VNSTOCK_API_KEY` | | vnstock API key |
| `FIREANT_ACCESS_TOKEN` | | FireAnt OAuth2 Bearer token (scope: `symbols-read`) |
| `FIREANT_API_BASE` | `https://restv2.fireant.vn` | FireAnt REST API base |

### Database (shared với Spring Boot)

| Variable | Description |
|----------|-------------|
| `FINFLOW_DATABASE_URL` | Full MySQL connection URL (alternative) |
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | MySQL connection (individual fields) |

### Pricing

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_DEEPSEEK_INPUT_PRICE_PER_1M` | `0.10` | USD per 1M input tokens |
| `CHAT_DEEPSEEK_OUTPUT_PRICE_PER_1M` | `0.40` | USD per 1M output tokens |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Dev server
uvicorn app.main:app --reload --port 8001

# Batch crawler
python -m app.jobs.batch_crawler
```

## Offline Pipelines (scripts/)

### ML Forecast Pipelines

Hệ thống dùng **2 pipeline** riêng biệt, mỗi pipeline gồm 4 XGBoost models (bank_revenue, bank_profit, nonbank_revenue, nonbank_profit):

| Pipeline | Train data | Predict | Mục đích |
|----------|-----------|---------|----------|
| **`eval_pipeline`** | → 2024 | 2025 | Đánh giá chất lượng model (so sánh dự báo 2025 vs số liệu thực 2025) |
| **`production_pipeline`** | → 2025 | 2026 | Chatbot dùng 4 model `.joblib` để dự báo theo request |

**Chatbot chỉ đọc `production_pipeline`** — config trong `app/core/config.py` (các biến `CHAT_FORECAST_*`) trỏ vào thư mục này.

#### Bước 1: Build macro data

Trước khi train model, cần build file CSV chứa dữ liệu vĩ mô hàng năm (GDP growth, CPI, tỷ giá, hàng hóa, VNINDEX).

```bash
# Tự động fetch từ FireAnt API + yfinance (cần FIREANT_ACCESS_TOKEN trong .env)
python scripts/financial_training/build_macro_data.py

# Nếu không có API token, dùng fallback data có sẵn trong code
python scripts/financial_training/build_macro_data.py --skip-fireant --skip-yfinance

# Bao gồm năm hiện tại (dữ liệu chưa đầy đủ)
python scripts/financial_training/build_macro_data.py --include-current-year
```

Output: `artifacts/macro/macro_yearly_train.csv` (nonbank, 18 cols) và `macro_yearly_train_bank.csv` (bank, 17 cols).

Dữ liệu nguồn:

| Nguồn | Features |
|-------|----------|
| FireAnt API | `gdp_growth_yoy_pct` (GDP Annual Growth Rate), `cpi_inflation_yoy_pp`, VNINDEX daily quotes |
| yfinance | `usd_vnd_yoy_pct`, `gold_gc_log`, `oil_brent_log`, `nat_gas_log`, `sugar_log`, `coffee_log`, `rice_log`, `bdry_shipping_etf_log` |
| Manual in-code | `interest_deposit_12m_pct`, `interest_loan_short_pct`, `interest_loan_midlong_pct`, `hrc_log`, `iron_ore_log`, `coal_log`, `rubber_log` |

GDP và USD/VND dùng **tốc độ tăng trưởng YoY %** (stationary) thay vì log giá trị tuyệt đối — giúp model extrapolate tốt hơn khi dự báo năm mới.

Flags: `--start-year 2013`, `--end-year` (mặc định: năm trước), `--out-dir`, `--skip-fireant`, `--skip-yfinance`, `--include-current-year`.

#### Bước 2: Train pipeline

```bash
# Walk-forward eval — một lệnh đánh giá ALL/VN100/VN30 qua nhiều fold
python scripts/financial_training/run_model_walk_forward_eval.py \
  --source db \
  --fold-years 2022,2023,2024,2025 \
  --out-dir artifacts/models/walk_forward_eval

# Production pipeline — chatbot dùng
python scripts/financial_training/run_final_model_pipeline.py \
  --source db \
  --train-target-year-max 2025 \
  --predict-target-year 2026 \
  --out-dir artifacts/models/production_pipeline
```

Output walk-forward eval: `fold_metrics.csv`, `aggregate_metrics.csv`, `summary.json`, `report.md`, kèm từng fold ở `fold_<year>/`.
Output production pipeline cần cho runtime: 4 `.joblib` models + `summary.json`.

Flags thường dùng:

| Flag | Default | Mô tả |
|------|---------|-------|
| `--source db\|csv` | `db` | Đọc data từ MySQL hoặc CSV preprocessed |
| `--symbols ACB,VEA,...` | `ACB,VEA,NLG,DGC,PNJ,MWG,VIB,VPB` | Symbols cho báo cáo so sánh |
| `--nonbank-feature-budget` | `50` | Số features tối đa cho nonbank (RFE selection) |
| `--steel-boost` | `1.0` | Hệ số tăng cường tương tác ngành thép |
| `--recency-weight-mode` | `exp` | Trọng số theo thời gian: `none`, `linear`, `exp` |
| `--enable-robust-clip` | `true` | Clip outlier ratio/percent columns theo quantile train |
| `--enable-debt-interest-adjustment` | `true` | Tăng lãi suất cho DN đòn bẩy cao |
| `--profit-huber-slope` | `0` | Huber loss cho profit (0=tắt, dùng MAE). Thử nghiệm cho thấy tắt tốt hơn |
| `--profit-winsorize-lower-q` | `0` | Winsorize profit target (0=tắt). Thử nghiệm cho thấy tắt tốt hơn |

> **Ghi chú thử nghiệm:** Huber loss + Winsorize đã được A/B test và cho kết quả kém hơn (R² profit rớt từ 0.920→0.884 VN30). Nguyên nhân: lợi nhuận bluechip biến động mạnh nhưng đó là tính chu kỳ thật, không phải nhiễu — nén lại sẽ phá hủy tín hiệu. Giữ code để thử nghiệm nhưng default = tắt.

Walk-forward eval dùng expanding window:

| Fold | Train | Test |
|------|-------|------|
| 2022 | <= 2021 | 2022 |
| 2023 | <= 2022 | 2023 |
| 2024 | <= 2023 | 2024 |
| 2025 | <= 2024 | 2025 |

Script tự tính metric cho 3 scope: `ALL`, `VN100`, `VN30` từ `predict_detail.csv` của từng fold. Có thể thêm `--reuse-existing` để chỉ aggregate lại các fold đã chạy.

#### Bước 3: Runtime forecast (1 mã, nhiều năm)

```bash
python scripts/financial_training/test_final_models_forecast.py \
  --symbol ACB --to-year 2030 \
  --model-dir artifacts/models/production_pipeline \
  --source db
```

Chatbot gọi script này khi user hỏi forecast. `forecast_service.py` ghi output vào thư mục tạm, đọc kết quả, rồi xóa ngay; runtime không cache file trong `artifacts/`.

Flags: `--base-year 2025`, `--predict-target revenue|profit|both`, `--history-mode recursive|use-actual-when-available`, `--top-features 8`.

#### Artifacts structure

```
artifacts/
├── macro/
│   ├── macro_yearly_train.csv          # Nonbank macro (18 cols, 2013-2025)
│   └── macro_yearly_train_bank.csv     # Bank macro (17 cols, 2013-2025)
└── models/
    ├── walk_forward_eval/              # Đánh giá expanding-window ALL/VN100/VN30
    │   ├── fold_2022/
    │   ├── fold_2023/
    │   ├── fold_2024/
    │   ├── fold_2025/
    │   ├── fold_metrics.csv
    │   ├── aggregate_metrics.csv
    │   ├── summary.json
    │   └── report.md
    └── production_pipeline/            # Chatbot dùng
        ├── bank_revenue_next.joblib
        ├── bank_profit_after_tax_next.joblib
        ├── nonbank_revenue_next.joblib
        ├── nonbank_profit_after_tax_next.joblib
        └── summary.json
```

### Annual Report RAG Chunking

```bash
python scripts/financial_training/chunk_annual_reports.py
```

Input: PDFs trong `artifacts/rag/annual_reports/raw_pdfs/`.
Output: `artifacts/rag/annual_reports/chunks/annual_reports_chunks.json`.

Flags: `--min-chars 300`, `--max-chars 2400`, `--ocr-image-only`, `--ocr-backend auto|vision|none`, `--include-other`.

### Full RAG Pipeline (Crawl -> Chunk -> Index)

```bash
# Pull all symbols from DB, run full pipeline from VietstockFinance
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --report-years 5

# Storage-safe streaming mode
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --report-years 5 --streaming --delete-pdf-after-chunk

# With SQLite checkpoint + local LLM repair
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --report-years 5 --streaming --delete-pdf-after-chunk \
  --checkpoint-backend sqlite \
  --output-chunks-db artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite \
  --llm-repair-garbled-chunks

# Symbol-shard workers (parallel)
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --worker-mode symbol-shard-partition --shard-count 8 \
  --report-years 5 --streaming --delete-pdf-after-chunk \
  --checkpoint-backend sqlite --reset-output \
  --output-chunks-db artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite
```

Key flags: `--skip-crawl`, `--parser-backend kreuzberg|pymupdf`, `--exchange-filter HOSE,HNX,UPCOM`, `--limit-symbols N`, `--rebuild-fts-only`, `--ocr-fix-garbled`, `--llm-repair-thinking`.

Outputs: raw PDFs, crawl manifest, chunks SQLite/JSON with FTS5 BM25 keyword index, per-worker shard DBs.

### Embeddings (Voyage + Qdrant)

```bash
VOYAGE_EMBED_BATCH_SIZE=128 \
bash scripts/financial_training/build_voyage_embedding_qdrant.sh --qdrant-recreate-collection
```

Flags: `--rebuild-model`, `--limit 1000`, `--qdrant-recreate-collection`, `--qdrant-collection`.

## Caching Note

Response caching cho `/api/v1/ai/analytics-insights` được implement ở Spring Boot backend (Redis), không phải ở service này.

## Runtime Artifacts

`crawler_state.json`, `failed_report.json` và thư mục `artifacts/` đều nằm trong `.gitignore`.
