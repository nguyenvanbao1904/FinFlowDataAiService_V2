# FinFlow Data & AI Service

This is a standalone Python microservice that runs alongside the Java Spring Boot Backend and SwiftUI iOS App.

## Responsibilities

1. **Data Crawling**: Fetching financial statements, price histories, and market indicators from external sources.
2. **AI & Analysis**: Running predictions, calculating technical indicators, and providing intelligent portfolio insights using LangChain / OpenAI.
3. **Data Sync**: Pushing processed data to the Java Backend via internal HTTP APIs (REST).

## Architecture

```
app/
├── main.py                    # FastAPI entry + routes
├── core/config.py             # pydantic-settings configuration
├── domain/
│   ├── entities/              # Request/response DTOs (Pydantic models)
│   └── ports/                 # Abstract contracts (e.g. BackendSyncPort)
├── services/
│   ├── chat/                  # ReAct agent: orchestrator, LLM client, tools, RAG, valuation
│   ├── transaction_prefill_service.py
│   ├── analytics_insights_service.py
│   ├── chat_orchestrator_service.py
│   ├── forecast_tool_service.py
│   ├── rag_retrieval_service.py
│   ├── market_data_tool_client.py
│   ├── crawler_service.py     # vnstock + FireAnt data crawling
│   ├── fireant_profile.py
│   ├── icb_normalization.py
│   ├── icb_tree_sync.py
│   └── vision_ocr.py         # Apple Vision OCR (macOS ARM64)
├── clients/
│   └── java_backend_client.py # HTTP sync to Java backend
└── jobs/
    └── batch_crawler.py       # Multi-process market data crawler
scripts/
└── financial_training/        # Offline pipelines: model training, RAG indexing, embedding
```

Runtime artifacts (`crawler_state.json`, `failed_report.json`) are gitignored.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Apple Vision OCR backend is enabled automatically on macOS ARM64 when `--ocr-image-only` is used (via `--ocr-backend auto`).

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
Output also includes value-investing metadata: `category_priority_score`, `category_priority_label`, `value_importance_stars`, `chapter_hint`, `taxonomy_version`.

By default, the script keeps only value-investing focused categories:

- `mdna`, `strategy`, `risk`, `governance`, `sustainability`, `business_overview`

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
- `--ocr-backend auto|vision|none`: OCR backend (`auto` => `vision` on macOS ARM64)
- `--ocr-language vie+eng`: Vision OCR language tokens (`vie+eng` => `vi-VN,en-US`)
- `--ocr-dpi 200`: render DPI used for Vision OCR
- `--include-other`: include non-focus chunks (`other`) when needed

## Annual Report RAG Full Pipeline (Crawl -> Chunk -> Index)

Run one command to:

- Crawl/download PDFs from internet sources into `artifacts/rag/annual_reports/raw_pdfs`
- Chunk PDFs into `artifacts/rag/annual_reports/chunks/annual_reports_chunks.json`
- Build BM25 retrieval index into `artifacts/rag/annual_reports/index/annual_reports_bm25_index.json`

Command examples:

```bash
# Pull all symbols from DB table companies and run full pipeline (single entrypoint)
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --cafef-years 5

# Storage-safe mode for large runs: download -> chunk -> delete each PDF
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --cafef-years 5 \
  --streaming \
  --delete-pdf-after-chunk

# Concurrent-safe checkpoint (SQLite, default) + local LLM repair (non-thinking by default)
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --cafef-years 5 \
  --exchange-filter HOSE,HNX,UPCOM \
  --streaming \
  --delete-pdf-after-chunk \
  --checkpoint-backend sqlite \
  --output-chunks-db artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite \
  --llm-repair-garbled-chunks \
  --llm-repair-model mlx-community/gemma-4-e2b-it-4bit

# Symbol-shard workers (8 workers, each worker writes its own SQLite then auto-merge)
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --worker-mode symbol-shard-partition \
  --shard-count 8 \
  --cafef-years 5 \
  --streaming \
  --delete-pdf-after-chunk \
  --checkpoint-backend sqlite \
  --reset-output \
  --output-chunks-db artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite

# Optional: test subset first
python scripts/financial_training/run_annual_report_rag_pipeline_from_db.py \
  --cafef-years 5 \
  --limit-symbols 30 \
  --skip-index
```

Useful flags:

- `--skip-crawl`: do not fetch internet links, process existing PDFs only
- `--parser-backend kreuzberg|pymupdf`: chunk extraction backend (`kreuzberg` default, fallback-safe)
- `--ocr-backend auto|vision|none`: OCR backend (`auto` picks Vision on macOS ARM64)
- `--cafef-years 5`: number of latest annual reports (BCTN) per symbol
- `run_annual_report_rag_pipeline_from_db.py`: fetch all symbols from MySQL table `companies` (`id` + `exchange`) and crawl BCTN automatically
- `--limit-symbols N`: limit first N symbols for dry-run validation
- `--exchange-filter HOSE,HNX,UPCOM`: optional exchange filter in DB mode
- `--streaming`: process each downloaded PDF immediately during crawl (recommended at scale)
- `--delete-pdf-after-chunk`: remove each downloaded PDF after successful chunking in streaming mode
- `--checkpoint-backend sqlite|json`: chunk checkpoint storage backend (`sqlite` default, concurrent-safe)
- `--output-chunks-db ...`: SQLite checkpoint path for chunk persistence/resume
- `--reset-output`: clear existing checkpoint/output before a new streaming run
- `--skip-final-export`: keep chunks only in checkpoint backend, skip final JSON export
- `--worker-mode single|exchange-partition|symbol-shard-partition`: execution mode
- `--shard-count 8`: number of symbol shards/workers (for symbol-shard-partition)
- `--shard-index 0`: run one specific shard in single mode (advanced/manual run)
- `--workers-dir ...`: worker artifact root directory
- `--skip-index`: stop after chunking
- `--crawl-timeout 30`: timeout per HTTP request
- `--min-chars`, `--max-chars`, `--overlap-chars`: chunking controls
- `--ocr-image-only`: OCR pages without text layer
- `--ocr-backend`: OCR backend selector (Vision recommended on Apple Silicon)
- `--ocr-force-all-pages`: force OCR on all pages (slower, helps with legacy font corruption)
- `--ocr-fix-garbled`: detect garbled text pages and OCR-repair them
- `--ocr-garbled-page-ratio-threshold`: OCR all text pages when garbled-page ratio is high
- `--keep-garbled-chunks`: keep low-quality garbled chunks (default is to drop them)
- `--llm-repair-garbled-chunks`: call local Gemma/OpenAI-compatible API to rewrite garbled chunks
- `--llm-repair-thinking`: opt-in thinking mode for local Gemma repair (default is non-thinking for speed)
- `--include-other`: keep non-focus categories

Outputs:

- Raw PDFs: `artifacts/rag/annual_reports/raw_pdfs`
- Crawl manifest: `artifacts/rag/annual_reports/raw_pdfs/crawl_manifest.json`
- Chunks checkpoint DB: `artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite`
- Chunks JSON: `artifacts/rag/annual_reports/chunks/annual_reports_chunks.json`
- BM25 index: `artifacts/rag/annual_reports/index/annual_reports_bm25_index.json`
- Per-worker shard DBs: `artifacts/rag/annual_reports/workers/shard_XX_of_YY/chunks.sqlite` (auto-merged into final DB)

## Annual Report Embeddings (MLX BGE-M3)

After chunk DB is ready, build semantic embeddings from SQLite chunks into a dedicated embeddings DB.
You can upsert to Qdrant in the same run (embed + vector DB in one command).

Default model is set to:

- `mlx-community/bge-m3-mlx-fp16`

Run:

```bash
python scripts/financial_training/embed_annual_reports_chunks_mlx.py \
  --chunks-db artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite \
  --output-db artifacts/rag/annual_reports/embeddings/annual_reports_embeddings.sqlite \
  --embed-model mlx-community/bge-m3-mlx-fp16 \
  --batch-size 16
```

Embed and upsert to Qdrant in one command:

```bash
python scripts/financial_training/embed_annual_reports_chunks_mlx.py \
  --chunks-db artifacts/rag/annual_reports/chunks/annual_reports_chunks.sqlite \
  --output-db artifacts/rag/annual_reports/embeddings/annual_reports_embeddings.sqlite \
  --embed-model mlx-community/bge-m3-mlx-fp16 \
  --batch-size 16 \
  --qdrant-upsert \
  --qdrant-url http://127.0.0.1:6333 \
  --qdrant-collection annual_report_chunks_bge_m3
```

Embed directly in local MLX runtime (no `/v1/embeddings` server needed):

```bash
python scripts/financial_training/embed_annual_reports_chunks_mlx.py \
  --embed-mode mlx-local \
  --embed-model mlx-community/bge-m3-mlx-fp16 \
  --batch-size 16 \
  --qdrant-upsert \
  --qdrant-url http://127.0.0.1:6333 \
  --qdrant-collection annual_report_chunks_bge_m3
```

Useful flags:

- `--embed-base-url http://127.0.0.1:9090/v1`: local OpenAI-compatible endpoint
- `--rebuild-model`: clear existing rows of selected model and rebuild
- `--limit 1000`: dry-run with first pending 1000 chunks
- `--sleep-ms 50`: reduce thermal pressure during long runs
- `--qdrant-upsert`: enable upsert to Qdrant while embedding
- `--qdrant-recreate-collection`: recreate Qdrant collection before upsert
- `--qdrant-distance cosine|dot|euclid|manhattan`: vector distance metric
- `--embed-mode http|mlx-local`: choose API endpoint mode or direct local MLX mode

Runtime summaries in terminal:

- `[CRAWL][CODES]`: known stock codes before/after crawl, new codes, files per code
- `[RAW][CODES]`: current inventory of crawled PDFs grouped by inferred stock code
- `[CAFEF][SUMMARY]`: symbols requested, symbols resolved, and number of BCTN links found

Notes for Cafef mode:

- URL template is supported via symbol input: `https://cafef.vn/du-lieu/hsx/{symbol}-bao-cao-tai-chinh.chn`
- The script reads Cafef financial-report feed and keeps only rows named `Báo cáo thường niên`.
- Rows like `Bản điều lệ` / `Bản cáo bạch` are excluded.
- For each symbol, only the latest `--cafef-years` unique years are downloaded.
- OCR runs through Apple Vision when `--ocr-backend auto|vision` is active on macOS ARM64.

## AI Transaction Prefill API

- Endpoint: `POST /api/v1/ai/transaction-prefill`
- Purpose: receive OCR/manual text from FE, call local/OpenAI-compatible LLM (JSON mode), return structured prefill JSON. This endpoint does not save data to backend.
- Security: when `INTERNAL_API_KEY` is configured, caller must provide header `X-Internal-Api-Key`.

### Required env

- `LOCAL_LLM_BASE_URL`
- `LOCAL_LLM_MODEL`
- `INTERNAL_API_KEY` (must match backend `INTERNAL_API_KEY` for `/api/internal/**`)
- Optional:
  - `LOCAL_LLM_API_KEY` (default: `no-key-required`)
  - `LLM_TIMEOUT_SECONDS` (default: `60`)

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
