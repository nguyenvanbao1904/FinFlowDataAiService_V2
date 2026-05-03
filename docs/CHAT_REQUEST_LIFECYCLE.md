# Chat Request Lifecycle — Điều gì xảy ra khi user gõ 1 prompt

> **Mục đích file này:** Mô tả từng bước hệ thống xử lý 1 lượt chat từ khi user nhấn Send tới khi thấy câu trả lời. Dùng cho debug, onboarding dev mới, hoặc tham khảo khi đổi prompt/tool.
>
> **Cập nhật:** 2026-05-01 (sau Phase 7 — pydantic-ai best practices)

---

## 0. Bird's-eye view

```
┌────────┐  REST   ┌──────────────┐  HTTP  ┌──────────────────┐
│  iOS   │ ──────► │  Spring Boot │ ─────► │  data_ai_service │
│ App    │         │   Backend    │        │    (FastAPI)     │
└────────┘         └──────────────┘        └────────┬─────────┘
                                                    │
                          ┌─────────────────────────┴──────────────────┐
                          │                                            │
                          ▼                                            ▼
                  ┌───────────────┐                          ┌──────────────────┐
                  │  DeepSeek LLM │                          │  Tools layer     │
                  │  (OpenAI-     │ ◄──── tool calls ──────► │  - Backend HTTP  │
                  │   compatible) │                          │  - Qdrant + RAG  │
                  └───────────────┘                          │  - ML forecast   │
                                                             └──────────────────┘
```

---

## 1. Request đến FastAPI

**Endpoint:** `POST /api/v1/ai/chat/orchestrate`

**Auth:** Header `X-Internal-Api-Key` (Spring Boot tự inject — iOS không thấy)

**File handler:** [`app/routers/chat.py`](../app/routers/chat.py)

```python
@router.post("/orchestrate")
async def chat_orchestrate(request: ChatOrchestrateRequest):
    return await get_chat_orchestrator().orchestrate(request)
```

**Body shape (`ChatOrchestrateRequest`):**

```json
{
  "thread_id": "thread-uuid",
  "user_id": "user-uuid",
  "user_message": "HPG có rẻ không?",
  "context_summary": "...",
  "last_messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

`get_chat_orchestrator()` là singleton — `ChatOrchestrator` chỉ được build 1 lần khi service start, sau đó reuse cho mọi request.

---

## 2. ChatOrchestrator.orchestrate

**File:** [`app/services/chat/orchestrator.py:54`](../app/services/chat/orchestrator.py)

### 2.1. Build per-request `AppDeps`

```python
deps = AppDeps(
    user_id=request.user_id,           # ← inject vào tools, KHÔNG vào schema LLM
    market_client=self._market_client,  # shared HTTP wrapper
    rag_service=self._rag_service,      # shared Qdrant client
)
```

**Vì sao tách deps:** `user_id` không được expose ra schema mà LLM thấy → LLM không thể bị prompt-inject để gọi `add_transaction(user_id="kẻ tấn công")`. Tool function lấy `user_id` qua `ctx.deps.user_id`.

### 2.2. Build `message_history`

Convert `request.last_messages` + `context_summary` thành format pydantic-ai:

- `context_summary` → `ModelRequest(UserPromptPart(...))` (turn ẩn ở đầu)
- `last_messages[-8:]` → ModelRequest cho user, ModelResponse cho assistant
- Cap 8 turns gần nhất (`_HISTORY_LIMIT`)

### 2.3. Gọi `agent.run`

```python
result = await self._agent.run(
    request.user_message,
    message_history=_build_history(request),
    deps=deps,
    usage_limits=_USAGE_LIMITS,  # cap 8 LLM calls + 12 tool calls/turn
)
```

`_USAGE_LIMITS` chặn adversarial cost burn: nếu LLM kẹt loop tool, sau 8 lần gọi LLM hoặc 12 tool calls → raise → `_error_response`.

---

## 3. Bên trong `agent.run` (pydantic-ai làm)

**File:** [`app/services/chat/agent_tools.py`](../app/services/chat/agent_tools.py) — định nghĩa agent + tools.

### 3.1. Build prompt cho LLM

Pydantic-ai gom các phần sau thành 1 array `messages`:

1. **System prompt** (dynamic, từ `@agent.system_prompt`):
   - `AGENT_SYSTEM_PROMPT` (CFO persona, hướng dẫn chọn tool, format rules)
   - `+` Thời gian VN hiện tại
   - `+` `USER_ID của người dùng hiện tại: {ctx.deps.user_id}`

2. **Message history** (từ `_build_history`)

3. **User message** mới (`request.user_message`)

4. **Tools schema** (auto-generated từ Python signature + docstring của 19-20 `@agent.tool` functions)

### 3.2. ReAct loop

```
loop tối đa 8 lần (UsageLimits.request_limit):
  1. Gửi messages + tools lên DeepSeek
  2. Nhận response:
       - tool_calls (list)? → đi tiếp bước 3
       - text content?      → BREAK, đây là final answer
  3. Với mỗi tool_call:
       - pydantic-ai validate arguments theo type hints
       - lookup tool function trong agent registry
       - inject ctx (deps + retry counter)
       - await tool function
       - bắt exception:
           - ModelRetry  → gửi message lại cho LLM, lặp lại bước 1
           - khác        → bubble lên orchestrator
       - serialize result, append vào messages
  4. Quay lại bước 1
```

### 3.3. `@agent.output_validator` — sanitize trước khi return

Trước khi trả output cho orchestrator, pydantic-ai chạy `_sanitize_output` trong agent_tools.py:

- Strip markdown (`**bold**`, `# heading`)
- Replace technical jargon (`tác động cùng chiều` → `hỗ trợ`)
- Convert snake_case feature names sang nhãn tiếng Việt (`nim_pct` → `biên lãi thuần`)

→ Không cần gọi sanitize tay ở orchestrator nữa.

---

## 4. Tools — chuyện gì xảy ra khi LLM gọi tool

Có **4 loại tool**, mỗi loại routing khác nhau:

### 4.1. Market data tools (13 cái)

VD: `get_company_metrics`, `get_company_financial_series`, `get_company_daily_valuations`, ...

```
agent_tools.py
   └─ _call_market(ctx, "get_company_metrics", symbol="HPG")
        └─ MarketDataToolClient.execute_tool_call(...)
             └─ dispatch table _ROUTES → (path, params builder)
             └─ httpx.GET http://backend/investment/query/companies/HPG/analysis
             └─ _POST_PROCESSORS[name](payload) → strip dữ liệu thừa, tóm tắt
        └─ Return data hoặc raise ModelRetry/RuntimeError
```

**Post-processors quan trọng:**
- `get_company_daily_valuations`: tóm raw 1825 ngày × 5 năm thành `{pe_median, pb_median, ...}` (tiết kiệm context LLM)
- `get_company_metrics`: chỉ giữ field `overview` (bỏ shareholders, dividends thừa)
- `get_company_live_valuation_snapshot`: tính nhãn "rẻ tương đối" / "đắt tương đối" so với median

### 4.2. Personal finance tools (5 cái)

`get_personal_finance_report`, `get_user_transaction_context`, `add_transaction`, `get_user_budgets`, `add_budget`

```
agent_tools.py
   └─ _pf_request(ctx, "GET", "/transaction/finance-report")
        └─ httpx.{method}(backend_url, params={"userId": ctx.deps.user_id}, ...)
        └─ Return JSON
```

**Bảo mật:** `user_id` lấy từ `ctx.deps.user_id` → không phải từ args LLM truyền. Schema LLM chỉ thấy: `amount`, `type`, `categoryId`, `accountId`, `note`, `transactionDate`. Không có cách nào LLM "đổi user".

### 4.3. `compute_fair_value` (tool đặc biệt)

```
agent_tools.compute_fair_value(symbol, target_year)
   └─ valuation_inputs.fetch_valuation_inputs(market_client, symbol, target_year)
        └─ asyncio.gather(...) — fan out 4-N backend calls SONG SONG:
             - get_company_metrics
             - get_company_financial_series (annualLimit=6)
             - get_company_daily_valuations (5 năm)
             - get_company_live_valuation_snapshot
             - get_company_forecast (1 hoặc nhiều năm tùy target_year)
        └─ Aggregate → dict inputs cho compute
   └─ valuation_engine.compute_fair_value(inputs)  ← pure Python, no I/O
        └─ Industry playbook (ICB code → BANK/RETAIL/REAL_ESTATE/...)
        └─ CAGR (forecast hoặc historical)
        └─ Gordon growth + P/E target + P/B target
        └─ Return {fair_price, verdict, weights_used, summary}
```

→ Tool này tự gọi 4-5 backend calls, LLM **không** cần gọi nhiều tool trước đó.

### 4.4. `search_annual_reports` (RAG + rerank)

```
agent_tools.search_annual_reports(ticker, query)
   └─ RagRetrievalService.retrieve(query, ticker, years)
        ├─ asyncio.gather:
        │    ├─ vector_search:
        │    │    ├─ embed query qua local MLX server (bge-m3, port 9091)
        │    │    └─ Qdrant query (top 25) với filter ticker + năm
        │    └─ keyword_search:
        │         └─ SQLite full-text (top 25) trên chunks DB
        ├─ RRF merge (Reciprocal Rank Fusion) → top 30 candidates
        ├─ Load chunk text từ SQLite
        └─ Rerank (nếu CHAT_RAG_RERANK_ENABLED=true):
             ├─ POST /v1/rerank với (query, 30 documents)
             ├─ bge-reranker-v2-m3 cross-encoder trên Apple MPS
             ├─ Re-score 30 cặp (query, chunk) → sort theo relevance
             └─ Trả top 6 (precision +20-30% vs RRF only)
   └─ Return list[{chunk_id, source_title, page_number, text, rerank_score}]
```

**Vì sao có rerank**: vector search dùng bi-encoder (encode query và chunk độc
lập rồi cosine) — nhanh nhưng coarse. Cross-encoder đọc `(query, chunk)` cùng
lúc với attention chéo → nắm semantic chính xác hơn. Pattern best-practice
2026: retrieve nhiều rồi rerank, không phải retrieve ít rồi giữ.

**Fallback**: nếu rerank server down/timeout → log warning + fallback RRF
top 6. Không bao giờ block agent.

---

## 5. Self-healing với `ModelRetry`

Tool có thể `raise ModelRetry(message)` để **gửi message đó vào tool result cho LLM** thay vì crash:

| Tình huống | Behavior |
|---|---|
| Symbol sai (404 từ backend) | `ModelRetry("Không tìm thấy dữ liệu (404)...")` → LLM tự gọi `suggest_companies` để kiểm tra mã |
| Backend trả 400 | `ModelRetry("Backend từ chối yêu cầu: ...")` → LLM sửa params và gọi lại |
| `compute_fair_value` thiếu data | `ModelRetry("Không đủ dữ liệu để định giá X. Hãy thử suggest_companies('X') để xác minh mã.")` |
| `search_annual_reports` thiếu ticker | `ModelRetry("search_annual_reports cần cả ticker và query...")` |

→ LLM đọc message này như một tool result bình thường, tự sửa hành vi.

**Phân biệt với `RuntimeError`:** lỗi network/timeout/upstream → `RuntimeError` → bubble lên orchestrator → trả lỗi cho user. ModelRetry dành cho lỗi **logic có thể sửa được**.

---

## 6. Build response trả Spring Boot

Sau khi `agent.run()` xong, orchestrator extract metadata từ message log:

**File:** `orchestrator._extract_tool_io(result)`

Walk qua mọi `result.all_messages()`, mỗi `part`:
- `ToolCallPart` → append vào `tool_calls` (name + arguments)
- `ToolReturnPart` → append vào `tool_results` (name + parsed data + ok=True)
- `RetryPromptPart` → append vào `tool_results` (ok=False + error message)

Sau đó:
- Pick `last_ticker`, `last_year` từ tool args → `context_update` (Spring Boot lưu vào `chat_threads`)
- Pick top 5 RAG chunks → `citations`
- Detect câu hỏi mở → `needs_clarification`
- Compute cost: `estimate_cost(input_tokens, output_tokens)` theo giá DeepSeek

**Response shape (`ChatOrchestrateResponse`):**

```json
{
  "assistant_message": "HPG đang giao dịch...",
  "needs_clarification": false,
  "provider": "deepseek",
  "model": "deepseek-chat",
  "input_tokens": 4521,
  "output_tokens": 312,
  "total_tokens": 4833,
  "cost_usd": 0.0006,
  "latency_ms": 8420,
  "tool_calls": [
    {"name": "compute_fair_value", "arguments": {"symbol": "HPG"}}
  ],
  "tool_results": [
    {"name": "compute_fair_value", "ok": true, "data": {...}}
  ],
  "citations": [
    {"chunk_id": "...", "source_title": "...", "page_number": 12, "score": 0.85}
  ],
  "context_update": {
    "last_ticker": "HPG",
    "last_year": 2026
  }
}
```

Spring Boot lưu vào DB (`chat_messages`, `chat_message_sources`) rồi forward `assistant_message` + citations về iOS.

---

## 7. Error paths

```
Tool raise ModelRetry         → LLM tự sửa, không crash
Tool raise RuntimeError       → bubble lên agent.run → orchestrator catches
                                  → return error response, latency=0
Tool timeout (>30s)           → httpx raises → RuntimeError path
LLM API timeout (>60s)        → httpx raises → orchestrator catches
LLM API 4xx/5xx               → bubble lên orchestrator
UsageLimitExceeded            → bubble lên orchestrator (loop quá dài)
```

Mọi lỗi cuối cùng đều convert thành `ChatOrchestrateResponse` có `assistant_message` thân thiện với user (`"Xin lỗi, đã xảy ra lỗi: ..."`), không leak stack trace.

---

## 8. Mini sequence diagram (HPG fair value)

```
User: "HPG có rẻ không, định giá thử xem"

iOS  ───► Spring Boot ───► /chat/orchestrate
                              │
                              ▼
                       Build AppDeps + history
                              │
                              ▼
                       agent.run("HPG có rẻ không...")
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    Turn 1: gửi DeepSeek → response: tool_call(compute_fair_value, symbol=HPG)
              │
              ▼
    Tool execution:
       └─ fetch_valuation_inputs (parallel):
            ├─ GET /companies/HPG/analysis
            ├─ GET /companies/HPG/analysis/financials?annualLimit=6
            ├─ GET /companies/HPG/analysis/valuations/daily?...
            ├─ GET /companies/HPG/analysis (snapshot mode)
            └─ GET /companies/HPG/forecast?targetYear=2026
       └─ valuation_engine.compute_fair_value(...)
       └─ return {fair_price: 32500, verdict: "RẺ", ...}
              │
              ▼
    Turn 2: gửi DeepSeek (kèm tool result)
              → response: text "HPG đang giao dịch ở 28,500đ..."
              │
              ▼
    @agent.output_validator: strip markdown, sanitize jargon
              │
              ▼
    Orchestrator extract tool_calls + citations + cost
              │
              ▼
    Return ChatOrchestrateResponse
              │
Spring Boot ◄─┘ → lưu DB → forward iOS
                              │
                              ▼
                            User thấy câu trả lời
```

---

## 9. Performance characteristics

| Metric | Thông thường | Worst case |
|---|---|---|
| Tổng latency end-to-end | 4-12s | 30s (UsageLimits cap) |
| LLM calls per turn | 1-3 | 8 (cap) |
| Tool calls per turn | 0-5 | 12 (cap) |
| Backend HTTP calls | 0-10 | 20+ (compute_fair_value với target_year xa) |
| Token usage | 2k-8k | 30k |
| Cost (DeepSeek) | $0.0001-0.001 | ~$0.005 |

---

## 10. Khi cần debug

| Triệu chứng | Check ở đâu |
|---|---|
| LLM gọi sai tool | `AGENT_SYSTEM_PROMPT` trong `prompts/agent_prompt.py` (hướng dẫn chọn tool) |
| Tool nhận args sai | Type hints trong `agent_tools.py` — pydantic-ai validate trước khi call |
| Backend trả 400/404 | Path template trong `_ROUTES` của `market_data_client.py` |
| LLM không tự sửa khi tool fail | Đang `raise RuntimeError` thay vì `ModelRetry` — đổi sang `ModelRetry` với message hữu ích |
| Response leak markdown/jargon | `_sanitize_output` trong `agent_tools.py` + `vietnamese_text.sanitize_user_facing_message` |
| Cost runaway | `_USAGE_LIMITS` trong `orchestrator.py` |
| Citations sai | `_pick_citations` trong orchestrator + RAG retrieval logic trong `rag_client.py` |
| Behavior thay đổi sau migrate | Run `python -m pytest tests/` — 10 baseline parity tests |
| Quality regression sau đổi prompt | Run `python -m evals.run_chat_eval` — golden dataset thật với DeepSeek |

---

## 11. Bật trace log để xem mọi I/O với DeepSeek

Khi cần debug "DeepSeek nhận gì, trả gì, qua bao nhiêu turn" — bật per-request trace.

### Bật

Trong `data_ai_service/.env`:

```bash
CHAT_TRACE_ENABLED=true
# CHAT_TRACE_DIR=...  # mặc định: artifacts/chat_traces/
```

Restart FastAPI. Mỗi request `/chat/orchestrate` sẽ ghi 1 file JSON ~10-50 KB vào `CHAT_TRACE_DIR`.

### Filename format

```
{YYYYMMDD-HHMMSS-mmm}_{thread_id}_{user_id}_{model_prefix}.json
```

Ví dụ: `20260502-154338-348_thread-demo_u-demo_deepse.json`

### Cấu trúc 1 trace file

```json
{
  "ts": "2026-05-02T15:43:38.348",
  "latency_ms": 8420,
  "error": null,
  "request": {
    "thread_id": "...",
    "user_id": "...",
    "user_message": "HPG có rẻ không?",
    "context_summary": "...",
    "history_messages": [...]
  },
  "agent_messages": [
    // pydantic-ai dump — đây là phần xem được FULL prompt + tool I/O
    {
      "kind": "request",
      "parts": [
        {"part_kind": "system-prompt", "content": "Bạn là CFO..."},
        {"part_kind": "user-prompt", "content": "HPG có rẻ không?"}
      ]
    },
    {
      "kind": "response",
      "parts": [
        {"part_kind": "tool-call", "tool_name": "compute_fair_value", "args": {...}}
      ]
    },
    {
      "kind": "request",
      "parts": [
        {"part_kind": "tool-return", "tool_name": "compute_fair_value", "content": {...}}
      ]
    },
    {
      "kind": "response",
      "parts": [
        {"part_kind": "text", "content": "HPG đang giao dịch..."}
      ]
    }
  ],
  "response": {
    "assistant_message": "...",
    "tool_calls": [...],
    "tool_results": [...],
    "citations": [...],
    "input_tokens": 4521,
    "output_tokens": 312,
    "cost_usd": 0.0006
  }
}
```

### Đọc nhanh 1 trace

```bash
# Latest trace
ls -t artifacts/chat_traces/ | head -1 | xargs -I{} python -m json.tool artifacts/chat_traces/{}

# Filter theo user
ls artifacts/chat_traces/*u-USER123*.json

# Tìm trace có error
grep -l '"error":' artifacts/chat_traces/*.json | head
```

### Lưu ý

- File ghi **best-effort**: nếu disk lỗi hoặc path sai → log warning, không crash request.
- File chứa **toàn bộ prompt** bao gồm `user_id` + `context_summary` + lịch sử chat → coi là PII, **không commit lên git**, không share công khai.
- Ghi đồng bộ trong response path → mỗi request +5-20ms latency. Production scale cao thì không nên bật mãi.
- Khi gặp `error` (UsageLimits exceeded, agent crash, tool timeout) → file vẫn được ghi với `error.type` + `error.message`, dù `agent_messages=null`.

### Cleanup

```bash
# Xoá trace cũ hơn 7 ngày
find artifacts/chat_traces -name "*.json" -mtime +7 -delete
```

---

## 12. File map

```
app/
├── routers/chat.py                       ← REST endpoint
├── core/dependencies.py                  ← singleton orchestrator
├── services/chat/
│   ├── orchestrator.py                   ← entry point (build deps, run agent, build response)
│   ├── agent_tools.py                    ← @agent.tool registry + AppDeps + system prompt
│   ├── prompts/agent_prompt.py           ← AGENT_SYSTEM_PROMPT (CFO persona, tool selection rules)
│   ├── valuation_engine.py               ← compute_fair_value pure Python
│   ├── valuation_inputs.py               ← fan-out backend calls cho compute_fair_value
│   └── utils/
│       ├── json_io.py                    ← parse_llm_json shared helper
│       └── vietnamese_text.py            ← sanitize markdown + jargon → tiếng Việt user-friendly
│   └── trace_writer.py                   ← per-request JSON trace dump (CHAT_TRACE_ENABLED)
├── infrastructure/
│   ├── llm_agent.py                      ← DeepSeek model factory + cost
│   ├── market_data_client.py             ← HTTP wrapper Spring Boot + post-processors
│   └── rag_client.py                     ← Qdrant + SQLite + RRF
├── models/chat.py                        ← Pydantic request/response shapes
└── core/config.py                        ← Settings từ .env

tests/                                    ← parity baseline (mock LLM + backend)
evals/                                    ← golden dataset (real LLM, đắt, chỉ chạy khi đổi prompt)
```
