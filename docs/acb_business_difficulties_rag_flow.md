# Luong RAG khi user hoi: "kho khan cua ACB trong kinh doanh la gi?"

Tai lieu nay mo ta tung buoc runtime hien tai se lam gi khi user hoi cau: **"kho khan cua ACB trong kinh doanh la gi?"**

## 1. Backend goi Data AI Service

Khi user gui tin nhan, backend build payload gom `thread_id`, `user_id`, `user_message`, `context_summary`, `last_messages`, roi POST sang Data AI Service endpoint `/api/v1/ai/chat/orchestrate`.

Code minh chung:

- [RestDataAiChatAdapter.java](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/backend/src/main/java/com/finflow/backend/ai_chat/infrastructure/adapter/RestDataAiChatAdapter.java:43)
- Cac field payload duoc set tai lines 45-59:

```java
payload.put("thread_id", command.threadId());
payload.put("user_id", command.userId());
payload.put("user_message", command.userMessage());
payload.put("context_summary", command.contextSummary());
payload.put("last_messages", messages);
```

Voi cau hoi nay, `user_message` se la noi dung user vua nhap. Neu user hoi bang tieng Viet khong dau, agent van nhan raw text do.

## 2. ChatOrchestrator tao agent deps va chay LLM agent

Data AI Service nhan request vao `ChatOrchestrator.orchestrate()`.

Code minh chung:

- [orchestrator.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/services/chat/orchestrator.py:65)

Tai lines 66-70, orchestrator tao dependencies cho tool:

```python
deps = AppDeps(
    user_id=request.user_id,
    market_client=self._market_client,
    rag_service=self._rag_service,
    cfo_context=_is_cfo_context(request.user_message),
)
```

Tai lines 83-89, agent duoc chay voi user message va history:

```python
result = await self._agent.run(
    request.user_message,
    message_history=_build_history(request),
    deps=deps,
    usage_limits=_USAGE_LIMITS,
)
```

Voi cau "kho khan cua ACB trong kinh doanh la gi?", agent se can thong tin dinh tinh tu bao cao thuong nien, nen kha nang cao se goi RAG tool `search_annual_reports`.

## 3. Agent co RAG tool neu `CHAT_RAG_ENABLED=true`

Agent duoc build trong `build_chat_agent()`.

Code minh chung:

- [agent_tools.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/services/chat/agent_tools.py:58)
- Tai lines 88-92:

```python
_register_market_tools(agent)
_register_personal_finance_tools(agent)
_register_compute_tool(agent)
if bool(settings.CHAT_RAG_ENABLED):
    _register_rag_tool(agent)
```

Nghia la neu config RAG bat, agent co tool doc bao cao thuong nien.

## 4. LLM chon tool `search_annual_reports`

Tool RAG duoc khai bao tai:

- [agent_tools.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/services/chat/agent_tools.py:629)

Docstring cua tool noi ro dung khi can "chien luoc kinh doanh, rui ro, quan tri, ke hoach mo rong, trien vong nganh, giai thich nguyen nhan bien dong tai chinh":

```python
"""Tim kiem thong tin dinh tinh tu bao cao thuong nien (~700 cong ty, 5 nam 2019-2024).
Dung khi can: chien luoc kinh doanh, rui ro, quan tri, ke hoach mo rong, trien vong nganh,
giai thich nguyen nhan bien dong tai chinh.
...
"""
```

Voi cau hoi "kho khan cua ACB trong kinh doanh la gi?", LLM nen goi:

```json
{
  "ticker": "ACB",
  "query": "kho khan trong kinh doanh rui ro thach thuc chien luoc"
}
```

Luu y: query cu the do LLM quyet dinh, code khong hard-code query nay. Code chi bat buoc co `ticker` va `query`.

Tai lines 641-650, tool normalize input va goi retrieval:

```python
ticker_clean = (ticker or "").strip().upper()
query_clean = (query or "").strip()
chunks = await ctx.deps.rag_service.retrieve(
    query=query_clean, ticker=ticker_clean, years=None,
)
```

Nen ticker vao retrieval se la `ACB`; years hien la `None`, tuc la search tat ca nam cua ACB trong kho chunk.

## 5. RagRetrievalService chay song song vector search va keyword BM25 search

Runtime RAG nam o:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:59)

Tai lines 68-73:

```python
normalized_years = self._normalize_years(years)

vector_task = asyncio.create_task(self._vector_search(query=query, ticker=ticker, years=normalized_years))
keyword_task = asyncio.create_task(self._keyword_search(query=query, ticker=ticker, years=normalized_years))

vector_hits_raw, keyword_hits_raw = await asyncio.gather(vector_task, keyword_task, return_exceptions=True)
```

Voi `years=None`, `normalized_years` la list rong. Hai nhanh chay song song:

- Qdrant semantic vector search.
- SQLite FTS5 BM25 keyword search.

Neu mot nhanh loi, code khong lam fail toan bo retrieval; no log warning va dung list rong cho nhanh loi.

## 6. Nhanh vector: embed query roi query Qdrant voi filter `stock_code=ACB`

Vector search bat dau tai:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:217)

Tai lines 221-225:

```python
vector = await self._embed_query(query)
if not vector:
    return []

return await asyncio.to_thread(self._query_qdrant, vector, ticker, years)
```

Embedding query goi local embedding endpoint tai lines 227-254. Sau do `_query_qdrant()` filter theo ticker:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:259)

Tai lines 273-287:

```python
conditions = []
if ticker:
    conditions.append(FieldCondition(key="stock_code", match=MatchValue(value=str(ticker).upper())))
if year is not None:
    conditions.append(FieldCondition(key="year", match=MatchValue(value=int(year))))
query_filter = Filter(must=conditions) if conditions else None

response = client.query_points(
    collection_name=self.qdrant_collection,
    query=vector,
    query_filter=query_filter,
    limit=self.vector_topk,
    with_payload=True,
)
```

Voi cau hoi nay, filter Qdrant se co:

```python
stock_code = "ACB"
```

Ket qua vector la list chunk ids, score semantic tu Qdrant:

```python
hits = [{"chunk_id": cid, "score": score, "source": "vector"} ...]
```

## 7. Nhanh keyword: SQLite FTS5 BM25 voi filter `stock_code=ACB`

Keyword search bat dau tai:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:305)

Tai lines 312-316, code uu tien FTS5/BM25:

```python
fts_hits = self._keyword_search_fts_sync(query, ticker, years)
if fts_hits is not None:
    return fts_hits

return self._keyword_search_count_fallback(query, ticker, years)
```

Nghia la keyword-count cu chi la fallback neu DB chua co FTS5 hoac SQLite loi FTS.

FTS query duoc build an toan tai:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:440)

Tai lines 440-464:

```python
tokens = re.findall(r"[\wÀ-ỹ]{2,}", query.lower())
...
if len(safe) >= 3:
    terms.append(f'"{safe}"*')
else:
    terms.append(f'"{safe}"')
return " OR ".join(terms)
```

Vi du neu LLM query la `"kho khan trong kinh doanh rui ro thach thuc chien luoc"`, query FTS co dang gan nhu:

```text
"kho"* OR "khan"* OR "trong"* OR "kinh"* OR "doanh"* OR "rui"* OR "ro" OR "thach"* OR "thuc"* OR "chien"* OR "luoc"*
```

SQLite BM25 query nam tai lines 328-342:

```python
SELECT chunk_id, bm25(chunks_fts) AS bm25_score
FROM chunks_fts
WHERE chunks_fts MATCH ?
...
AND stock_code = ?
ORDER BY bm25_score ASC
LIMIT ?
```

Voi cau hoi nay, params se gom:

```python
[fts_query, "ACB", self.keyword_topk]
```

SQLite `bm25()` score cang nho cang lien quan. Code dao dau score tai lines 354-360:

```python
score = -float(raw_score)
hits.append({"chunk_id": str(chunk_id), "score": score, "source": "keyword"})
```

## 8. Hop nhat vector + keyword bang RRF

Sau khi co `vector_hits` va `keyword_hits`, code merge bang Reciprocal Rank Fusion:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:92)

Tai lines 92-96:

```python
merged = self._merge_rrf(vector_hits, keyword_hits, limit=candidate_limit)
details = await asyncio.to_thread(self._load_chunk_details, [row["chunk_id"] for row in merged])
```

Ham `_merge_rrf()` tai:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:479)

Tai lines 482-488:

```python
for rank, hit in enumerate(hits, start=1):
    chunk_id = str(hit.get("chunk_id", "")).strip()
    if not chunk_id:
        continue
    rank_scores[chunk_id] += 1.0 / (60 + rank)
```

Y nghia:

- Chunk xuat hien rank cao trong Qdrant duoc diem RRF.
- Chunk xuat hien rank cao trong BM25 duoc diem RRF.
- Chunk xuat hien tot o ca hai nhanh se duoc cong diem tu ca hai phia.

## 9. Load text va metadata chunk tu SQLite

Sau RRF, code can text that su de dua vao LLM. No load details tu SQLite:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:96)

Sau do build candidates tai lines 98-119:

```python
source_title = (
    payload.get("subsection_title")
    or payload.get("chapter_hint")
    or payload.get("source_file")
    or payload.get("category")
    or "Annual report chunk"
)
page_number = payload.get("page_start")
...
"text": text[:1800],
```

Moi candidate co:

- `chunk_id`
- `source_title`
- `page_number`
- `score`
- `text`

## 10. Neu bat reranker, rerank lai candidate truoc khi tra top chunks

Tai lines 124-135:

- [rag_client.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/infrastructure/rag_client.py:124)

```python
if rerank_enabled and len(candidates) > 1:
    reranked = await self._rerank(query, candidates)
    if reranked is not None:
        final = reranked[: self.final_topk]
        return final
    logger.info("Rerank fallback to RRF order")
```

Neu reranker service chay tot, ket qua cuoi cung la top chunks sau cross-encoder rerank. Neu reranker fail, code fallback ve thu tu RRF.

## 11. Tool tra ve toi LLM toi da 6 chunks

Sau khi `retrieve()` tra ve chunks, tool `search_annual_reports` chi tra toi da 6 chunks va cat text moi chunk 1200 ky tu:

- [agent_tools.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/services/chat/agent_tools.py:651)

```python
return [
    {
        "chunk_id": c.get("chunk_id"),
        "source_title": c.get("source_title"),
        "page_number": c.get("page_number"),
        "text": (c.get("text") or "")[:1200],
    }
    for c in (chunks or [])[:6]
]
```

LLM se doc nhung doan nay va tong hop cau tra loi ve kho khan cua ACB trong kinh doanh, vi du cac nhom noi dung co the la:

- Ap luc tang truong tin dung.
- Rui ro no xau/chi phi du phong.
- Canh tranh lai suat va NIM.
- Bien dong kinh te vi mo.
- Yeu cau quan tri rui ro, an toan von, chuyen doi so.

Noi dung chinh xac phu thuoc top chunks lay duoc tu bao cao ACB.

## 12. Orchestrator lay tool output lam citations

Sau khi agent co final answer, orchestrator doc lai message log de lay tool calls, tool results, va RAG chunks:

- [orchestrator.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/services/chat/orchestrator.py:189)

Tai lines 205-216:

```python
elif isinstance(part, ToolReturnPart):
    parsed = parse_llm_json(part.content) if isinstance(part.content, str) else part.content
    tool_results.append({...})
    if part.tool_name == _RAG_TOOL_NAME and isinstance(parsed, list):
        rag_chunks.extend(c for c in parsed if isinstance(c, dict))
```

Sau do tao response tai lines 126-140:

```python
response = ChatOrchestrateResponse(
    assistant_message=message,
    ...
    tool_calls=tool_calls,
    tool_results=tool_results,
    citations=[ChatCitation(**c) for c in _pick_citations(rag_chunks)],
    context_update=context_update,
)
```

Citations duoc pick tai:

- [orchestrator.py](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/data_ai_service/app/services/chat/orchestrator.py:277)

```python
{
    "chunk_id": r.get("chunk_id"),
    "source_title": r.get("source_title"),
    "page_number": r.get("page_number"),
    "score": r.get("score"),
}
```

Luu y hien tai: `search_annual_reports` khong tra `score`, nen citation score co the la `null` o response. Neu muon score hien len, can them `"score": c.get("score")` vao return cua tool.

## 13. Backend nhan response va luu citations

Backend parse citations tu Data AI response tai:

- [RestDataAiChatAdapter.java](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/backend/src/main/java/com/finflow/backend/ai_chat/infrastructure/adapter/RestDataAiChatAdapter.java:68)

Tai lines 68-77:

```java
citations.add(new Citation(
        asText(row, "chunk_id"),
        asText(row, "source_title"),
        asInteger(row, "page_number"),
        asDouble(row, "score")
));
```

Sau do `SendChatMessageUseCase` luu vao `chat_message_sources`:

- [SendChatMessageUseCase.java](/Users/nguyenvanbao/MyWorkspace/FinFlow_v2/backend/src/main/java/com/finflow/backend/ai_chat/application/usecase/SendChatMessageUseCase.java:119)

Tai lines 119-127:

```java
for (AiChatGatewayPort.Citation citation : aiResult.citations()) {
    sources.add(ChatMessageSource.builder()
            .messageId(assistantMessage.getId())
            .chunkId(citation.chunkId())
            .sourceTitle(citation.sourceTitle())
            .pageNumber(citation.pageNumber())
            .score(citation.score() == null ? null : BigDecimal.valueOf(citation.score()).setScale(6, RoundingMode.HALF_UP))
            .build());
}
```

## Tom tat ngan gon

Voi cau hoi **"kho khan cua ACB trong kinh doanh la gi?"**, flow runtime la:

1. Backend gui message sang Data AI `/chat/orchestrate`.
2. Orchestrator chay pydantic-ai agent.
3. Agent thay day la cau hoi dinh tinh/rui ro/kinh doanh nen goi `search_annual_reports`.
4. Tool goi `rag_service.retrieve(query=..., ticker="ACB", years=None)`.
5. RAG chay song song:
   - Qdrant semantic search filter `stock_code=ACB`.
   - SQLite FTS5 BM25 keyword search filter `stock_code=ACB`.
6. Hai nhanh hop nhat bang RRF.
7. Load text + page metadata tu SQLite.
8. Optional reranker sap xep lai.
9. Tool tra toi da 6 chunks cho LLM.
10. LLM tong hop cau tra loi.
11. Orchestrator trich citations tu chunks.
12. Backend luu assistant message va citations.

