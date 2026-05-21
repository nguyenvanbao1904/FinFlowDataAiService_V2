from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.core.http_client import get_http_client

logger = logging.getLogger(__name__)


class RagRetrievalService:
    def __init__(self) -> None:
        self.chunks_db_path = Path(settings.CHAT_RAG_CHUNKS_DB)
        self.qdrant_url = (settings.CHAT_QDRANT_URL or "").strip()
        self.qdrant_api_key = (settings.CHAT_QDRANT_API_KEY or "").strip()
        self.qdrant_collection = (settings.CHAT_QDRANT_COLLECTION or "").strip()
        self.vector_topk = max(1, int(settings.CHAT_RAG_TOPK_VECTOR))
        self.keyword_topk = max(1, int(settings.CHAT_RAG_TOPK_KEYWORD))
        self.final_topk = max(1, int(settings.CHAT_RAG_TOPK_FINAL))
        self._qdrant_client: Any = None
        self._retrieve_traces: list[dict[str, Any]] = []

    def pop_retrieve_traces(self) -> list[dict[str, Any]]:
        """Return accumulated RAG debug traces for the current request and clear."""
        traces = self._retrieve_traces
        self._retrieve_traces = []
        return traces

    def _get_qdrant_client(self) -> Any:
        """Lazy-init and cache QdrantClient across queries."""
        if self._qdrant_client is not None:
            return self._qdrant_client
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            logger.warning("qdrant-client not installed, vector search disabled")
            return None

        try:
            self._qdrant_client = QdrantClient(
                url=self.qdrant_url.rstrip("/"),
                api_key=self.qdrant_api_key or None,
                timeout=15.0,
            )
            return self._qdrant_client
        except Exception:
            logger.exception("Failed to create QdrantClient")
            return None

    async def retrieve(
        self,
        query: str,
        ticker: str | None,
        years: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        normalized_years = self._normalize_years(years)

        vector_task = asyncio.create_task(self._vector_search(query=query, ticker=ticker, years=normalized_years))
        keyword_task = asyncio.create_task(self._keyword_search(query=query, ticker=ticker, years=normalized_years))

        vector_hits_raw, keyword_hits_raw = await asyncio.gather(vector_task, keyword_task, return_exceptions=True)

        vector_hits = vector_hits_raw if isinstance(vector_hits_raw, list) else []
        keyword_hits = keyword_hits_raw if isinstance(keyword_hits_raw, list) else []

        if isinstance(vector_hits_raw, BaseException):
            logger.warning("Vector search failed: %s", vector_hits_raw)
        if isinstance(keyword_hits_raw, BaseException):
            logger.warning("Keyword search failed: %s", keyword_hits_raw)

        # Pull a wider candidate pool when reranking is on, so the cross-encoder
        # has enough signal to surface the truly best chunks.
        rerank_enabled = bool(settings.CHAT_RAG_RERANK_ENABLED)
        candidate_limit = (
            max(int(settings.CHAT_RAG_RERANK_CANDIDATES), self.final_topk)
            if rerank_enabled
            else max(self.final_topk * 2, 8)
        )

        merged = self._merge_rrf(vector_hits, keyword_hits, limit=candidate_limit)
        if not merged:
            return []

        details = await asyncio.to_thread(self._load_chunk_details, [row["chunk_id"] for row in merged])

        candidates: list[dict[str, Any]] = []
        for row in merged:
            chunk_id = row["chunk_id"]
            detail = details.get(chunk_id, {})
            payload = detail.get("payload", {})
            text = detail.get("text", "")
            source_title = (
                payload.get("subsection_title")
                or payload.get("chapter_hint")
                or payload.get("source_file")
                or payload.get("category")
                or "Annual report chunk"
            )
            page_number = payload.get("page_start")
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "source_title": str(source_title),
                    "page_number": int(page_number) if isinstance(page_number, int) else None,
                    "score": float(row["score"]),
                    "text": text,
                }
            )

        trace_enabled = bool(settings.CHAT_TRACE_ENABLED)

        if rerank_enabled and len(candidates) > 1:
            reranked = await self._rerank_voyage(query, candidates)
            if reranked is not None:
                final = self._prepare_llm_contexts(reranked[: self.final_topk])
                if trace_enabled:
                    self._retrieve_traces.append({
                        "query": query,
                        "pre_rerank": _snapshot(candidates),
                        "post_rerank": _snapshot(final, score_key="rerank_score"),
                    })
                return final
            logger.info("Rerank fallback to RRF order")

        final = self._prepare_llm_contexts(candidates[: self.final_topk])
        if trace_enabled:
            self._retrieve_traces.append({
                "query": query,
                "pre_rerank": _snapshot(final),
                "post_rerank": None,
            })
        return final

    @staticmethod
    def _prepare_llm_contexts(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_chars = int(settings.CHAT_RAG_CONTEXT_MAX_CHARS or 0)
        if max_chars <= 0:
            return [dict(chunk) for chunk in chunks]

        prepared: list[dict[str, Any]] = []
        for chunk in chunks:
            row = dict(chunk)
            text = row.get("text")
            if isinstance(text, str) and len(text) > max_chars:
                row["text"] = text[:max_chars]
            prepared.append(row)
        return prepared

    async def _rerank_voyage(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        endpoint = (settings.VOYAGE_RERANK_URL or "https://api.voyageai.com/v1/rerank").strip()
        api_key = (settings.VOYAGE_API_KEY or "").strip()
        model = (settings.VOYAGE_RERANK_MODEL or "rerank-2.5-lite").strip()
        if not endpoint or not api_key or not model:
            logger.warning("Voyage rerank is not configured")
            return None

        documents = [c.get("text") or "" for c in candidates]
        payload = {
            "query": query,
            "documents": documents,
            "model": model,
            "top_k": self.final_topk,
            "return_documents": False,
            "truncation": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(max(2, int(settings.CHAT_RAG_RERANK_TIMEOUT_SECONDS)))

        try:
            client = get_http_client()
            response = await client.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Voyage rerank HTTP %s: %s",
                exc.response.status_code, exc.response.text[:300],
            )
            return None
        except httpx.TimeoutException:
            logger.warning("Voyage rerank timeout (%ds)", timeout.read)
            return None
        except Exception as exc:
            logger.warning("Voyage rerank failed: %s: %s", type(exc).__name__, exc)
            return None

        results = body.get("data") if isinstance(body, dict) else None
        if not isinstance(results, list):
            results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list) or not results:
            logger.warning(
                "Voyage rerank returned empty/invalid body: keys=%s",
                list(body.keys()) if isinstance(body, dict) else type(body).__name__,
            )
            return None

        reordered: list[dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            score = entry.get("relevance_score")
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                continue
            chunk = dict(candidates[idx])
            try:
                chunk["rerank_score"] = float(score)
            except (TypeError, ValueError):
                pass
            chunk["rerank_provider"] = "voyage"
            reordered.append(chunk)
        return reordered or None

    async def _vector_search(self, query: str, ticker: str | None, years: list[int]) -> list[dict[str, Any]]:
        if not (self.qdrant_url and self.qdrant_collection):
            return []

        vector = await self._embed_query(query)
        if not vector:
            return []

        return await asyncio.to_thread(self._query_qdrant, vector, ticker, years)

    async def _embed_query(self, query: str) -> list[float]:
        base_url = (settings.VOYAGE_EMBED_BASE_URL or "").strip()
        model = (settings.VOYAGE_EMBED_MODEL or "").strip()
        if not base_url or not model:
            return []

        endpoint = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {(settings.VOYAGE_API_KEY or '').strip()}",
        }
        payload = {"model": model, "input": query}
        input_type = (settings.VOYAGE_EMBED_INPUT_TYPE or "query").strip().lower()
        if input_type:
            payload["input_type"] = input_type
        timeout = httpx.Timeout(max(5, int(settings.LLM_TIMEOUT_SECONDS)))

        try:
            client = get_http_client()
            response = await client.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()

            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list) or not data:
                return []
            first = data[0] if isinstance(data[0], dict) else {}
            embedding = first.get("embedding")
            if not isinstance(embedding, list):
                return []
            return [float(x) for x in embedding]
        except Exception:
            logger.exception("Embedding query failed")
            return []

    def _query_qdrant(self, vector: list[float], ticker: str | None, years: list[int]) -> list[dict[str, Any]]:
        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue
        except ImportError:
            return []

        client = self._get_qdrant_client()
        if client is None:
            return []

        try:
            year_candidates = years if years else [None]
            aggregated: dict[str, float] = {}

            for year in year_candidates:
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
                for point in list(response.points or []):
                    payload = point.payload or {}
                    chunk_id = str(payload.get("chunk_id", "")).strip()
                    if not chunk_id:
                        continue
                    score = float(point.score)
                    prev = aggregated.get(chunk_id)
                    if prev is None or score > prev:
                        aggregated[chunk_id] = score

            hits = [{"chunk_id": cid, "score": score, "source": "vector"} for cid, score in aggregated.items()]
            hits.sort(key=lambda x: float(x["score"]), reverse=True)
            return hits[: self.vector_topk]
        except Exception:
            logger.exception("Qdrant query failed")
            return []

    async def _keyword_search(self, query: str, ticker: str | None, years: list[int]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._keyword_search_sync, query, ticker, years)

    def _keyword_search_sync(self, query: str, ticker: str | None, years: list[int]) -> list[dict[str, Any]]:
        if not self.chunks_db_path.exists():
            return []

        fts_hits = self._keyword_search_fts_sync(query, ticker, years)
        if fts_hits is not None:
            return fts_hits

        return self._keyword_search_count_fallback(query, ticker, years)

    def _keyword_search_fts_sync(
        self,
        query: str,
        ticker: str | None,
        years: list[int],
    ) -> list[dict[str, Any]] | None:
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT chunk_id, bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
        """
        params: list[Any] = [fts_query]
        if ticker:
            sql += " AND stock_code = ?"
            params.append(str(ticker).upper())
        if years:
            placeholders = ",".join("?" for _ in years)
            sql += f" AND CAST(year AS INTEGER) IN ({placeholders})"
            params.extend([int(y) for y in years])
        sql += " ORDER BY bm25_score ASC LIMIT ?"
        params.append(self.keyword_topk)

        try:
            with sqlite3.connect(str(self.chunks_db_path)) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.info("SQLite FTS keyword search unavailable, falling back to count scoring: %s", exc)
            return None
        except Exception:
            logger.exception("SQLite FTS keyword query failed for %s", self.chunks_db_path)
            return None

        hits: list[dict[str, Any]] = []
        for chunk_id, raw_score in rows:
            try:
                score = -float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            hits.append({"chunk_id": str(chunk_id), "score": score, "source": "keyword"})
        return hits

    def _keyword_search_count_fallback(self, query: str, ticker: str | None, years: list[int]) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        sql = "SELECT chunk_id, chunk_json FROM chunks WHERE 1=1"
        params: list[Any] = []
        if ticker:
            sql += " AND stock_code = ?"
            params.append(str(ticker).upper())
        if years:
            placeholders = ",".join("?" for _ in years)
            sql += f" AND year IN ({placeholders})"
            params.extend([int(y) for y in years])
        sql += " LIMIT 800"

        rows: list[tuple[str, str]] = []
        try:
            with sqlite3.connect(str(self.chunks_db_path)) as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception:
            logger.exception("SQLite keyword query failed for %s", self.chunks_db_path)
            return []

        scored: list[dict[str, Any]] = []
        for chunk_id, chunk_json in rows:
            try:
                payload = json.loads(chunk_json or "{}")
            except Exception:
                payload = {}
            text = self._extract_text(payload)
            if not text:
                continue
            score = self._keyword_score(text, keywords)
            if score <= 0:
                continue
            scored.append({"chunk_id": str(chunk_id), "score": float(score), "source": "keyword"})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.keyword_topk]

    def _load_chunk_details(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids or not self.chunks_db_path.exists():
            return {}

        placeholders = ",".join("?" for _ in chunk_ids)
        sql = f"SELECT chunk_id, chunk_json FROM chunks WHERE chunk_id IN ({placeholders})"
        out: dict[str, dict[str, Any]] = {}

        try:
            with sqlite3.connect(str(self.chunks_db_path)) as conn:
                for chunk_id, chunk_json in conn.execute(sql, chunk_ids).fetchall():
                    try:
                        payload = json.loads(chunk_json or "{}")
                        if not isinstance(payload, dict):
                            payload = {}
                    except Exception:
                        payload = {}
                    out[str(chunk_id)] = {
                        "payload": payload,
                        "text": self._extract_text(payload),
                    }
        except Exception:
            logger.exception("SQLite chunk detail load failed")
            return {}

        return out

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        for key in ("text", "chunk_text", "content"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        tokens = re.findall(r"[\wÀ-ỹ]{2,}", query.lower())
        stopwords = {"la", "là", "cua", "của", "the", "va", "và", "cho", "nhung", "những", "nam", "năm", "bao", "nhiêu"}
        dedup: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in stopwords:
                continue
            if tok not in seen:
                seen.add(tok)
                dedup.append(tok)
        return dedup[:12]

    @staticmethod
    def _build_fts_query(query: str) -> str:
        terms: list[str] = []
        for token in RagRetrievalService._extract_keywords(query):
            safe = token.replace('"', '""')
            if not safe:
                continue
            if len(safe) >= 3:
                terms.append(f'"{safe}"*')
            else:
                terms.append(f'"{safe}"')
        return " OR ".join(terms)

    @staticmethod
    def _keyword_score(text: str, keywords: list[str]) -> float:
        hay = text.lower()
        score = 0.0
        for idx, kw in enumerate(keywords):
            count = hay.count(kw)
            if count <= 0:
                continue
            weight = 1.0 if idx < 3 else 0.6
            score += count * weight
        return score

    @staticmethod
    def _merge_rrf(vector_hits: list[dict[str, Any]], keyword_hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        rank_scores: dict[str, float] = defaultdict(float)

        def add_rrf(hits: list[dict[str, Any]]) -> None:
            for rank, hit in enumerate(hits, start=1):
                chunk_id = str(hit.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                rank_scores[chunk_id] += 1.0 / (60 + rank)

        add_rrf(vector_hits)
        add_rrf(keyword_hits)

        merged = [{"chunk_id": chunk_id, "score": score} for chunk_id, score in rank_scores.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:limit]

    @staticmethod
    def _normalize_years(years: list[int] | None) -> list[int]:
        if not years:
            return []
        out: list[int] = []
        seen: set[int] = set()
        for value in years:
            try:
                y = int(value)
            except Exception:
                continue
            if y < 1990 or y > 2100:
                continue
            if y in seen:
                continue
            seen.add(y)
            out.append(y)
        out.sort()
        return out[:6]


def _snapshot(candidates: list[dict[str, Any]], score_key: str = "score") -> list[dict[str, Any]]:
    """Compact summary of each candidate for trace — no full text."""
    out = []
    for i, c in enumerate(candidates):
        out.append({
            "rank": i + 1,
            "chunk_id": c.get("chunk_id"),
            "source_title": (c.get("source_title") or "")[:80],
            "page_number": c.get("page_number"),
            "rrf_score": c.get("score"),
            "rerank_score": c.get("rerank_score"),
        })
    return out
