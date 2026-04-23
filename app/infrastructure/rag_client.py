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

        merged = self._merge_rrf(vector_hits, keyword_hits, limit=max(self.final_topk * 2, 8))
        if not merged:
            return []

        details = await asyncio.to_thread(self._load_chunk_details, [row["chunk_id"] for row in merged])

        output: list[dict[str, Any]] = []
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
            output.append(
                {
                    "chunk_id": chunk_id,
                    "source_title": str(source_title),
                    "page_number": int(page_number) if isinstance(page_number, int) else None,
                    "score": float(row["score"]),
                    "text": text[:1800],
                }
            )

        return output[: self.final_topk]

    async def _vector_search(self, query: str, ticker: str | None, years: list[int]) -> list[dict[str, Any]]:
        if not (self.qdrant_url and self.qdrant_collection):
            return []

        vector = await self._embed_query(query)
        if not vector:
            return []

        return await asyncio.to_thread(self._query_qdrant, vector, ticker, years)

    async def _embed_query(self, query: str) -> list[float]:
        base_url = (settings.LOCAL_EMBEDDING_BASE_URL or "").strip()
        model = (settings.LOCAL_EMBEDDING_MODEL or "").strip()
        if not base_url or not model:
            return []

        endpoint = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {(settings.LOCAL_EMBEDDING_API_KEY or 'no-key-required').strip() or 'no-key-required'}",
        }
        payload = {"model": model, "input": query}
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
