"""Verify Voyage rerank wiring in RagRetrievalService."""
from __future__ import annotations

import json

import pytest
import respx
import httpx


@pytest.fixture
def rag_env(monkeypatch):
    """Configure settings so retrieve() uses mocked Voyage URLs."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "VOYAGE_API_KEY", "pa-test-key")
    monkeypatch.setattr(settings, "VOYAGE_EMBED_BASE_URL", "http://voyage.test/v1")
    monkeypatch.setattr(settings, "VOYAGE_EMBED_MODEL", "voyage-3.5-lite")
    monkeypatch.setattr(settings, "VOYAGE_EMBED_INPUT_TYPE", "query")
    monkeypatch.setattr(settings, "VOYAGE_RERANK_URL", "http://voyage.test/v1/rerank")
    monkeypatch.setattr(settings, "VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
    monkeypatch.setattr(settings, "CHAT_QDRANT_URL", "http://qdrant.test")
    monkeypatch.setattr(settings, "CHAT_QDRANT_COLLECTION", "test")
    monkeypatch.setattr(settings, "CHAT_RAG_CHUNKS_DB", "/dev/null/missing.sqlite")
    monkeypatch.setattr(settings, "CHAT_RAG_RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "CHAT_RAG_RERANK_CANDIDATES", 64)
    monkeypatch.setattr(settings, "CHAT_RAG_TOPK_FINAL", 5)


def _mock_embed_handler(request):
    return httpx.Response(200, json={
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
    })


async def test_rerank_reorders_candidates(rag_env):
    """Reranker reorders candidates and final list reflects the rerank scores."""
    from app.infrastructure.rag_client import RagRetrievalService

    service = RagRetrievalService()

    # Replace the heavy parts with deterministic stubs.
    candidate_pool = [
        {"chunk_id": f"c{i}", "score": 0.5, "source": "vector"}
        for i in range(10)
    ]

    async def fake_vector(*args, **kwargs):
        return candidate_pool

    async def fake_keyword(*args, **kwargs):
        return []

    def fake_load_details(chunk_ids):
        return {
            cid: {
                "payload": {"source_file": f"report-{cid}", "page_start": 1},
                "text": f"Chunk text for {cid}",
            }
            for cid in chunk_ids
        }

    service._vector_search = fake_vector  # type: ignore
    service._keyword_search = fake_keyword  # type: ignore
    service._load_chunk_details = fake_load_details  # type: ignore

    # Reranker reverses the order: c9 most relevant, c0 least.
    def rerank_handler(request):
        body = json.loads(request.content.decode())
        n_docs = len(body["documents"])
        # Simulate cross-encoder picking the last documents as most relevant.
        results = [
            {"index": i, "relevance_score": float(i) / n_docs, "document": {"text": body["documents"][i]}}
            for i in reversed(range(n_docs))
        ]
        top_k = body.get("top_k") or n_docs
        return httpx.Response(200, json={"data": results[:top_k], "model": "test"})

    with respx.mock(assert_all_called=False) as m:
        m.post("http://voyage.test/v1/rerank").mock(side_effect=rerank_handler)
        result = await service.retrieve(query="chiến lược kinh doanh", ticker="HPG")

    assert len(result) == 5
    # Rerank reverses, so c9 should be first.
    assert result[0]["chunk_id"] == "c9"
    assert result[-1]["chunk_id"] == "c5"
    # Rerank score is preserved.
    assert "rerank_score" in result[0]
    assert result[0]["rerank_provider"] == "voyage"
    assert result[0]["rerank_score"] > result[-1]["rerank_score"]


async def test_rerank_failure_falls_back_to_rrf(rag_env, monkeypatch):
    """If rerank endpoint errors, retrieve still returns RRF top-K."""
    from app.infrastructure.rag_client import RagRetrievalService

    service = RagRetrievalService()

    async def fake_vector(*args, **kwargs):
        return [{"chunk_id": f"c{i}", "score": 1.0 / (i + 1), "source": "vector"} for i in range(10)]

    async def fake_keyword(*args, **kwargs):
        return []

    def fake_load_details(chunk_ids):
        return {cid: {"payload": {}, "text": f"text-{cid}"} for cid in chunk_ids}

    service._vector_search = fake_vector  # type: ignore
    service._keyword_search = fake_keyword  # type: ignore
    service._load_chunk_details = fake_load_details  # type: ignore

    with respx.mock(assert_all_called=False) as m:
        m.post("http://voyage.test/v1/rerank").mock(return_value=httpx.Response(503))
        result = await service.retrieve(query="rủi ro", ticker="VCB")

    # Falls back to RRF order.
    assert len(result) == 5
    assert result[0]["chunk_id"] == "c0"  # RRF preserves the highest-RRF item


async def test_rerank_disabled_skips_endpoint(rag_env, monkeypatch):
    """When CHAT_RAG_RERANK_ENABLED=false, never POST to rerank URL."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "CHAT_RAG_RERANK_ENABLED", False)

    from app.infrastructure.rag_client import RagRetrievalService
    service = RagRetrievalService()

    async def fake_vector(*args, **kwargs):
        return [{"chunk_id": "c0", "score": 1.0, "source": "vector"}]

    async def fake_keyword(*args, **kwargs):
        return []

    def fake_load_details(chunk_ids):
        return {cid: {"payload": {}, "text": f"text-{cid}"} for cid in chunk_ids}

    service._vector_search = fake_vector  # type: ignore
    service._keyword_search = fake_keyword  # type: ignore
    service._load_chunk_details = fake_load_details  # type: ignore

    with respx.mock(assert_all_called=False) as m:
        rerank_route = m.post("http://voyage.test/v1/rerank").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = await service.retrieve(query="x", ticker="HPG")

    assert len(result) == 1
    assert rerank_route.call_count == 0
