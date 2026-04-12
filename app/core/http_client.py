"""Shared httpx.AsyncClient — single connection pool for all outbound HTTP.

Eliminates per-request client creation overhead (TCP handshakes, TLS negotiation).
All services share this pool. Per-request timeout overrides are supported:
    await get_http_client().post(url, timeout=httpx.Timeout(30.0))
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(max(5, settings.LLM_TIMEOUT_SECONDS)),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
        logger.info("Shared HTTP client closed")
