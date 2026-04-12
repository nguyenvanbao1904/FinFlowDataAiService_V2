import logging

import httpx

from app.core.config import settings
from app.core.http_client import get_http_client
from app.domain.ports.backend_sync_port import BackendSyncPort

logger = logging.getLogger(__name__)


class JavaBackendClient(BackendSyncPort):
    """Client to communicate with the internal FinFlow Java Backend."""

    def __init__(self) -> None:
        self.base_url = settings.JAVA_BACKEND_URL or "http://localhost:8080/api/internal"
        self.internal_api_key = settings.INTERNAL_API_KEY.strip()

    async def push_data(self, endpoint: str, data: list[dict]) -> int | None:
        if not data:
            return None

        url = f"{self.base_url}/investment/sync/{endpoint}"
        headers: dict[str, str] = {}
        if self.internal_api_key:
            headers["X-Internal-Api-Key"] = self.internal_api_key

        client = get_http_client()
        try:
            response = await client.post(
                url, json=data, headers=headers, timeout=httpx.Timeout(30.0),
            )
            response.raise_for_status()
            logger.info("Pushed %d records to /%s", len(data), endpoint)
            return response.status_code
        except httpx.HTTPError as e:
            logger.error("Failed to push data to /%s: %s", endpoint, e)
            raise
