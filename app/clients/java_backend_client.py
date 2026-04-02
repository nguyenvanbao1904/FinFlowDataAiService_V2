import httpx
import logging
from app.core.config import settings
from app.domain.ports.backend_sync_port import BackendSyncPort

logger = logging.getLogger(__name__)

class JavaBackendClient(BackendSyncPort):
    """Client to communicate with the internal FinFlow Java Backend."""
    
    def __init__(self):
        # Default fallback if setting is not correctly loaded
        self.base_url = settings.JAVA_BACKEND_URL or "http://localhost:8080/api/internal"
        self.internal_api_key = settings.INTERNAL_API_KEY.strip()
        
    async def push_data(self, endpoint: str, data: list[dict]):
        if not data:
            return None
            
        url = f"{self.base_url}/investment/sync/{endpoint}"
        headers = {}
        if self.internal_api_key:
            headers["X-Internal-Api-Key"] = self.internal_api_key
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                response.raise_for_status()
                logger.info(f"✅ Successfully pushed {len(data)} records to /{endpoint}")
                return response.status_code
            except httpx.HTTPError as e:
                logger.error(f"❌ Failed to push data to /{endpoint}: {e}")
                raise
