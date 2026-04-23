from __future__ import annotations

from typing import Protocol


class BackendSyncPort(Protocol):
    async def push_data(self, endpoint: str, data: list[dict]) -> int | None:
        ...
