from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class ArtifactStorePort(ABC):
    """
    Port: persists and retrieves training artifacts (model/scaler/metadata).
    """

    @abstractmethod
    def save(self, artifact: Any, artifact_name: str, output_dir: Path) -> Path:
        """
        Persists artifact and returns the saved path.
        """

    @abstractmethod
    def save_metadata(
        self, metadata: Mapping[str, Any], artifact_name: str, output_dir: Path
    ) -> Path:
        """
        Persists metadata and returns the saved path.
        """

