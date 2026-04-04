from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib

from app.domain.financial_training.ports.artifact_store import (
    ArtifactStorePort,
)


class SklearnArtifactStore(ArtifactStorePort):
    """
    Adapter: saves sklearn artifacts (model, scaler) and metadata.
    """

    def save(self, artifact: Any, artifact_name: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{artifact_name}.joblib"
        joblib.dump(artifact, out)
        return out

    def save_metadata(
        self,
        metadata: Mapping[str, Any],
        artifact_name: str,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{artifact_name}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        return out

