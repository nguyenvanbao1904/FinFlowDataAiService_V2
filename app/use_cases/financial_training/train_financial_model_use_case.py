from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from app.domain.financial_training.ports.artifact_store import (
    ArtifactStorePort,
)
from app.domain.financial_training.ports.model_training import (
    ModelTrainerPort,
)


@dataclass(frozen=True)
class TrainRequest:
    features: Any
    labels: Any
    dataset_metadata: Optional[Mapping[str, Any]] = None
    artifact_name: str = "model"


@dataclass(frozen=True)
class TrainResult:
    artifact_path: Any


class TrainFinancialModelUseCase:
    """
    Use case: trains a model and stores the artifacts.
    """

    def __init__(
        self,
        trainer: ModelTrainerPort,
        artifact_store: ArtifactStorePort,
    ) -> None:
        self._trainer = trainer
        self._artifact_store = artifact_store

    def execute(
        self,
        request: TrainRequest,
        output_dir,
    ) -> TrainResult:
        trained = self._trainer.train(
            request.features,
            request.labels,
            dataset_metadata=request.dataset_metadata,
        )

        # Concrete artifact_store decides exact file extension/format.
        artifact_path = self._artifact_store.save(
            artifact=trained, artifact_name=request.artifact_name, output_dir=output_dir
        )

        if request.dataset_metadata is not None:
            self._artifact_store.save_metadata(
                metadata=request.dataset_metadata,
                artifact_name=f"{request.artifact_name}_metadata",
                output_dir=output_dir,
            )

        return TrainResult(artifact_path=artifact_path)

