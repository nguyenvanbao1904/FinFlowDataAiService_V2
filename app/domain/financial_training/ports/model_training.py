from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


class ModelTrainerPort(ABC):
    """
    Port: trains a model from preprocessed features/labels.
    """

    @abstractmethod
    def train(
        self,
        features: Any,
        labels: Any,
        dataset_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Returns trained model/artifacts (opaque).
        """

