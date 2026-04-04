from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence

from app.domain.financial_training.entities.raw_row import FinancialRawRow


class FinancialPreprocessorPort(ABC):
    """
    Port: transforms raw rows into model-ready inputs.

    - Domain/use-cases treat outputs as opaque objects.
    - Concrete adapter decides implementation details (pandas, numpy, etc.).
    """

    @abstractmethod
    def transform(
        self,
        raw_rows: Sequence[FinancialRawRow],
        dataset_spec: Any,
    ) -> Any:
        """
        Returns a framework-agnostic container (e.g., dict of numpy arrays).
        """

