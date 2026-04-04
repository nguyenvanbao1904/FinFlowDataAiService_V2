from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional


class FinancialDataSourcePort(ABC):
    """
    Port: provides raw rows for training.
    Domain layer must not import pandas/sklearn.
    """

    @abstractmethod
    def load_raw_rows(
        self,
        symbol_filter: Optional[Iterable[str]] = None,
    ) -> Iterable[Mapping]:
        """
        Returns an iterable of dict-like rows.
        """

