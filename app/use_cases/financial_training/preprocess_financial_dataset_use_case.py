from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from app.domain.financial_training.entities.dataset_spec import (
    FinancialDatasetSpec,
)
from app.domain.financial_training.entities.raw_row import FinancialRawRow
from app.domain.financial_training.ports.data_source import (
    FinancialDataSourcePort,
)
from app.domain.financial_training.ports.preprocessing import (
    FinancialPreprocessorPort,
)


@dataclass(frozen=True)
class PreprocessRequest:
    symbol_filter: Optional[Iterable[str]] = None
    dataset_spec: FinancialDatasetSpec = None


@dataclass(frozen=True)
class PreprocessResult:
    preprocessed: Any


class PreprocessFinancialDatasetUseCase:
    """
    Use case: orchestrates reading raw data and calling the preprocessor.

    Note: no pandas/sklearn logic here.
    """

    def __init__(
        self,
        data_source: FinancialDataSourcePort,
        preprocessor: FinancialPreprocessorPort,
    ) -> None:
        self._data_source = data_source
        self._preprocessor = preprocessor

    def execute(self, request: PreprocessRequest) -> PreprocessResult:
        if request.dataset_spec is None:
            raise ValueError("dataset_spec is required")

        # raw rows are dict-like; mapping to FinancialRawRow is adapter responsibility later.
        raw_rows: Sequence[Mapping] = list(
            self._data_source.load_raw_rows(symbol_filter=request.symbol_filter)
        )

        raw_typed = [
            FinancialRawRow(
                symbol=str(r.get("symbol")),
                year=int(r.get("year")),
                quarter=int(r.get("quarter", 0)),
                payload=r,
            )
            for r in raw_rows
        ]

        preprocessed = self._preprocessor.transform(
            raw_typed, dataset_spec=request.dataset_spec
        )
        return PreprocessResult(preprocessed=preprocessed)

