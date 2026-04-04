from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from app.domain.financial_training.entities.dataset_spec import FinancialDatasetSpec
from app.domain.financial_training.ports.data_source import FinancialDataSourcePort
from app.domain.financial_training.ports.preprocessing import FinancialPreprocessorPort
from app.use_cases.financial_training.preprocess_financial_dataset_use_case import (
    PreprocessFinancialDatasetUseCase,
    PreprocessRequest,
)
from app.use_cases.financial_training.train_financial_model_use_case import (
    TrainFinancialModelUseCase,
    TrainRequest,
)


@dataclass(frozen=True)
class FullPipelineRequest:
    symbol_filter: Optional[Iterable[str]] = None
    dataset_spec: FinancialDatasetSpec = None


class RunFullFinancialPipelineUseCase:
    """
    Use case: preprocess -> train.

    Concrete wiring happens in infrastructure/wiring.py
    """

    def __init__(
        self,
        preprocess_use_case: PreprocessFinancialDatasetUseCase,
        train_use_case: TrainFinancialModelUseCase,
    ) -> None:
        self._preprocess_use_case = preprocess_use_case
        self._train_use_case = train_use_case

    def execute(self, request: FullPipelineRequest, output_dir: Path) -> Any:
        preprocess_res = self._preprocess_use_case.execute(
            PreprocessRequest(
                symbol_filter=request.symbol_filter,
                dataset_spec=request.dataset_spec,
            )
        )

        # The preprocessor output contract is up to you (features/labels).
        features = preprocess_res.preprocessed.get("features")
        labels = preprocess_res.preprocessed.get("labels")
        metadata = preprocess_res.preprocessed.get("metadata", None)

        train_res = self._train_use_case.execute(
            TrainRequest(
                features=features,
                labels=labels,
                dataset_metadata=metadata,
                artifact_name="financial_model",
            ),
            output_dir=output_dir,
        )
        return train_res

