from __future__ import annotations

from pathlib import Path

from app.adapters.financial_training.features.pandas_preprocessor import (
    PandasFinancialPreprocessor,
)
from app.adapters.financial_training.ml.sklearn_artifact_store import (
    SklearnArtifactStore,
)
from app.adapters.financial_training.ml.sklearn_trainer import (
    SklearnFinancialTrainer,
)
from app.adapters.financial_training.repositories.csv_financial_data_source import (
    CsvFinancialDataSource,
)
from app.domain.financial_training.entities.dataset_spec import (
    FinancialDatasetSpec,
)
from app.use_cases.financial_training.preprocess_financial_dataset_use_case import (
    PreprocessFinancialDatasetUseCase,
)
from app.use_cases.financial_training.run_full_financial_pipeline_use_case import (
    RunFullFinancialPipelineUseCase,
)
from app.use_cases.financial_training.train_financial_model_use_case import (
    TrainFinancialModelUseCase,
)


def build_full_financial_pipeline(
    bank_csv_path: Path,
    non_bank_csv_path: Path,
    dataset_spec: FinancialDatasetSpec,
) -> RunFullFinancialPipelineUseCase:
    """
    Composition root: wire ports <-> adapters <-> use cases.

    NOTE: adapters are stubs for preprocessing/training right now.
    """

    data_source = CsvFinancialDataSource(
        bank_csv_path=bank_csv_path,
        non_bank_csv_path=non_bank_csv_path,
    )
    preprocessor = PandasFinancialPreprocessor()
    trainer = SklearnFinancialTrainer()
    artifact_store = SklearnArtifactStore()

    preprocess_uc = PreprocessFinancialDatasetUseCase(
        data_source=data_source,
        preprocessor=preprocessor,
    )
    train_uc = TrainFinancialModelUseCase(trainer=trainer, artifact_store=artifact_store)

    return RunFullFinancialPipelineUseCase(
        preprocess_use_case=preprocess_uc,
        train_use_case=train_uc,
    )

