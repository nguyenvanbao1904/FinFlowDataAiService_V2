from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from app.domain.financial_training.ports.data_source import FinancialDataSourcePort


@dataclass(frozen=True)
class CsvFinancialDataSource(FinancialDataSourcePort):
    """
    Adapter: loads raw rows from the exported CSVs.

    This is the bridge between your DB-export artifacts and the training pipeline.
    """

    bank_csv_path: Path
    non_bank_csv_path: Path

    def load_raw_rows(
        self,
        symbol_filter: Optional[Iterable[str]] = None,
    ) -> Sequence[Mapping]:
        # This adapter currently reads both csv files and concatenates.
        # You can later add batching / memory control as needed.
        bank_df = pd.read_csv(self.bank_csv_path)
        non_bank_df = pd.read_csv(self.non_bank_csv_path)
        df = pd.concat([bank_df, non_bank_df], ignore_index=True)

        if symbol_filter is not None:
            symbol_set = set(symbol_filter)
            df = df[df["symbol"].isin(symbol_set)]

        # Return rows as dict-like mappings.
        return df.to_dict(orient="records")

