from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FinancialRawRow:
    """
    A normalized representation of one row from:
      - financial_training_bank.csv
      - financial_training_non_bank.csv

    The CSV has many columns depending on bank vs non-bank, so we keep the
    row payload as a generic dict.
    """

    symbol: str
    year: int
    quarter: int
    payload: Mapping[str, Any] = field(default_factory=dict)

