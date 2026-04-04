from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FinancialDatasetSpec:
    """
    Defines what columns are treated as:
      - features (X)
      - labels/targets (y)
      - ids / metadata (period keys)

    This is intentionally framework-free and contains no pandas/sklearn types.
    """

    feature_columns: List[str]
    label_columns: List[str]
    id_columns: List[str]
    optional_numeric_fill: Optional[float] = None
    extra: Dict[str, str] = None

