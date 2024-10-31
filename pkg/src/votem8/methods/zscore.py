"""This module provides a function to calculate the Z-score consensus score."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def zscore_consensus(
    df: "pd.DataFrame",
    columns: list,
    id_column: str = "ID",
) -> "pd.DataFrame":
    """Calculates the Z-score consensus score."""
    df = df[[id_column, *columns]].copy()
    z_scores = (df[columns] - df[columns].mean()) / df[columns].std()
    df['Zscore'] = z_scores.mean(axis=1)
    return df[[id_column, 'Zscore']]
