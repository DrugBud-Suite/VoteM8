"""This module contains methods for calculating the Rank by Vote (RbV) consensus score."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def rbv_consensus(df: "pd.DataFrame", columns: list, id_column: str = "ID") -> "pd.DataFrame":
    """Calculates the Rank by Vote (RbV) consensus score."""
    df = df[[id_column, *columns]].copy()
    df["RbV"] = (df[columns] > df[columns].quantile(0.95)).sum(axis=1)
    return df[[id_column, "RbV"]]
