"""This module provides a function to calculate the WSM consensus score."""

from typing import TYPE_CHECKING

import numpy as np
from pymcdm import weights as w
from pymcdm.methods import WSM

if TYPE_CHECKING:
    import pandas as pd


def wsm_consensus(
    df: "pd.DataFrame", columns: list, id_column: str = "ID", weights: list | None = None
) -> "pd.DataFrame":
    """Calculates the WSM consensus score."""
    df = df[[id_column, *columns]].copy()
    values = df[columns].to_numpy()
    # Handle weights
    if weights is None:
        # Use equal weights if none are provided
        weights_array = w.equal_weights(values)
    else:
        # Ensure weights are mapped correctly to columns
        weights_array = np.array(weights, dtype=float)
        # Normalize weights to sum to 1
        weights_array = weights_array / weights_array.sum()
    wsm = WSM()
    types = np.ones(len(columns))
    df["WSM"] = wsm(values, weights, types)
    return df[[id_column, "WSM"]]
