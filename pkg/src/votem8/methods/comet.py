"""This module provides a function to calculate the COMET consensus score."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from pymcdm import weights as w
from pymcdm.methods import COMET, TOPSIS
from pymcdm.methods.comet_tools import MethodExpert


def comet_consensus(
    df: "pd.DataFrame", columns: list, id_column: str = "ID", weights: list[float] | None = None
) -> "pd.DataFrame":
    """Calculates the COMET consensus score."""
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
    types = np.ones(len(columns))
    c_values = COMET.make_cvalues(values)
    expert = MethodExpert(TOPSIS(), weights, types)
    comet = COMET(c_values, expert)
    df["COMET"] = comet(values, weights, types)
    return df[[id_column, "COMET"]]
