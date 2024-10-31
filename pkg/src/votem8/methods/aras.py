"""ARAS (Additive Ratio Assessment) consensus scoring implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymcdm import weights as w
from pymcdm.methods import ARAS

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


def aras_consensus(
    data: pd.DataFrame,
    columns: list[str],
    id_column: str = "ID",
    weights: dict[str, float] | NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Calculate the ARAS consensus score."""
    scoring_data = data[[id_column, *columns]].copy()
    values = scoring_data[columns].to_numpy()

    if weights is None:
        weights_array = w.equal_weights(values)
    else:
        if isinstance(weights, dict):
            weights_array = np.array([weights[col] for col in columns], dtype=float)
        else:
            weights_array = np.array(weights, dtype=float)
        weights_array = weights_array / weights_array.sum()

    aras = ARAS()
    types = np.ones(len(columns))
    scoring_data["ARAS"] = aras(values, weights_array, types)
    return scoring_data[[id_column, "ARAS"]]
