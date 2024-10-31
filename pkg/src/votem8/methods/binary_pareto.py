"""Binary Pareto optimization consensus scoring implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from paretoset import paretoset

from votem8.utils.utils import weigh_dataframe

if TYPE_CHECKING:
    from numpy.typing import NDArray


def binary_pareto_consensus(
    data: pd.DataFrame,
    columns: list[str],
    id_column: str = "ID",
    weights: dict[str, float] | NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Calculate binary Pareto optimization score."""
    scoring_data = data[[id_column, *columns]].copy()
    weighted_data = weigh_dataframe(scoring_data, columns, id_column, weights)
    criteria_columns = weighted_data.columns.drop(id_column, errors="ignore")
    sense = ["max"] * len(criteria_columns)
    mask = paretoset(weighted_data[criteria_columns], sense=sense)

    weighted_data["BinaryPareto"] = 0
    weighted_data.loc[mask, "BinaryPareto"] = 1
    return weighted_data[[id_column, "BinaryPareto"]]
