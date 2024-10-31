"""Pareto ranking consensus scoring implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paretoset import paretoset

from votem8.utils.utils import weigh_dataframe

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray


def pareto_consensus(
    data: pd.DataFrame,
    columns: list[str],
    id_column: str = "ID",
    weights: dict[str, float] | NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Calculate Pareto ranking consensus score."""
    scoring_data = data[[id_column, *columns]].copy()
    weighted_data = weigh_dataframe(scoring_data, columns, id_column, weights)
    criteria_columns = weighted_data.columns.drop(id_column, errors="ignore")
    sense = ["max"] * len(criteria_columns)

    scoring_data["Pareto"] = 0
    rank = 1
    remaining_data = weighted_data.copy()

    while not remaining_data.empty:
        mask = paretoset(remaining_data[criteria_columns], sense=sense)
        current_ids = remaining_data.loc[mask, id_column]
        scoring_data.loc[scoring_data[id_column].isin(current_ids), "Pareto"] = rank
        remaining_data = remaining_data.loc[~mask]
        rank += 1

    scoring_data["Pareto"] = len(scoring_data) + 1 - scoring_data["Pareto"]
    return scoring_data[[id_column, "Pareto"]]
