"""Exponential Consensus Ranking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


def ecr_consensus(
    data: pd.DataFrame,
    columns: list[str],
    id_column: str = "ID",
    weights: dict[str, float] | NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Calculate the ECR consensus score."""
    scoring_data = data[[id_column, *columns]].copy()
    sigma = 0.05 * len(scoring_data)
    ranks = scoring_data[columns].rank(method="average", ascending=False)

    if weights is not None:
        if isinstance(weights, dict):
            weights_array = np.array([weights[col] for col in columns], dtype=float)
        else:
            weights_array = np.array(weights, dtype=float)
        weights_array = weights_array / weights_array.sum()
        ecr_scores = np.exp(-ranks / sigma)
        weighted_ecr_scores = ecr_scores * weights_array
        scoring_data["ECR"] = weighted_ecr_scores.sum(axis=1) / sigma
    else:
        ecr_scores = np.exp(-ranks / sigma)
        scoring_data["ECR"] = ecr_scores.sum(axis=1) / sigma

    return scoring_data[[id_column, "ECR"]]
