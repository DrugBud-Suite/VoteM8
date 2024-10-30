import numpy as np
import pandas as pd


def ECR(
    df: pd.DataFrame,
    columns: list,
    id_column: str = "ID",
    weights=None,
) -> pd.DataFrame:
    """
    Calculates the Exponential Consensus Ranking (ECR) score,
    incorporating weights.

    Parameters
    ----------
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of column names to consider.
    - id_column (str): Name of the ID column (default: "ID").
    - weights: dict or array-like, optional.
        Weights for the criteria.

    Returns
    -------
    - pd.DataFrame: DataFrame with original ID column and new 'ECR' column.
    """
    df = df[[id_column] + columns].copy()
    sigma = 0.05 * len(df)

    # Compute ranks
    ranks = df[columns].rank(method="average", ascending=False)

    # Handle weights
    if weights is not None:
        if isinstance(weights, dict):
            weights_array = np.array([weights[col] for col in columns], dtype=float)
        else:
            weights_array = np.array(weights, dtype=float)
            if len(weights_array) != len(columns):
                raise ValueError("Length of weights must match number of columns")
        # Normalize weights
        weights_array = weights_array / weights_array.sum()
        # Multiply ecr_scores by weights after exponentiation
        ecr_scores = np.exp(-ranks / sigma)
        weighted_ecr_scores = ecr_scores * weights_array
        df["ECR"] = weighted_ecr_scores.sum(axis=1) / sigma
    else:
        # Compute ECR without weights
        ecr_scores = np.exp(-ranks / sigma)
        df["ECR"] = ecr_scores.sum(axis=1) / sigma

    return df[[id_column, "ECR"]]
