import pandas as pd
import numpy as np


def RbR(df: pd.DataFrame,
        columns: list,
        id_column: str = "ID",
        weights=None) -> pd.DataFrame:
    """
    Calculates the Rank by Rank (RbR) consensus score,
    incorporating weights.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of column names to consider.
    - id_column (str): Name of the ID column (default: "ID").
    - weights: dict or array-like, optional.
        Weights for the criteria.

    Returns:
    - pd.DataFrame: DataFrame with original ID column and new 'RbR' column.
    """
    df = df[[id_column] + columns].copy()

    # Compute ranks
    ranks = df[columns].rank(ascending=True)

    # Handle weights
    if weights is not None:
        if isinstance(weights, dict):
            weights_array = np.array([weights[col] for col in columns],
                                     dtype=float)
        else:
            weights_array = np.array(weights, dtype=float)
            if len(weights_array) != len(columns):
                raise ValueError(
                    "Length of weights must match number of columns")
        # Normalize weights
        weights_array = weights_array / weights_array.sum()
        # Compute weighted average of ranks
        df['RbR'] = (ranks * weights_array).sum(axis=1)
    else:
        # Compute unweighted mean of ranks
        df['RbR'] = ranks.mean(axis=1)

    return df[[id_column, 'RbR']]