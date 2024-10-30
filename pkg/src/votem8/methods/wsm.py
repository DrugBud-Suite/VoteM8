import pandas as pd
import numpy as np
from pymcdm.methods import WSM
from pymcdm import weights as w


def WSM_consensus(df: pd.DataFrame,
                  columns: list,
                  id_column: str = "ID",
                  weights=None) -> pd.DataFrame:
    """
    Calculates the WSM consensus score.
    """
    df = df[[id_column] + columns].copy()
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
    df['WSM'] = wsm(values, weights, types)
    return df[[id_column, 'WSM']]
