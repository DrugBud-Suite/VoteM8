import pandas as pd
import numpy as np
from pymcdm.methods import VIKOR
from pymcdm import weights as w
from pymcdm.helpers import rrankdata


def VIKOR_consensus(df: pd.DataFrame,
                    columns: list,
                    id_column: str = "ID",
                    weights=None) -> pd.DataFrame:
    """
    Calculates the VIKOR consensus score.
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
    vikor = VIKOR()
    types = np.ones(len(columns))
    df['VIKOR'] = rrankdata(vikor(values, weights, types, v=0.5))
    return df[[id_column, 'VIKOR']]