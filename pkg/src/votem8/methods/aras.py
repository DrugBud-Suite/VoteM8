import pandas as pd
import numpy as np
from pymcdm.methods import ARAS
from pymcdm import weights as w


def ARAS_consensus(df: pd.DataFrame,
                   columns: list,
                   id_column: str = "ID",
                   weights=None) -> pd.DataFrame:
    """
    Calculates the ARAS (ARAS) consensus score.
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
    aras = ARAS()
    types = np.ones(len(columns))
    df['ARAS'] = aras(values, weights, types)
    return df[[id_column, 'ARAS']]