import numpy as np
import pandas as pd
from pymcdm import weights as w
from pymcdm.methods import PROMETHEE_II


def PROMETHEE_II_consensus(
    df: pd.DataFrame, columns: list, id_column: str = "ID", weights=None
) -> pd.DataFrame:
    """
    Calculates the PROMETHEE II consensus score.
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
    promethee = PROMETHEE_II("usual")
    types = np.ones(len(columns))
    df["PROMETHEE_II"] = promethee(values, weights, types)
    return df[[id_column, "PROMETHEE_II"]]
