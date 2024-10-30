import pandas as pd
from paretoset import paretoset

from ..utils.utils import weigh_dataframe


def BinaryPareto(
    df: pd.DataFrame, columns: list, id_column: str = "ID", weights=None
) -> pd.DataFrame:
    """
    Assigns a binary value (1 or 0) to solutions based on Pareto optimality.

    Args:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to consider for Pareto optimality
    id_column (str): Name of the ID column (default: "ID")

    Returns
    -------
    pd.DataFrame: DataFrame with original ID column and new 'BinaryPareto' column
    """
    df = df[[id_column] + columns].copy()
    weighted_df = weigh_dataframe(df, columns, id_column, weights)
    criteria_columns = weighted_df.columns.drop(id_column, errors="ignore")
    sense = ["max"] * len(criteria_columns)
    mask = paretoset(weighted_df[criteria_columns], sense=sense)

    # Create 'BinaryPareto' column and assign values
    weighted_df["BinaryPareto"] = 0
    weighted_df.loc[mask, "BinaryPareto"] = 1

    return weighted_df[[id_column, "BinaryPareto"]]
