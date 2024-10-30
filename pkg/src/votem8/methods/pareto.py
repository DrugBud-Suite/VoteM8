import pandas as pd
from paretoset import paretoset

from ..utils.utils import weigh_dataframe


def Pareto(df: pd.DataFrame, columns: list, id_column: str = "ID", weights=None) -> pd.DataFrame:
    """
    Assigns Pareto ranks to solutions based on iterative Pareto optimality,
    incorporating weights by duplicating criteria columns according to integer weights.

    Parameters
    ----------
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of column names to consider for Pareto optimality.
    - id_column (str): Name of the ID column (default: "ID").
    - weights: dict or array-like, optional.
        Weights for the criteria.

    Returns
    -------
    - pd.DataFrame: DataFrame with original ID column and new 'Pareto' column.
    """
    # Copy necessary columns
    df = df[[id_column] + columns].copy()

    # Apply the weighting function
    weighted_df = weigh_dataframe(df, columns, id_column, weights)

    # Identify criteria columns
    criteria_columns = weighted_df.columns.drop(id_column, errors="ignore")
    sense = ["max"] * len(criteria_columns)  # Assuming all criteria are to be maximized

    # Initialize Pareto rank column
    df["Pareto"] = 0

    rank = 1
    remaining_weighted_df = weighted_df.copy()
    remaining_df = df.copy()

    while not remaining_weighted_df.empty:
        # Perform Pareto analysis on the weighted data
        mask = paretoset(remaining_weighted_df[criteria_columns], sense=sense)

        # Get IDs of current Pareto optimal solutions
        current_ids = remaining_weighted_df.loc[mask, id_column]

        # Assign current rank to these IDs in the original DataFrame
        df.loc[df[id_column].isin(current_ids), "Pareto"] = rank

        # Remove Pareto optimal solutions from remaining data
        remaining_weighted_df = remaining_weighted_df.loc[~mask]
        remaining_df = remaining_df.loc[~remaining_df[id_column].isin(current_ids)]

        rank += 1

    # Reverse the ranks so that the best rank has the highest number
    df["Pareto"] = len(df) + 1 - df["Pareto"]

    return df[[id_column, "Pareto"]]
