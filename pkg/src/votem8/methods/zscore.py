import pandas as pd


def Zscore(
    df: pd.DataFrame,
    columns: list,
    id_column: str = "ID",
) -> pd.DataFrame:
    """
    Calculates the Z-score consensus score.
    """
    df = df[[id_column] + columns].copy()
    z_scores = (df[columns] - df[columns].mean()) / df[columns].std()
    df['Zscore'] = z_scores.mean(axis=1)
    return df[[id_column, 'Zscore']]
