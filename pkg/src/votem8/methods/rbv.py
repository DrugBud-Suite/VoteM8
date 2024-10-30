import pandas as pd


def RbV(df: pd.DataFrame,
        columns: list,
        id_column: str = "ID") -> pd.DataFrame:
    """
    Calculates the Rank by Vote (RbV) consensus score.
    """
    df = df[[id_column] + columns].copy()
    df['RbV'] = (df[columns] > df[columns].quantile(0.95)).sum(axis=1)
    return df[[id_column, 'RbV']]
