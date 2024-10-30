import pandas as pd
from pathlib import Path
from typing import Union
from rdkit import Chem
from rdkit.Chem import PandasTools
import numpy as np
from math import gcd
from functools import reduce
from fractions import Fraction


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV or SDF file.
    
    Args:
        file_path (Union[str, Path]): Path to the input file.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Raises:
        ValueError: If the file format is not supported.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() == '.sdf':
        return load_sdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_sdf(file_path: Path) -> pd.DataFrame:
    """
    Load data from an SDF file.
    
    Args:
        file_path (Path): Path to the SDF file.
    
    Returns:
        pd.DataFrame: Loaded data with RDKit molecules.
    """
    df = PandasTools.LoadSDF(str(file_path),
                             molColName='ROMol',
                             includeFingerprints=False)

    # Convert RDKit molecules to SMILES strings
    df['SMILES'] = df['ROMol'].apply(lambda x: Chem.MolToSmiles(x)
                                     if x is not None else None)

    # Drop the ROMol column as it's not easily serializable
    df = df.drop('ROMol', axis=1)

    return df


def weigh_dataframe(df: pd.DataFrame,
                    columns: list,
                    id_column: str = "ID",
                    weights=None) -> pd.DataFrame:
    # Ensure id_column is in the dataframe
    if id_column not in df.columns:
        raise ValueError(
            f"The specified id_column '{id_column}' is not present in the dataframe."
        )

    # Handle weights
    if weights is not None:
        if isinstance(weights, dict):
            # Map weights to columns in order
            weights_list = [weights[col] for col in columns]
        else:
            # Assume weights is an array-like object
            weights_list = list(weights)
            if len(weights_list) != len(columns):
                raise ValueError(
                    "Length of weights must match number of columns")
        # Normalize weights
        weights_array = np.array(weights_list, dtype=float)
        weights_array = weights_array / weights_array.sum()

        # Convert weights to fractions with common denominator
        fractions = [Fraction(w).limit_denominator() for w in weights_array]
        denominators = [f.denominator for f in fractions]
        lcm_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)

        # Scale fractions to have common denominator and get integer weights
        integer_weights = [
            int(fraction * lcm_denominator) for fraction in fractions
        ]

        # Now replicate columns according to integer weights
        replicated_columns = []
        for col, weight in zip(columns, integer_weights):
            replicated_columns.extend([col] * weight)
        # Create new DataFrame with replicated columns
        df_replicated = df[[id_column]].copy()
        for idx, col in enumerate(replicated_columns):
            df_replicated[f"{col}_{idx}"] = df[col]
    else:
        # If no weights provided, use original columns
        df_replicated = df[columns + [id_column]].copy()

    # Ensure id_column is the first column
    cols = df_replicated.columns.tolist()
    cols.remove(id_column)
    df_replicated = df_replicated[[id_column] + cols]

    return df_replicated
