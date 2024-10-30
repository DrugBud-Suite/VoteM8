import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pymcdm import weights as w

from votem8.methods.aras import ARAS_consensus
from votem8.methods.binary_pareto import BinaryPareto
from votem8.methods.ecr import ECR
from votem8.methods.pareto import Pareto
from votem8.methods.rbr import RbR
from votem8.methods.rbv import RbV
from votem8.methods.topsis import TOPSIS_consensus
from votem8.methods.vikor import VIKOR_consensus
from votem8.methods.waspas import WASPAS_consensus
from votem8.methods.wpm import WPM_consensus
from votem8.methods.wsm import WSM_consensus
from votem8.methods.zscore import Zscore
from votem8.utils.utils import load_data

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s")

# Dictionary of available consensus methods
_METHODS = {
    "ECR": ECR,
    "RbR": RbR,
    "RbV": RbV,
    "Zscore": Zscore,
    # 'COMET': COMET_consensus,
    "TOPSIS": TOPSIS_consensus,
    "WASPAS": WASPAS_consensus,
    "VIKOR": VIKOR_consensus,
    "ARAS": ARAS_consensus,
    # 'PROMETHEE_II': PROMETHEE_II_consensus,
    "WPM": WPM_consensus,
    "WSM": WSM_consensus,
    "BinaryPareto": BinaryPareto,
    "Pareto": Pareto,
}


def add_consensus_method(name: str, method: Callable):
    """Add a new consensus method to the available methods."""
    _METHODS[name] = method
    logging.info(f"Added {name} to consensus methods.")


def get_available_methods() -> list[str]:
    """Return a list of available consensus methods."""
    return list(_METHODS.keys())


def load_and_validate_data(
    data: str | Path | pd.DataFrame, id_column: str, columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load data and validate columns.

    Parameters
    ----------
    - data: Union[str, Path, pd.DataFrame]
    The data to be processed. Can be a file path or a DataFrame.
    - id_column: str
    The column that contains the unique identifiers.
    - columns: List[str], optional
    The columns to be used in scoring.

    Returns
    -------
    - data: pd.DataFrame
    The loaded and validated DataFrame.
    - valid_columns: List[str]
    The list of valid numeric columns for scoring.
    """
    # Load data if it's a file path
    if isinstance(data, (str, Path)):
        logging.debug(f"Loading data from {data}")
        data = load_data(data)

    if data.empty:
        logging.error("Input data is empty")
        raise ValueError("Input data is empty")

    # Ensure id_column is present
    if id_column not in data.columns:
        logging.error(f"ID column '{id_column}' not found in data")
        raise ValueError(f"ID column '{id_column}' not found in the data")

    # Filter columns if specified
    if columns:
        valid_columns = [
            col
            for col in columns
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
        ]
        if not valid_columns:
            logging.error("No valid numeric columns found in specified columns")
            raise ValueError("None of the specified columns were found in the data or were numeric")
        data = data[[id_column] + valid_columns]
    else:
        valid_columns = [
            col
            for col in data.columns
            if col != id_column and pd.api.types.is_numeric_dtype(data[col])
        ]

    if not valid_columns:
        logging.error("No valid numeric columns found for scoring")
        raise ValueError("No valid numeric columns found for scoring")

    logging.info(f"Found {len(valid_columns)} valid columns for analysis")
    return data, valid_columns


def handle_nan_values(
    data: pd.DataFrame, valid_columns: list[str], nan_strategy: str
) -> pd.DataFrame:
    """
    Handle NaN values in the data based on the specified strategy.

    Parameters
    ----------
    - data: pd.DataFrame
    The DataFrame to process.
    - valid_columns: List[str]
    The columns to check for NaN values.
    - nan_strategy: str
    Strategy to handle NaN values ('raise', 'drop', 'fill_mean', 'fill_median', 'interpolate').

    Returns
    -------
    - pd.DataFrame
    The DataFrame after handling NaN values.
    """
    if nan_strategy == "raise":
        if data[valid_columns].isna().any().any():
            raise ValueError("Input data contains NaN values in scoring columns")
    elif nan_strategy == "drop":
        data = data.dropna(subset=valid_columns)
    elif nan_strategy == "fill_mean":
        data[valid_columns] = data[valid_columns].fillna(data[valid_columns].mean())
    elif nan_strategy == "fill_median":
        data[valid_columns] = data[valid_columns].fillna(data[valid_columns].median())
    elif nan_strategy == "interpolate":
        data[valid_columns] = data[valid_columns].interpolate()
    else:
        logging.error(f"Invalid nan_strategy: {nan_strategy}")
        raise ValueError(f"Invalid nan_strategy: {nan_strategy}")

    return data


def get_weights(
    data: pd.DataFrame, valid_columns: list[str], weights: dict | str | None
) -> np.ndarray | None:
    """
    Compute the weights for the scoring columns.

    Parameters
    ----------
    - data: pd.DataFrame
    The DataFrame containing the data.
    - valid_columns: List[str]
    The columns to compute weights for.
    - weights: Union[dict, str, None]
    Weights for the columns. Can be a dict or a string specifying a weighting method.

    Returns
    -------
    - Optional[np.ndarray]
    The array of weights, or None if no weights are specified.
    """
    weights_array = None
    if weights is not None:
        if isinstance(weights, dict):
            # Ensure all keys in weights are in valid_columns
            if not all(col in valid_columns for col in weights.keys()):
                logging.error("Weights dict contains invalid columns")
                raise ValueError("Weights dict contains invalid columns")
            # Map weights to valid_columns in order
            weights_array = np.array([weights[col] for col in valid_columns], dtype=float)
            # Normalize weights to sum to 1
            weights_array = weights_array / weights_array.sum()
        elif isinstance(weights, str):
            # Use weighting method from pymcdm.weights
            weighting_methods = {
                "equal": w.equal_weights,
                "entropy": w.entropy_weights,
                "standard_deviation": w.standard_deviation_weights,
                "gini": w.gini_weights,
                "variance": w.variance_weights,
            }
            if weights.lower() in weighting_methods:
                weighting_function = weighting_methods[weights.lower()]
                # pymcdm expects a numpy array
                values = data[valid_columns].to_numpy()
                weights_array = weighting_function(values)
            else:
                logging.error(f"Invalid weighting method: {weights}")
                raise ValueError(f"Invalid weighting method: {weights}")
        else:
            logging.error("Weights must be a dict or a string")
            raise ValueError("Weights must be a dict or a string")
    if weights is None:
        weights_array = np.ones(len(valid_columns))
    logging.info(f"Using weights: {weights_array}")
    return weights_array


def select_methods(methods: str | list[str], available_methods: dict) -> list[Callable]:
    """
    Select consensus methods to apply.

    Parameters
    ----------
    - methods: Union[str, List[str]]
    The consensus methods to apply. Can be 'all' or a list of method names.
    - available_methods: dict
    Dictionary of available methods.

    Returns
    -------
    - List[Callable]
    List of selected method functions.
    """
    if methods == "all":
        selected_methods = list(available_methods.values())
    elif isinstance(methods, str):
        if methods not in available_methods:
            logging.error(f"Invalid method: {methods}")
            raise ValueError(f"Invalid method: {methods}")
        selected_methods = [available_methods[methods]]
    else:
        selected_methods = []
        for method in methods:
            if method not in available_methods:
                logging.error(f"Invalid method: {method}")
                raise ValueError(f"Invalid method: {method}")
            selected_methods.append(available_methods[method])

    return selected_methods


def apply_selected_methods(
    data: pd.DataFrame,
    valid_columns: list[str],
    id_column: str,
    selected_methods: list[Callable],
    weights_array: np.ndarray | None,
    normalize: bool,
    aggregation: str,
) -> list[pd.DataFrame]:
    """
    Apply the selected consensus methods to the data.

    Parameters
    ----------
    - data: pd.DataFrame
    The DataFrame containing the data.
    - valid_columns: List[str]
    The columns to be used in scoring.
    - id_column: str
    The column that contains the unique identifiers.
    - selected_methods: List[Callable]
    List of selected method functions.
    - weights_array: Optional[np.ndarray]
    The array of weights, or None if no weights are specified.
    - normalize: bool
    Whether to normalize the scores.
    - aggregation: str
    How to aggregate results ('best' or 'avg').

    Returns
    -------
    - List[pd.DataFrame]
    List of DataFrames with the results from each method.
    """
    results = []
    for method in selected_methods:
        start_time = time.time()
        # Ensure the method accepts weights
        try:
            result = method(data, valid_columns, id_column, weights=weights_array)
        except TypeError:
            # If the method does not accept weights, call without weights
            result = method(data, valid_columns, id_column)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Time taken for {method.__name__}: {execution_time:.4f} seconds")

        score_column = [col for col in result.columns if col != id_column][-1]
        # Aggregate results
        if aggregation == "best":
            result = (
                result.sort_values(score_column, ascending=False)
                .groupby(id_column)
                .first(numeric_only=True)
                .reset_index()
            )
        elif aggregation == "avg":
            result = result.groupby(id_column).mean(numeric_only=True).reset_index()
        else:
            logging.error("aggregation must be 'best' or 'avg'")
            raise ValueError("aggregation must be 'best' or 'avg'")

        if normalize:
            min_score = result[score_column].min()
            max_score = result[score_column].max()
            if max_score != min_score:
                result[score_column] = (result[score_column] - min_score) / (max_score - min_score)
            else:
                result[score_column] = 0  # or any appropriate value

        result = result.sort_values(by=score_column, ascending=False)
        result = result.reset_index(drop=True)
        results.append(result)

    return results


def combine_results(results: list[pd.DataFrame], id_column: str) -> pd.DataFrame:
    """
    Combine the results from different methods.

    Parameters
    ----------
    - results: List[pd.DataFrame]
    List of DataFrames with the results from each method.
    - id_column: str
    The column that contains the unique identifiers.

    Returns
    -------
    - pd.DataFrame
    The combined DataFrame with results from all methods.
    """
    final_result = results[0]
    for result in results[1:]:
        final_result = pd.merge(final_result, result, on=id_column, how="outer")

    # Sort the final result
    score_columns = [col for col in final_result.columns if col != id_column]
    final_result = final_result.sort_values(by=score_columns, ascending=False)
    final_result = final_result.reset_index(drop=True)

    return final_result


def save_results(final_result: pd.DataFrame, output: str | Path) -> Path:
    """
    Save the final results to the specified output file.

    Parameters
    ----------
    - final_result: pd.DataFrame
    The DataFrame containing the final results.
    - output: Union[str, Path]
    File path to save the results.

    Returns
    -------
    - Path
    The output file path.
    """
    output_path = Path(output)
    final_result.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
    return output_path


def apply_consensus_scoring(
    data: str | Path | pd.DataFrame,
    methods: str | list[str] = "all",
    columns: list[str] | None = None,
    id_column: str = "ID",
    normalize: bool = True,
    aggregation: str = "best",
    nan_strategy: str = "raise",
    weights: dict | str | None = None,
    output: str | Path | None = None,
) -> pd.DataFrame | Path:
    """
    Apply consensus scoring methods to the provided data.

    Parameters
    ----------
    - data: Union[str, Path, pd.DataFrame]
    The data to be processed. Can be a file path or a DataFrame.
    - methods: Union[str, List[str]], default 'all'
    The consensus methods to apply. Can be 'all' or a list of method names.
    - columns: List[str], optional
    The columns to be used in scoring.
    - id_column: str, default 'ID'
    The column that contains the unique identifiers.
    - normalize: bool, default True
    Whether to normalize the scores.
    - aggregation: str, default 'best'
    How to aggregate results ('best' or 'avg').
    - nan_strategy: str, default 'raise'
    Strategy to handle NaN values ('raise', 'drop', 'fill_mean', 'fill_median', 'interpolate').
    - weights: Union[dict, str, None], optional
    Weights for the columns. Can be a dict or a string specifying a weighting method.
    - output: Union[str, Path], optional
    File path to save the results.

    Returns
    -------
    - Union[pd.DataFrame, Path]
    The final scoring results as a DataFrame or the output file path.
    """
    logging.info("Starting consensus scoring process")
    logging.debug(
        f"Input parameters: methods={methods}, normalize={normalize}, "
        f"aggregation={aggregation}, nan_strategy={nan_strategy}"
    )

    # Load data and validate columns
    data, valid_columns = load_and_validate_data(data, id_column, columns)

    # Handle NaN values
    logging.debug(f"Handling NaN values using strategy: {nan_strategy}")
    data = handle_nan_values(data, valid_columns, nan_strategy)

    # Handle weights
    logging.debug("Computing weights")
    weights_array = get_weights(data, valid_columns, weights)

    # Select methods
    logging.info("Applying consensus scoring methods")
    selected_methods = select_methods(methods, _METHODS)

    # Apply methods
    results = apply_selected_methods(
        data, valid_columns, id_column, selected_methods, weights_array, normalize, aggregation
    )

    # Combine results
    logging.debug("Combining results")
    final_result = combine_results(results, id_column)

    # Save or return results
    logging.info("Consensus scoring completed successfully")
    if output:
        return save_results(final_result, output)
    return final_result


def describe_method(method_name: str) -> str:
    """Return a description of the specified consensus method."""
    if method_name not in _METHODS:
        raise ValueError(f"Unknown method: {method_name}")
    return _METHODS[method_name].__doc__ or "No description available."
