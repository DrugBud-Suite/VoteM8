"""Core VoteM8 functionality for consensus scoring implementation.

This module provides the main interface for applying consensus scoring methods
to input data, including data validation, method selection, and result aggregation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pymcdm import weights as w

if TYPE_CHECKING:
    from collections.abc import Callable

from votem8.methods.aras import aras_consensus
from votem8.methods.binary_pareto import binary_pareto_consensus
from votem8.methods.ecr import ecr_consensus
from votem8.methods.pareto import pareto_consensus
from votem8.methods.rbr import rbr_consensus
from votem8.methods.rbv import rbv_consensus
from votem8.methods.topsis import topsis_consensus
from votem8.methods.vikor import vikor_consensus
from votem8.methods.waspas import waspas_consensus
from votem8.methods.wpm import wpm_consensus
from votem8.methods.wsm import wsm_consensus
from votem8.methods.zscore import zscore_consensus
from votem8.utils.utils import load_data

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s")

# Update method dictionary with new lowercase names
_METHODS = {
    'ecr': ecr_consensus,
    'rbr': rbr_consensus,
    'rbv': rbv_consensus,
    'zscore': zscore_consensus,
    'topsis': topsis_consensus,
    'waspas': waspas_consensus,
    'vikor': vikor_consensus,
    'aras': aras_consensus,
    'wpm': wpm_consensus,
    'wsm': wsm_consensus,
    'binary_pareto': binary_pareto_consensus,
    'pareto': pareto_consensus
}

# Dictionary of available consensus methods

def add_consensus_method(name: str, method: Callable):
    """Add a new consensus method to the available methods."""
    _METHODS[name] = method
    logging.info("Added %s to consensus methods.", name)


def get_available_methods() -> list[str]:
    """Return a list of available consensus methods."""
    return list(_METHODS.keys())


def load_and_validate_data(
        data: str | Path | pd.DataFrame,
        id_column: str,
        columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load data and validate columns.

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
    if isinstance(data, str | Path):
        logging.debug("Loading data from %s", data)
        data = load_data(data)

    if data.empty:
        logging.error("Input data is empty")
        msg = "Input data is empty"
        raise ValueError(msg)

    # Ensure id_column is present
    if id_column not in data.columns:
        logging.error("ID column '%s' not found in data", id_column)
        msg = f"ID column '{id_column}' not found in the data"
        raise ValueError(msg)

    # Filter columns if specified
    if columns:
        valid_columns = [
            col for col in columns
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
        ]
        if not valid_columns:
            logging.error(
                "No valid numeric columns found in specified columns")
            msg = "None of the specified columns were found in the data or were numeric"
            raise ValueError(msg)
        data = data[[id_column, *valid_columns]]
    else:
        valid_columns = [
            col for col in data.columns
            if col != id_column and pd.api.types.is_numeric_dtype(data[col])
        ]

    if not valid_columns:
        logging.error("No valid numeric columns found for scoring")
        msg = "No valid numeric columns found for scoring"
        raise ValueError(msg)

    logging.info("Found %d valid columns for analysis", len(valid_columns))
    return data, valid_columns


def handle_nan_values(data: pd.DataFrame, valid_columns: list[str],
                      nan_strategy: str) -> pd.DataFrame:
    """Handle NaN values in the data based on the specified strategy.

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
    if nan_strategy == 'raise':
        if data[valid_columns].isna().any().any():
            msg = "Input data contains NaN values in scoring columns"
            raise ValueError(msg)
    elif nan_strategy == 'drop':
        data = data.dropna(subset=valid_columns)
    elif nan_strategy == "fill_mean":
        data[valid_columns] = data[valid_columns].fillna(
            data[valid_columns].mean())
    elif nan_strategy == "fill_median":
        data[valid_columns] = data[valid_columns].fillna(
            data[valid_columns].median())
    elif nan_strategy == "interpolate":
        data[valid_columns] = data[valid_columns].interpolate()
    else:
        logging.error("Invalid nan_strategy: %s", nan_strategy)
        msg = f"Invalid nan_strategy: {nan_strategy}"
        raise ValueError(msg)

    return data


def get_weights(data: pd.DataFrame, valid_columns: list[str],
                weights: dict | str | None) -> np.ndarray | None:
    """Compute the weights for the scoring columns.

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
            if not all(col in valid_columns for col in weights):
                logging.error("Weights dict contains invalid columns")
                msg = "Weights dict contains invalid columns"
                raise ValueError(msg)
            # Map weights to valid_columns in order
            weights_array = np.array([weights[col] for col in valid_columns],
                                     dtype=float)
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
                logging.error("Invalid weighting method: %s", weights)
                msg = f"Invalid weighting method: {weights}"
                raise ValueError(msg)
        else:
            logging.error("Weights must be a dict or a string")
            msg = "Weights must be a dict or a string"
            raise ValueError(msg)
    if weights is None:
        weights_array = np.ones(len(valid_columns))
    logging.info("Using weights: %s", weights_array)
    return weights_array


def select_methods(methods: str | list[str],
                   available_methods: dict) -> list[Callable]:
    """Select consensus methods to apply.

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
    if methods == 'all':
        selected_methods = list(available_methods.values())
    elif isinstance(methods, str):
        if methods not in available_methods:
            logging.error("Invalid method: %s", methods)
            msg = f"Invalid method: {methods}"
            raise ValueError(msg)
        selected_methods = [available_methods[methods]]
    else:
        selected_methods = []
        for method in methods:
            if method not in available_methods:
                logging.error("Invalid method: %s", method)
                msg = f"Invalid method: {method}"
                raise ValueError(msg)
            selected_methods.append(available_methods[method])

    return selected_methods


def apply_selected_methods(
        data: pd.DataFrame,
        valid_columns: list[str],
        id_column: str,
        selected_methods: list[Callable],
        weights_array: np.ndarray | None,
        normalize: bool = True,  # noqa: FBT001, FBT002
        aggregation: str = 'best') -> list[pd.DataFrame]:
    """Apply the selected consensus methods to the data.

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
            result = method(data,
                            valid_columns,
                            id_column,
                            weights=weights_array)
        except TypeError:
            # If the method does not accept weights, call without weights
            result = method(data, valid_columns, id_column)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info("Time taken for %s: %.4f seconds", method.__name__,
                     execution_time)

        score_column = [col for col in result.columns if col != id_column][-1]
        # Aggregate results
        if aggregation == "best":
            result = (result.sort_values(
                score_column, ascending=False).groupby(id_column).first(
                    numeric_only=True).reset_index())
        elif aggregation == "avg":
            result = result.groupby(id_column).mean(
                numeric_only=True).reset_index()
        else:
            logging.error("aggregation must be 'best' or 'avg'")
            msg = "aggregation must be 'best' or 'avg'"
            raise ValueError(msg)

        if normalize:
            min_score = result[score_column].min()
            max_score = result[score_column].max()
            if max_score != min_score:
                result[score_column] = (result[score_column] -
                                        min_score) / (max_score - min_score)
            else:
                result[score_column] = 0  # or any appropriate value

        result = result.sort_values(by=score_column, ascending=False)
        result = result.reset_index(drop=True)
        results.append(result)

    return results


def combine_results(results: list[pd.DataFrame],
                    id_column: str) -> pd.DataFrame:
    """Combine the results from different methods.

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
        final_result = final_result.merge(result, on=id_column, how='outer')

    # Sort the final result
    score_columns = [col for col in final_result.columns if col != id_column]
    final_result = final_result.sort_values(by=score_columns, ascending=False)
    return final_result.reset_index(drop=True)


def save_results(final_result: pd.DataFrame, output: str | Path) -> Path:
    """Save the final results to the specified output file.

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
    logging.info("Results saved to %s", output_path)
    return output_path


def apply_consensus_scoring(
        data: str | Path | pd.DataFrame,
        methods: str | list[str] = 'all',
        columns: list[str] | None = None,
        id_column: str = 'ID',
        aggregation: str = 'best',
        nan_strategy: str = 'raise',
        weights: dict | str | None = None,
        output: str | Path | None = None,
        normalize: bool = True) -> pd.DataFrame | Path:  # noqa: FBT001, FBT002
    """Apply consensus scoring methods to the provided data.

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
        "Input parameters: methods=%s, normalize=%s, aggregation=%s, nan_strategy=%s",
        methods, normalize, aggregation, nan_strategy)

    # Load data and validate columns
    data, valid_columns = load_and_validate_data(data, id_column, columns)

    # Handle NaN values
    logging.debug("Handling NaN values using strategy: %s", nan_strategy)
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
    return final_result


def describe_method(method_name: str) -> str:
    """Return a description of the specified consensus method."""
    if method_name not in _METHODS:
        msg = f"Unknown method: {method_name}"
        raise ValueError(msg)
    return _METHODS[method_name].__doc__ or "No description available."
