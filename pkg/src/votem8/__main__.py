"""Main command-line interface of the package."""

import argparse
import logging
import sys
from pathlib import Path

from votem8 import __version__, apply_consensus_scoring, get_available_methods


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the VoteM8 Consensus Scoring CLI.

    Returns
    -------
    - argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VoteM8 Consensus Scoring CLI")

    parser.add_argument("input_file", type=Path, help="Path to input CSV or SDF file")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Consensus methods to apply. Use 'all' for all methods.",
    )
    parser.add_argument("--columns",
                        nargs="+",
                        help="Columns to consider for scoring")
    parser.add_argument("--id-column",
                        default="ID",
                        help="Name of the ID column")
    parser.add_argument(
        "--aggregation",
        choices=["best", "avg"],
        default="best",
        help="Aggregation method: 'best' or 'avg'",
    )
    parser.add_argument("--normalize",
                        action="store_true",
                        help="Enable normalization of results")
    parser.add_argument(
        "--nan-strategy",
        choices=["raise", "drop", "fill_mean", "fill_median", "interpolate"],
        default="raise",
        help="Strategy to handle NaN values",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help=
        "Weights for the columns. Can be a JSON string or a weighting method name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consensus_results.csv"),
        help="Output file path",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the current version of the package.",
    )
    return parser.parse_args()


def validate_methods(methods: str | list[str]) -> list[str]:
    """Validate the provided methods against available methods.

    Parameters
    ----------
    - methods: Union[str, List[str]]
        Methods provided by the user.

    Returns
    -------
    - List[str]
        List of valid methods.
    """
    available_methods = get_available_methods()

    if methods == ["all"]:
        return available_methods

    invalid_methods = [
        method for method in methods if method not in available_methods
    ]
    if invalid_methods:
        error_message = (
            f"Unknown methods: {', '.join(invalid_methods)}. "
            f"Available methods are: {', '.join(available_methods)}")
        raise ValueError(error_message)

    return methods


def parse_weights(weights_arg: str) -> dict | str | None:
    """Parse the weights argument provided by the user.

    Parameters
    ----------
    - weights_arg: str
        Weights provided by the user.

    Returns
    -------
    - Union[dict, str, None]
        Parsed weights as a dict or a weighting method name.
    """
    if weights_arg is None:
        return None

    # Try to parse as JSON
    try:
        import json

        weights = json.loads(weights_arg)
        if isinstance(weights, dict):
            return weights
    except json.JSONDecodeError:
        pass

    # Return as string (weighting method name)
    return weights_arg


def run_cli() -> int:
    """Run the CLI application.

    Returns
    -------
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        args = parse_arguments()

        # Set up logging
        logging.basicConfig(level=getattr(logging, args.log_level))

        # Validate methods
        methods = validate_methods(args.methods)

        # Parse weights
        weights = parse_weights(args.weights)

        # Apply consensus scoring
        result = apply_consensus_scoring(
            data=args.input_file,
            methods=methods,
            columns=args.columns,
            id_column=args.id_column,
            normalize=args.normalize,
            aggregation=args.aggregation,
            nan_strategy=args.nan_strategy,
            weights=weights,
            output=args.output,
        )
        result = apply_consensus_scoring(
            data=args.input_file,
            methods=methods,
            columns=args.columns,
            id_column=args.id_column,
            normalize=args.normalize,
            aggregation=args.aggregation,
            nan_strategy=args.nan_strategy,
            weights=weights,
            output=args.output,
        )

        logging.info("Results saved to %s", result)

    except ValueError:
        return 1  # Error
    except Exception:
        logging.exception("An error occurred during consensus scoring")
        return 2  # Unexpected error
    else:
        return 0  # Success


if __name__ == "__main__":
    sys.exit(run_cli())
