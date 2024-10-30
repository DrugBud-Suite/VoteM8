"""VoteM8: VoteM8 - A Python package for consensus scoring.

VoteM8 is a Python library for consensus scoring and ranking of multi-criteria
data. It implements multiple scoring methods including ECR, RbR, TOPSIS, VIKOR
and others to combine different evaluation metrics into unified rankings. The
library supports customizable weights, handles missing values, and provides both
normalization and aggregation options. It includes a command-line interface and
can process both CSV and SDF file formats, making it particularly useful for
processing molecular docking scores and other scientific data requiring
consensus analysis.

© 2024 DrugM8
SPDX-License-Identifier: MIT
"""

from votem8 import data
from votem8.votem8 import (
    add_consensus_method,
    apply_consensus_scoring,
    describe_method,
    get_available_methods,
)

__all__ = ["data", "__version_details__", "__version__"]

__version_details__: dict[str, str] = {"version": "0.0.0"}
"""Details of the currently installed version of the package,
including version number, date, branch, and commit hash."""

__version__: str = __version_details__["version"]
"""Version number of the currently installed package."""

__author__ = "Antoine Lacour"
__email__ = "alacournola+votem8@gmail.com"

__all__ = [
    "apply_consensus_scoring",
    "get_available_methods",
    "describe_method",
    "add_consensus_method",
]
