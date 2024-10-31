"""VoteM8 consensus scoring methods implementation.

This module provides various consensus scoring algorithms including ARAS, TOPSIS,
VIKOR, and others for combining multiple scoring criteria into unified rankings.
Each method implements a specific approach to consensus scoring with support for
weighted criteria and different aggregation strategies.
"""

from __future__ import annotations

from .aras import aras_consensus
from .binary_pareto import binary_pareto_consensus
from .comet import comet_consensus
from .ecr import ecr_consensus
from .pareto import pareto_consensus
from .promethee import promethee_consensus
from .rbr import rbr_consensus
from .rbv import rbv_consensus
from .topsis import topsis_consensus
from .vikor import vikor_consensus
from .waspas import waspas_consensus
from .wpm import wpm_consensus
from .wsm import wsm_consensus
from .zscore import zscore_consensus

__all__ = [
    "aras_consensus",
    "binary_pareto_consensus",
    "comet_consensus",
    "ecr_consensus",
    "pareto_consensus",
    "promethee_consensus",
    "rbr_consensus",
    "rbv_consensus",
    "topsis_consensus",
    "vikor_consensus",
    "waspas_consensus",
    "wpm_consensus",
    "wsm_consensus",
    "zscore_consensus",
]
