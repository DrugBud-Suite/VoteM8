"""
Consensus scoring methods implementations.
"""

from .aras import ARAS_consensus
from .binary_pareto import BinaryPareto
from .comet import COMET_consensus
from .ecr import ECR
from .pareto import Pareto
from .promethee import PROMETHEE_II_consensus
from .rbr import RbR
from .rbv import RbV
from .topsis import TOPSIS_consensus
from .vikor import VIKOR_consensus
from .waspas import WASPAS_consensus
from .wpm import WPM_consensus
from .wsm import WSM_consensus
from .zscore import Zscore

__all__ = [
    "ARAS_consensus",
    "BinaryPareto",
    "COMET_consensus",
    "ECR",
    "Pareto",
    "PROMETHEE_II_consensus",
    "RbR",
    "RbV",
    "TOPSIS_consensus",
    "VIKOR_consensus",
    "WASPAS_consensus",
    "WPM_consensus",
    "WSM_consensus",
    "Zscore",
]
