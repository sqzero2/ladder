"""
Graded Tutor — 因材施教 (differentiated instruction) upgrade for CogWall.

Splits the binary Socratic/Direct gate into a knowledge-graph-driven graded
disclosure level (L0 Socratic / L1 Hint / L2 Partial / L3 Direct).
"""

from .run import run_ladder_tutor_async
from .graph import build_graded_graph, VALID_ARMS
from .gate import compute_disclosure_level, LEVEL_PERMISSION

__all__ = [
    "run_ladder_tutor_async",
    "build_graded_graph",
    "VALID_ARMS",
    "compute_disclosure_level",
    "LEVEL_PERMISSION",
]
