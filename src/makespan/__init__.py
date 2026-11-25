"""Minimum-makespan orientation.

Orient each edge of a Graph Instance Builder export toward one endpoint so
that the maximum node load is as small as possible.
"""

from makespan.context import SolveContext
from makespan.factory import (
    Algorithm,
    AlgorithmFactory,
    AlgorithmLimitError,
    Pipeline,
    Step,
)
from makespan.io import parse
from makespan.model import Instance, Solution
from makespan.relaxation import SolverUnavailable, solver_available

# Importing the module registers the built-in pipelines.
import makespan.algorithms as algorithms  # noqa: F401


def solve(instance: Instance, algorithm: str, **config) -> Solution:
    """Run one registered algorithm on one instance.

    Args:
        instance: Graph whose edges should be oriented.
        algorithm: Registered name such as ``lp-round`` or ``brute``.
        **config: Extra keyword arguments forwarded to the algorithm builder.

    Returns:
        Solution: Complete orientation and its evaluated makespan.
    """
    return AlgorithmFactory.create(algorithm, **config).solve(instance)


__all__ = [
    "Algorithm",
    "AlgorithmFactory",
    "AlgorithmLimitError",
    "Instance",
    "Pipeline",
    "Solution",
    "SolveContext",
    "SolverUnavailable",
    "Step",
    "parse",
    "solve",
    "solver_available",
]
