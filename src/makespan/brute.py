"""Branch and bound over orientations of the unlocked edges.

The search prunes a partial orientation when any node load is already at
least as large as the incumbent makespan. Later edges cannot decrease a load.
"""

from __future__ import annotations

from makespan.context import SolveContext
from makespan.factory import AlgorithmLimitError, Step
from makespan.greedy import GreedyLightestStep
from makespan.model import ABS_TOL, Edge


class BruteForceStep(Step):
    """Exact search of unlocked edges, refused past ``max_edges``.

    Args:
        max_edges: Maximum number of unlocked edges this step will enumerate.
    """

    def __init__(self, max_edges: int = 24):
        """Remember the free-edge limit. A negative limit is rejected.

        Args:
            max_edges: Maximum number of unlocked edges this step will enumerate.
        """
        if max_edges < 0:
            raise ValueError("max_edges must be non-negative")
        self.max_edges = max_edges

    def run(self, ctx: SolveContext) -> None:
        """Enumerate orientations of the unlocked edges and keep the best.

        The incumbent is seeded with lightest-greedy when the context does not
        already have one, so branches can be pruned immediately. Edges are
        tried heavier-weight first, and each edge tries its lighter endpoint
        first. A partial load that already meets the incumbent makespan is
        abandoned, because later edges cannot reduce a load.

        Args:
            ctx: Solve state whose unlocked edges are searched.
        """
        free = ctx.free_edges()
        if len(free) > self.max_edges:
            raise AlgorithmLimitError(
                f"brute force supports at most {self.max_edges} free edges, "
                f"this instance has {len(free)}"
            )
        if ctx.best_so_far_assignment is None:
            GreedyLightestStep().run(ctx)
        ordered = sorted(
            free,
            key=lambda edge: (-max(edge.w_u, edge.w_v), edge.id),
        )
        load = ctx.base_load()
        if not ordered:
            return
        if any(value >= ctx.best_so_far_makespan - ABS_TOL for value in load.values()):
            return
        choice: dict[int, int] = {}

        def search(index: int) -> None:
            """Assign ``ordered[index]`` and recurse. ``index`` past the end records a solution.

            Args:
                index: Position in ``ordered`` of the next edge to orient.
            """
            if index == len(ordered):
                ctx.consider(ctx.assignment_from(choice))
                return
            edge = ordered[index]
            for head in _head_order(edge):
                weight = edge.weight_toward(head)
                load[head] += weight
                if load[head] < ctx.best_so_far_makespan - ABS_TOL:
                    choice[edge.id] = head
                    search(index + 1)
                load[head] -= weight

        search(0)


def _head_order(edge: Edge) -> tuple[int, int]:
    """Try the lighter endpoint before the heavier one. Ties try ``u`` first.

    Args:
        edge: Edge whose two endpoints are ordered for branching.

    Returns:
        tuple[int, int]: Endpoints in trial order, lighter first.
    """
    if edge.w_u <= edge.w_v:
        return edge.u, edge.v
    return edge.v, edge.u
