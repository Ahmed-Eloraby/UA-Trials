"""Turn a fractional orientation into an integral one.

Orienting each edge toward an endpoint whose fraction is at least one half
is a 2-approximation. The rounded load of a node is at most twice the
fractional load of the edges sent there, and that fractional load is at
most the optimal LP value.
"""

from __future__ import annotations

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import Edge


class ThresholdRoundStep(Step):
    """Round unlocked edges that have a stored fraction toward ``u``.

    An endpoint is eligible when its fraction is at least ``threshold``
    (default one half) within ``tolerance``. If both endpoints are eligible,
    the larger fraction wins, then the lighter weight, then ``u``.

    Args:
        threshold: Minimum fraction that makes an endpoint eligible.
        tolerance: Slack used when comparing fractions to ``threshold``.
    """

    def __init__(self, threshold: float = 0.5, tolerance: float = 1e-8):
        """Store the fraction cutoff. Values outside ``(0, 1]`` are rejected.

        Args:
            threshold: Minimum fraction that makes an endpoint eligible.
            tolerance: Slack used when comparing fractions to ``threshold``.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        self.threshold = threshold
        self.tolerance = tolerance

    def run(self, ctx: SolveContext) -> None:
        """Round every unlocked edge that has a stored fraction and record the orientation.

        Args:
            ctx: Solve state whose ``fractional`` map is rounded into heads.
        """
        free_heads: dict[int, int] = {}
        for edge in ctx.free_edges():
            if edge.id not in ctx.fractional:
                raise RuntimeError(
                    f"edge {edge.id} has no fractional value to round"
                )
            head = self._round_edge(edge, ctx.fractional[edge.id])
            free_heads[edge.id] = head
            ctx.orientation[edge.id] = head
        ctx.consider(ctx.assignment_from(free_heads))

    def _round_edge(self, edge: Edge, fraction_u: float) -> int:
        """Choose the head for one edge from its fraction toward ``u``.

        Args:
            edge: Edge whose endpoints are the rounding candidates.
            fraction_u: LP fraction of this edge oriented toward ``u``.

        Returns:
            int: Chosen head, either ``edge.u`` or ``edge.v``.
        """
        fraction_v = 1.0 - fraction_u
        u_ok = fraction_u >= self.threshold - self.tolerance
        v_ok = fraction_v >= self.threshold - self.tolerance
        if u_ok and v_ok:
            if fraction_u > fraction_v + self.tolerance:
                return edge.u
            if fraction_v > fraction_u + self.tolerance:
                return edge.v
            return edge.lighter_head()
        if u_ok:
            return edge.u
        if v_ok:
            return edge.v
        return edge.u if fraction_u >= fraction_v else edge.v
