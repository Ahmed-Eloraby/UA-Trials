"""Greedy orientations.

Lightest-endpoint is the simple upper bound used by search. Load-aware
greedy often tightens that bound by looking at the load so far.
"""

from __future__ import annotations

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import ABS_TOL, Edge, Instance


def lightest_heads(instance: Instance, ctx: SolveContext) -> dict[int, int]:
    """Orient every unlocked edge toward its lighter endpoint. Ties go to ``u``.

    Args:
        instance: Graph whose unlocked edges are oriented.
        ctx: Solve state whose locked edges are left out of the result.

    Returns:
        dict[int, int]: Map from each unlocked edge id to its lighter endpoint.
    """
    heads: dict[int, int] = {}
    for edge in instance.edges:
        if edge.id in ctx.locked:
            continue
        heads[edge.id] = edge.lighter_head()
    return heads


def load_heads(instance: Instance, ctx: SolveContext) -> dict[int, int]:
    """Orient unlocked edges toward the endpoint with the smaller resulting load.

    Edges are processed by descending heavier weight, then descending absolute
    difference, then ascending edge id. Ties in the resulting load go to the
    lighter weight, then to ``u``.

    Args:
        instance: Graph whose unlocked edges are oriented.
        ctx: Solve state supplying locked load and the free-edge list.

    Returns:
        dict[int, int]: Map from each unlocked edge id to the chosen endpoint.
    """
    load = ctx.base_load()
    heads: dict[int, int] = {}
    ordered = sorted(
        ctx.free_edges(),
        key=lambda edge: (-max(edge.w_u, edge.w_v), -abs(edge.w_u - edge.w_v), edge.id),
    )
    for edge in ordered:
        head = _lower_resulting_load(edge, load)
        heads[edge.id] = head
        load[head] += edge.weight_toward(head)
    return heads


def _lower_resulting_load(edge: Edge, load: dict[int, float]) -> int:
    """Choose the endpoint whose load after accepting ``edge`` is smaller.

    Equal resulting loads use the lighter weight, then endpoint ``u``.

    Args:
        edge: Edge whose two endpoints are compared.
        load: Current load of each node before this edge is assigned.

    Returns:
        int: Endpoint with the smaller resulting load.
    """
    cost_u = load[edge.u] + edge.w_u
    cost_v = load[edge.v] + edge.w_v
    if abs(cost_u - cost_v) <= ABS_TOL:
        if edge.w_u < edge.w_v - ABS_TOL:
            return edge.u
        if edge.w_v < edge.w_u - ABS_TOL:
            return edge.v
        return edge.u
    if cost_u < cost_v:
        return edge.u
    return edge.v


class GreedyLightestStep(Step):
    """Propose the lightest-endpoint orientation and record it if it improves."""

    def run(self, ctx: SolveContext) -> None:
        """Write the lightest-endpoint heads and keep them when they improve the incumbent.

        Args:
            ctx: Solve state that receives the proposed orientation.
        """
        free = lightest_heads(ctx.instance, ctx)
        for edge_id, head in free.items():
            ctx.orientation[edge_id] = head
        ctx.consider(ctx.assignment_from(free))


class GreedyLoadStep(Step):
    """Propose the load-aware greedy orientation and record it if it improves."""

    def run(self, ctx: SolveContext) -> None:
        """Write the load-aware heads and keep them when they improve the incumbent.

        Args:
            ctx: Solve state that receives the proposed orientation.
        """
        free = load_heads(ctx.instance, ctx)
        for edge_id, head in free.items():
            ctx.orientation[edge_id] = head
        ctx.consider(ctx.assignment_from(free))
