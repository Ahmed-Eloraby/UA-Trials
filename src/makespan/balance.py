"""Graph-balancing search.

One trial at a threshold T tries to build a complete orientation. The outer
search moves two fences and nothing else:

* ``lower`` moves only when a trial proves that T is infeasible.
* ``upper`` moves only to the measured makespan of a complete orientation
  that is strictly better than the incumbent.
* The trial value T, and twice that value, are never stored as ``upper``.

The search stops when ``upper <= (1 + eps) * lower``, and also when a trial
moves neither fence. Before the residual rounding, every trial runs the same
reductions: low-load stars, forced edges, and leaf bundles. What remains is
handed to a residual rounder. ``balance-lp`` rounds a feasible capacitated
LP at one half. ``balance-flow`` rounds by a saturating max flow.
"""

from __future__ import annotations

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.flow import assign_free_by_flow
from makespan.model import Edge, Instance, makespan
from makespan.relaxation import CapacitatedRelaxationStep
from makespan.rounding import ThresholdRoundStep

# A weight fits in a residual capacity when it is at most this much over.
_FIT = 1e-8
# Leaf loads and parent capacities use this slack.
_BOUND = 1e-6


class ResidualRounder:
    """Turn the edges still free after the combinatorial reductions into heads.

    ``partial`` maps every edge id to a head, or to ``None`` when the edge is
    still free. ``cap`` is the residual load each node can still accept.
    ``open_nodes`` are the nodes that were not closed as leaves. The rounder
    returns a complete orientation, or ``None`` when its relaxation proves
    that these residuals cannot hold the free edges.
    """

    def round(
        self,
        instance: Instance,
        partial: dict[int, int | None],
        cap: dict[int, float],
        open_nodes: set[int],
    ) -> dict[int, int] | None:
        """Return a complete orientation, or ``None`` when the relaxation is infeasible.

        Args:
            instance: Graph whose remaining free edges are rounded.
            partial: Map from every edge id to a chosen head, or ``None`` if free.
            cap: Residual load each node can still accept.
            open_nodes: Nodes that were not closed as leaves.

        Returns:
            dict[int, int] | None: Complete orientation, or ``None`` when the residual relaxation is infeasible.
        """
        raise NotImplementedError


class LpResidualRounder(ResidualRounder):
    """Residual rounder: capacitated LP, then threshold rounding at one half."""

    def round(
        self,
        instance: Instance,
        partial: dict[int, int | None],
        cap: dict[int, float],
        open_nodes: set[int],
    ) -> dict[int, int] | None:
        """Solve the residual LP on a scratch context and round its fractions.

        The scratch context is thrown away. Locked edges are the ones this
        trial has already oriented, so the LP and the rounding step see only
        the free edges.

        Args:
            instance: Graph whose remaining free edges are rounded.
            partial: Map from every edge id to a chosen head, or ``None`` if free.
            cap: Residual load each node can still accept.
            open_nodes: Nodes that were not closed as leaves.

        Returns:
            dict[int, int] | None: Complete orientation, or ``None`` when the residual relaxation is infeasible.
        """
        scratch = SolveContext(instance)
        for edge in instance.edges:
            head = partial[edge.id]
            if head is not None:
                scratch.lock(edge.id, head)
        CapacitatedRelaxationStep(cap, open_nodes).run(scratch)
        if not scratch.metadata.get("residual_feasible"):
            return None
        ThresholdRoundStep().run(scratch)
        if scratch.best_so_far_assignment is None:
            return None
        return dict(scratch.best_so_far_assignment.edge_node_map)


class FlowResidualRounder(ResidualRounder):
    """Residual rounder: one saturating max flow under the residual capacities.

    Closed leaves are given a negative capacity so the flow cannot send
    another edge into them.
    """

    def round(
        self,
        instance: Instance,
        partial: dict[int, int | None],
        cap: dict[int, float],
        open_nodes: set[int],
    ) -> dict[int, int] | None:
        """Assign every free edge by max flow, or return ``None`` if it cannot.

        Args:
            instance: Graph whose remaining free edges are rounded.
            partial: Map from every edge id to a chosen head, or ``None`` if free.
            cap: Residual load each node can still accept.
            open_nodes: Nodes that were not closed as leaves.

        Returns:
            dict[int, int] | None: Complete orientation, or ``None`` when the residual relaxation is infeasible.
        """
        locked = {
            edge.id: partial[edge.id]
            for edge in instance.edges
            if partial[edge.id] is not None
        }
        residual = {
            node: cap[node] if node in open_nodes else -1.0 for node in instance.nodes
        }
        return assign_free_by_flow(instance, locked, residual)


class BalanceSearchStep(Step):
    """Run the fenced threshold search and keep the best orientation it finds.

    ``rounder`` is the only piece that differs between the LP version and
    the max-flow version. ``eps`` is the relative gap at which the fences
    are close enough to stop. The initial incumbent is the lightest-endpoint
    orientation when the context does not already have a better one.

    Args:
        rounder: Residual LP or flow rounder used after combinatorial reductions.
        eps: Relative gap at which the search fences are close enough to stop.
    """

    def __init__(self, rounder: ResidualRounder, eps: float = 0.25):
        """Store the residual rounder and the stopping gap.

        Args:
            rounder: Residual LP or flow rounder used after combinatorial reductions.
            eps: Relative gap at which the search fences are close enough to stop.
        """
        self.rounder = rounder
        self.eps = eps

    def run(self, ctx: SolveContext) -> None:
        """Search thresholds and record the incumbent, the fences, and the trial count.

        Args:
            ctx: Solve state that receives the incumbent orientation and fences.
        """
        instance = ctx.instance
        if ctx.best_so_far_assignment is None:
            ctx.consider({edge.id: edge.lighter_head() for edge in instance.edges})
        if ctx.best_so_far_assignment is None:
            raise RuntimeError("balance search has no incumbent orientation")
        upper = ctx.best_so_far_makespan
        lower = ctx.analytic_lower_bound()
        incumbent = dict(ctx.best_so_far_assignment.edge_node_map)
        trials = 0
        while upper > (1.0 + self.eps) * lower:
            threshold = (lower + upper) / 2.0
            lower_before = lower
            upper_before = upper
            status, trial = decide(instance, threshold, self.rounder)
            if status == "reject":
                lower = threshold
            else:
                if trial is None:
                    raise RuntimeError("balance trial accepted without an orientation")
                else:
                    measured = makespan(instance, trial)
                    if measured < upper:
                        incumbent = trial
                        upper = measured

            if lower == lower_before and upper == upper_before:
                break
        ctx.consider(incumbent)
        ctx.raise_lower_bound(lower)
        ctx.metadata["balance_lower"] = lower
        ctx.metadata["balance_upper"] = upper
        ctx.metadata["balance_trials"] = trials


def decide(
    instance: Instance,
    threshold: float,
    rounder: ResidualRounder,
) -> tuple[str, dict[int, int] | None]:
    """Try one threshold and return ``accept`` or ``reject`` plus an orientation.

    Combinatorial reductions run first. What remains is handed to
    ``rounder``. A trial that already oriented every edge accepts that
    assignment. A trial the rounder cannot complete is a reject.

    Args:
        instance: Graph being oriented at this trial.
        threshold: Candidate makespan ``T`` used as residual capacity.
        rounder: Residual rounder applied to edges still free after reductions.

    Returns:
        tuple[str, dict[int, int] | None]: ``("accept", heads)`` or ``("reject", None)``.
    """
    cap = {node: float(threshold) for node in instance.nodes}
    orientation: dict[int, int | None] = {edge.id: None for edge in instance.edges}
    free = {edge.id for edge in instance.edges}

    while True:
        before = set(free)
        low_load_assignment(instance, free, cap, orientation)
        if force_closure(instance, free, cap, orientation) == "reject":
            return "reject", None
        status, closed = leaf_pruning(instance, free, cap, orientation, threshold)
        if status == "reject":
            return "reject", None
        if free == before:
            break

    snapshot = dict(orientation)
    open_nodes = {node for node in instance.nodes if node not in closed}
    if not free:
        return "accept", _complete_heads(instance, snapshot)
    rounded = rounder.round(instance, snapshot, cap, open_nodes)
    if rounded is None:
        return "reject", None
    return "accept", rounded


def low_load_assignment(
    instance: Instance,
    free: set[int],
    cap: dict[int, float],
    orientation: dict[int, int | None],
) -> None:
    """Give a node every free incident edge when their total weight fits.

    The test is repeated until a full pass assigns nothing. It does not
    reject: a star that does not fit is left free.

    Args:
        instance: Graph whose free stars are tested.
        free: Edge ids still unoriented. Assigned ids are removed in place.
        cap: Residual load each node can still accept. Updated in place.
        orientation: Partial heads. Newly assigned edges are written in place.
    """
    by_id = {edge.id: edge for edge in instance.edges}
    changed = True
    while changed:
        changed = False
        for node in instance.nodes:
            incident = [
                by_id[edge_id]
                for edge_id in free
                if node in (by_id[edge_id].u, by_id[edge_id].v)
            ]
            if not incident:
                continue
            total = sum(edge.weight_toward(node) for edge in incident)
            if total <= cap[node] + _FIT:
                for edge in incident:
                    orientation[edge.id] = node
                    free.remove(edge.id)
                cap[node] -= total
                changed = True


def force_closure(
    instance: Instance,
    free: set[int],
    cap: dict[int, float],
    orientation: dict[int, int | None],
) -> str:
    """Orient every free edge that fits on exactly one endpoint.

    An edge that fits on neither endpoint rejects the trial: no completion
    can respect the residual capacities. Fitting on both endpoints leaves
    the edge free. Caps shrink as edges are fixed, so the scan repeats
    until a pass changes nothing.

    Args:
        instance: Graph whose free edges are scanned.
        free: Edge ids still unoriented. Forced ids are removed in place.
        cap: Residual load each node can still accept. Updated in place.
        orientation: Partial heads. Forced edges are written in place.

    Returns:
        str: ``"accept"`` when every remaining free edge fits on both sides or was forced, otherwise ``"reject"``.
    """
    while True:
        changed = False
        for edge in instance.edges:
            if edge.id not in free:
                continue
            fits_u = edge.w_u <= cap[edge.u] + _FIT
            fits_v = edge.w_v <= cap[edge.v] + _FIT
            if not fits_u and not fits_v:
                return "reject"
            if fits_u and not fits_v:
                orientation[edge.id] = edge.u
                cap[edge.u] -= edge.w_u
                free.remove(edge.id)
                changed = True
            elif fits_v and not fits_u:
                orientation[edge.id] = edge.v
                cap[edge.v] -= edge.w_v
                free.remove(edge.id)
                changed = True
        if not changed:
            return "accept"


def leaf_pruning(
    instance: Instance,
    free: set[int],
    cap: dict[int, float],
    orientation: dict[int, int | None],
    threshold: float,
) -> tuple[str, set[int]]:
    """Orient one leaf bundle at a time and close that leaf.

    A leaf is a node with free edges to only one other node. Its bundle is
    ordered by nondecreasing ``weight toward the leaf / weight toward the
    parent``. A zero weight toward the parent sorts last. The longest prefix
    that fits on the leaf is kept, and the next edge is kept as well when
    the prefix is not the whole bundle. Every other edge in the bundle is
    sent to the parent. A parent whose residual capacity becomes negative
    rejects the trial. The bundle is already written into ``orientation``.

    Args:
        instance: Graph whose leaf bundles are closed.
        free: Edge ids still unoriented. Closed-bundle ids are removed in place.
        cap: Residual load each node can still accept. Updated in place.
        orientation: Partial heads. Bundle assignments are written in place.
        threshold: Candidate makespan ``T`` used to check closed-leaf loads.

    Returns:
        tuple[str, set[int]]: ``("accept", closed)`` or ``("reject", closed)``, where ``closed`` is the set of closed leaves.
    """
    by_id = {edge.id: edge for edge in instance.edges}
    closed: set[int] = set()
    while free:
        leaf, parent = _first_leaf(instance, free, by_id)
        if leaf is None or parent is None:
            return "accept", closed
        bundle = [
            by_id[edge_id]
            for edge_id in free
            if {by_id[edge_id].u, by_id[edge_id].v} == {leaf, parent}
            or {by_id[edge_id].v, by_id[edge_id].u} == {leaf, parent}
        ]
        bundle.sort(key=lambda edge: _leaf_sort_key(edge, leaf, parent), reverse=True)
        prefix: list[Edge] = []
        prefix_weight = 0.0
        for edge in bundle:
            weight = edge.weight_toward(leaf)
            if prefix_weight + weight <= cap[leaf] + _FIT:
                prefix.append(edge)
                prefix_weight += weight
            else:
                break
        chosen = list(prefix)
        if len(prefix) < len(bundle):
            chosen.append(bundle[len(prefix)])
        chosen_ids = {edge.id for edge in chosen}
        for edge in chosen:
            orientation[edge.id] = leaf
        tail_weight = 0.0
        for edge in bundle:
            if edge.id not in chosen_ids:
                orientation[edge.id] = parent
                tail_weight += edge.weight_toward(parent)
        cap[parent] -= tail_weight
        closed.add(leaf)
        leaf_load = sum(
            edge.weight_toward(leaf)
            for edge in instance.edges
            if orientation[edge.id] == leaf
        )
        if leaf_load > 2.0 * threshold + _BOUND:
            raise RuntimeError(
                f"closed leaf {leaf} load {leaf_load} exceeds 2T={2.0 * threshold}"
            )
        if cap[parent] < -_FIT:
            return "reject", closed
        for edge in bundle:
            free.remove(edge.id)
    return "accept", closed


def _complete_heads(
    instance: Instance, partial: dict[int, int | None]
) -> dict[int, int]:
    """Copy a fully assigned partial orientation into a heads map.

    Args:
        instance: Graph whose edges must all have a head in ``partial``.
        partial: Map from every edge id to a chosen head.

    Returns:
        dict[int, int]: Complete orientation of every edge.
    """
    heads: dict[int, int] = {}
    for edge in instance.edges:
        head = partial[edge.id]
        if head is None:
            raise RuntimeError(f"edge {edge.id} was left unoriented")
        heads[edge.id] = head
    return heads


def _first_leaf(
    instance: Instance,
    free: set[int],
    by_id: dict[int, Edge],
) -> tuple[int | None, int | None]:
    """First node, in instance order, whose free edges all touch one neighbor.

    Args:
        instance: Graph whose nodes are scanned in stored order.
        free: Edge ids that are still unoriented.
        by_id: Lookup from edge id to :class:`Edge`.

    Returns:
        tuple[int | None, int | None]: ``(leaf, parent)``, or ``(None, None)`` when no leaf remains.
    """
    for node in instance.nodes:
        neighbors: set[int] = set()
        for edge_id in free:
            edge = by_id[edge_id]
            if edge.u == node:
                neighbors.add(edge.v)
            elif edge.v == node:
                neighbors.add(edge.u)
        if len(neighbors) == 1:
            return node, next(iter(neighbors))
    return None, None


def _leaf_sort_key(edge: Edge, leaf: int, parent: int) -> tuple:
    """Nondecreasing leaf-weight / parent-weight. A zero parent weight sorts last.

    Args:
        edge: Bundle edge being ordered.
        leaf: Closed leaf endpoint.
        parent: Unique neighbor of that leaf.

    Returns:
        tuple: Sort key so zero parent-weight edges come last, then increasing leaf/parent ratio.
    """
    weight_toward_parent = edge.weight_toward(parent)
    weight_toward_leaf = edge.weight_toward(leaf)
    return (
        weight_toward_parent,
        (
            weight_toward_parent / weight_toward_leaf
            if weight_toward_leaf > 0
            else float("inf")
        ),
    )
    # if weight_toward_parent <= _FIT:
    #     return (1, 0.0, edge.id)
    # return (0, weight_toward_leaf / weight_toward_parent, edge.id)
