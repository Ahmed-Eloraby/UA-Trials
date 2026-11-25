"""Graph-structured balance search that never solves an LP.

The outer search bisects a threshold ``T`` between two fences:

* ``lower`` moves only when a trial proves that no orientation has makespan
  at most ``T``. Every reject carries a witness.
* ``accepted`` is the smallest threshold whose trial returned a complete
  orientation with makespan at most ``2T``.

The measured incumbent is kept apart from both fences, so ``T`` and ``2T``
are never stored as the upper bound. The final makespan is at most
``2 * accepted <= 2 (1 + eps) * lower <= 2 (1 + eps) * OPT``. With integral
weights the thresholds are integers, and ``eps = 0`` finds the exact
boundary of the trial.

One trial at ``T`` runs on a scratch :class:`SolveContext`:

1. Exact reductions, repeated until none applies. Each keeps some
   orientation of makespan at most ``T`` whenever one exists.

   * An edge that weighs 0 at an endpoint goes there.
   * A node whose free edges all fit takes them all.
   * An edge that fits on one endpoint only goes there. An edge that fits
     on neither proves ``T`` infeasible.
   * A leaf (a node whose free edges all reach one neighbour) keeps the
     subset of its bundle that relieves the parent most, by an exact 0/1
     knapsack. A parent that still overflows proves ``T`` infeasible.
   * A node can hold at most one edge heavier than half its residual
     capacity. Edges heavy at both ends must therefore form trees and
     single cycles; a denser component proves ``T`` infeasible, and a
     component with one cycle fixes its pendant edges and pushes every
     other heavy edge away from its nodes.

2. A free component that is a plain cycle takes the better of its two
   rotations. Both stay within ``T``.

3. :func:`certified_push` orients what is left. It ends with every load at
   most ``2T``, or with node prices that prove ``T`` infeasible
   (:class:`PriceCertificate`). The push follows the helpful-path idea of
   Gairing, Monien and Woclaw (ICALP 2005), with exact prices in place of
   their gain rounding.

:class:`PolishStep` then shortens the incumbent with chains of edge flips.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, inf

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import Edge, Instance, loads, makespan

# A weight fits in a residual capacity when it is at most this much over.
_FIT = 1e-8
# Relative gap under which two priced weights count as equal.
_TIGHT = 1e-12
# Slack on the 2T check of a finished trial.
_BOUND = 1e-6
# Branch-and-bound nodes allowed per leaf bundle before the greedy split is used.
_LEAF_NODE_LIMIT = 200_000
# Bundles with more items than this use the greedy split (recursion depth).
_LEAF_ITEM_LIMIT = 400
# Prices are divided down once the largest one passes this value.
_PRICE_CEILING = 1e100
# Improving flip chains applied by one polish call, at most.
_POLISH_ROUNDS = 10_000


@dataclass(frozen=True)
class Trial:
    """Outcome of one threshold trial.

    Args:
        status: ``"accept"`` (makespan at most ``2T``), ``"reject"`` (``T`` is
            proved infeasible), or ``"unknown"`` (an orientation without either
            guarantee).
        heads: Complete orientation, or ``None`` for a reject.
        reason: Rule that proved a reject, or why a trial is unknown.

    Returns:
        Trial: Frozen trial outcome.
    """

    status: str
    heads: dict[int, int] | None
    reason: str = ""


@dataclass(frozen=True)
class PriceCertificate:
    """Node set ``S`` and positive prices ``z`` that prove a threshold infeasible.

    In any orientation, even a fractional one, every edge with both ends in
    ``S`` adds at least ``min(z_u * w_u, z_v * w_v)`` to the priced load of
    ``S``. When those minima already exceed the priced room
    ``sum(z_v * (T - base_v))``, some node of ``S`` must exceed ``T``.

    Args:
        nodes: The node set ``S``.
        prices: Price of every node in ``S``.

    Returns:
        PriceCertificate: Frozen witness.
    """

    nodes: frozenset[int]
    prices: dict[int, float]

    def holds(
        self,
        edges: list[Edge],
        base: dict[int, Fraction],
        threshold: float,
    ) -> bool:
        """Check the witness in exact rational arithmetic.

        Args:
            edges: Free edges the witness speaks about.
            base: Exact load already fixed at every node of ``S``.
            threshold: Candidate makespan ``T``.

        Returns:
            bool: True when the priced edges inside ``S`` exceed the priced room.
        """
        limit = Fraction(threshold)
        price = {node: Fraction(value) for node, value in self.prices.items()}
        inside = Fraction(0)
        for edge in edges:
            if edge.u in self.nodes and edge.v in self.nodes:
                inside += min(
                    price[edge.u] * Fraction(edge.w_u),
                    price[edge.v] * Fraction(edge.w_v),
                )
        room = sum(
            (price[node] * (limit - base[node]) for node in self.nodes),
            Fraction(0),
        )
        return inside > room


@dataclass(frozen=True)
class PushResult:
    """Outcome of :func:`certified_push`.

    Args:
        status: ``"balanced"`` (every load at most ``2T``), ``"certificate"``
            (a witness was found), or ``"step-limit"``.
        heads: Orientation of every pushed edge.
        certificate: Witness when ``status`` is ``"certificate"``.

    Returns:
        PushResult: Frozen push outcome.
    """

    status: str
    heads: dict[int, int]
    certificate: PriceCertificate | None = None


class _TrialState:
    """Partial orientation of one trial, kept on a scratch :class:`SolveContext`.

    Fixed edges are locked on the context. ``cap`` is the load each node can
    still accept, and ``incident`` lists the free edges at each node.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
    """

    def __init__(self, instance: Instance, threshold: float):
        """Start with every edge free and every capacity equal to ``T``.

        Args:
            instance: Graph being oriented.
            threshold: Candidate makespan ``T``.
        """
        self.instance = instance
        self.threshold = float(threshold)
        self.ctx = SolveContext(instance)
        self.cap = {node: self.threshold for node in instance.nodes}
        self.free = {edge.id for edge in instance.edges}
        self.incident: dict[int, set[int]] = {node: set() for node in instance.nodes}
        for edge in instance.edges:
            self.incident[edge.u].add(edge.id)
            self.incident[edge.v].add(edge.id)

    def free_at(self, node: int) -> list[Edge]:
        """Free edges at ``node``, by edge id.

        Args:
            node: Node whose free edges are listed.

        Returns:
            list[Edge]: Free incident edges in id order.
        """
        return [self.instance.edge(edge_id) for edge_id in sorted(self.incident[node])]

    def fits(self, edge: Edge, node: int) -> bool:
        """Whether ``edge`` still fits on ``node``.

        Args:
            edge: Free edge being tested.
            node: Endpoint that would receive it.

        Returns:
            bool: True when its weight at ``node`` fits the residual capacity.
        """
        return edge.weight_toward(node) <= self.cap[node] + _FIT

    def lock(self, edge: Edge, head: int) -> None:
        """Fix ``edge`` toward ``head`` and charge its weight there.

        Args:
            edge: Free edge to fix.
            head: Endpoint that receives it.
        """
        self.ctx.lock(edge.id, head)
        self.cap[head] -= edge.weight_toward(head)
        self.free.discard(edge.id)
        self.incident[edge.u].discard(edge.id)
        self.incident[edge.v].discard(edge.id)

    def load(self, node: int) -> float:
        """Load already fixed at ``node``.

        Args:
            node: Node whose load is read.

        Returns:
            float: ``T`` minus the residual capacity.
        """
        return self.threshold - self.cap[node]

    def exact_base(self) -> dict[int, Fraction]:
        """Fixed load of every node, summed in exact rational arithmetic.

        Returns:
            dict[int, Fraction]: Exact fixed load of every node.
        """
        base = {node: Fraction(0) for node in self.instance.nodes}
        for edge_id in self.ctx.locked:
            head = self.ctx.orientation[edge_id]
            base[head] += Fraction(self.instance.edge(edge_id).weight_toward(head))
        return base

    def heads(self) -> dict[int, int]:
        """Complete orientation once every edge is fixed.

        Returns:
            dict[int, int]: Head of every edge.
        """
        return self.ctx.assignment_from({})


def decide_graph(
    instance: Instance,
    threshold: float,
    leaf_node_limit: int = _LEAF_NODE_LIMIT,
    max_push_steps: int | None = None,
) -> Trial:
    """Run one trial at ``threshold``.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        leaf_node_limit: Branch-and-bound nodes per leaf bundle before the
            greedy split is used.
        max_push_steps: Push and price steps allowed, or ``None`` for the
            default that grows with the residual size.

    Returns:
        Trial: Accept with makespan at most ``2T``, reject with a reason, or
        unknown.
    """
    state = _TrialState(instance, threshold)
    reason = _reduce(state, leaf_node_limit)
    if reason:
        return Trial("reject", None, reason)
    _rotate_plain_cycles(state)
    status, note = "accept", ""
    if state.free:
        residual = [instance.edge(edge_id) for edge_id in sorted(state.free)]
        base = {node: state.load(node) for node in instance.nodes}
        result = certified_push(residual, base, state.threshold, max_push_steps)
        if result.status == "certificate":
            if result.certificate is not None and result.certificate.holds(
                residual, state.exact_base(), state.threshold
            ):
                return Trial("reject", None, "price-certificate")
            status, note = "unknown", "certificate-failed-check"
        elif result.status == "step-limit":
            status, note = "unknown", "push-step-limit"
        for edge in residual:
            state.lock(edge, result.heads[edge.id])
    heads = state.heads()
    if status == "accept" and makespan(instance, heads) > 2.0 * state.threshold + _BOUND:
        status, note = "unknown", "above-2T"
    return Trial(status, heads, note)


def _reduce(state: _TrialState, leaf_node_limit: int) -> str:
    """Apply the exact reductions until a full pass fixes nothing.

    Args:
        state: Trial state, updated in place.
        leaf_node_limit: Branch-and-bound nodes per leaf bundle.

    Returns:
        str: Reject reason, or ``""`` when no rule proved ``T`` infeasible.
    """
    while True:
        changed = _absorb_zero_weights(state)
        changed |= _low_load_stars(state)
        for rule in (
            _forced_edges,
            lambda current: _exact_leaf_bundles(current, leaf_node_limit),
            _heavy_edge_rule,
        ):
            moved, reason = rule(state)
            if reason:
                return reason
            changed |= moved
        if not changed:
            return ""


def _absorb_zero_weights(state: _TrialState) -> bool:
    """Send every free edge that weighs 0 at an endpoint to that endpoint.

    Args:
        state: Trial state, updated in place.

    Returns:
        bool: True when an edge was fixed.
    """
    changed = False
    for edge_id in sorted(state.free):
        edge = state.instance.edge(edge_id)
        for node in edge.endpoints():
            if edge.weight_toward(node) <= 0.0:
                state.lock(edge, node)
                changed = True
                break
    return changed


def _low_load_stars(state: _TrialState) -> bool:
    """Give a node all of its free edges when their total fits.

    Args:
        state: Trial state, updated in place.

    Returns:
        bool: True when an edge was fixed.
    """
    changed = False
    progress = True
    while progress:
        progress = False
        for node in state.instance.nodes:
            star = state.free_at(node)
            if star and sum(edge.weight_toward(node) for edge in star) <= state.cap[node] + _FIT:
                for edge in star:
                    state.lock(edge, node)
                progress = changed = True
    return changed


def _forced_edges(state: _TrialState) -> tuple[bool, str]:
    """Fix every free edge that fits on exactly one endpoint.

    Args:
        state: Trial state, updated in place.

    Returns:
        tuple[bool, str]: Whether an edge was fixed, and ``"edge-fits-nowhere"``
        when some edge fits on neither endpoint.
    """
    changed = False
    progress = True
    while progress:
        progress = False
        for edge_id in sorted(state.free):
            if edge_id not in state.free:
                continue
            edge = state.instance.edge(edge_id)
            fits_u = state.fits(edge, edge.u)
            fits_v = state.fits(edge, edge.v)
            if not fits_u and not fits_v:
                return changed, "edge-fits-nowhere"
            if fits_u != fits_v:
                state.lock(edge, edge.u if fits_u else edge.v)
                progress = changed = True
    return changed, ""


def _exact_leaf_bundles(state: _TrialState, node_limit: int) -> tuple[bool, str]:
    """Split every leaf bundle so the parent keeps as much room as possible.

    A leaf is a node whose free edges all reach one neighbour, the parent.
    The leaf keeps the subset that fits in its capacity and has the largest
    total weight at the parent. That subset is found by an exact 0/1
    knapsack, so any feasible orientation leaves the parent no more room.

    Args:
        state: Trial state, updated in place.
        node_limit: Branch-and-bound nodes before the greedy split is used.

    Returns:
        tuple[bool, str]: Whether a bundle was fixed, and
        ``"leaf-parent-overflow"`` when the parent cannot take the rest.
    """
    changed = False
    while True:
        leaf, parent = _find_leaf(state)
        if leaf is None or parent is None:
            return changed, ""
        bundle = state.free_at(leaf)
        kept = {edge.id for edge in _leaf_share(bundle, leaf, parent, state.cap[leaf], node_limit)}
        for edge in bundle:
            state.lock(edge, leaf if edge.id in kept else parent)
        changed = True
        if state.cap[parent] < -_FIT:
            return changed, "leaf-parent-overflow"


def _find_leaf(state: _TrialState) -> tuple[int | None, int | None]:
    """First node, in instance order, whose free edges all reach one neighbour.

    Args:
        state: Trial state to scan.

    Returns:
        tuple[int | None, int | None]: ``(leaf, parent)``, or ``(None, None)``.
    """
    for node in state.instance.nodes:
        neighbours = {state.instance.edge(edge_id).other(node) for edge_id in state.incident[node]}
        if len(neighbours) == 1:
            return node, next(iter(neighbours))
    return None, None


def _leaf_share(
    bundle: list[Edge],
    leaf: int,
    parent: int,
    capacity: float,
    node_limit: int,
) -> list[Edge]:
    """Edges of a leaf bundle that the leaf keeps.

    Edges that weigh nothing at the leaf are always kept. Edges that weigh
    nothing at the parent are never kept. The others go to an exact knapsack,
    or to the greedy split (fitting prefix plus one edge) if the search is
    too large.

    Args:
        bundle: Free edges between ``leaf`` and ``parent``.
        leaf: Node that keeps part of the bundle.
        parent: The leaf's only neighbour.
        capacity: Residual capacity of the leaf.
        node_limit: Branch-and-bound nodes before the greedy split is used.

    Returns:
        list[Edge]: Edges oriented toward the leaf.
    """
    always = [edge for edge in bundle if edge.weight_toward(leaf) <= 0.0]
    items = [
        edge
        for edge in bundle
        if edge.weight_toward(leaf) > 0.0 and edge.weight_toward(parent) > 0.0
    ]
    items.sort(key=lambda edge: (edge.weight_toward(leaf) / edge.weight_toward(parent), edge.id))
    chosen = _knapsack(items, leaf, parent, capacity, node_limit)
    if chosen is None:
        chosen = _greedy_share(items, leaf, capacity)
    return always + chosen


def _knapsack(
    items: list[Edge],
    leaf: int,
    parent: int,
    capacity: float,
    node_limit: int,
) -> list[Edge] | None:
    """Exact 0/1 knapsack: most parent weight within the leaf capacity.

    ``items`` must be sorted by leaf weight over parent weight, so the greedy
    fractional fill is an upper bound for branch-and-bound. Among equal
    parent weights the lighter leaf load wins.

    Args:
        items: Candidate edges in ratio order.
        leaf: Node whose capacity limits the choice.
        parent: Node whose weight is maximised.
        capacity: Residual capacity of the leaf.
        node_limit: Search nodes allowed before giving up.

    Returns:
        list[Edge] | None: Chosen edges, or ``None`` when the search is too large.
    """
    if len(items) > _LEAF_ITEM_LIMIT:
        return None
    sizes = [edge.weight_toward(leaf) for edge in items]
    values = [edge.weight_toward(parent) for edge in items]
    room = capacity + _FIT
    best: list = [-1.0, inf, []]
    pick: list[int] = []
    visited = 0

    def bound(index: int, space: float) -> float:
        """Greedy fractional fill of ``space`` from ``items[index:]``.

        Args:
            index: First item still undecided.
            space: Leaf capacity still free.

        Returns:
            float: Upper bound on the parent weight still reachable.
        """
        total = 0.0
        for size, value in zip(sizes[index:], values[index:]):
            if size <= space:
                total += value
                space -= size
            else:
                return total + value * space / size
        return total

    def search(index: int, size: float, value: float) -> bool:
        """Decide ``items[index]`` and recurse.

        Args:
            index: Position of the next undecided item.
            size: Leaf weight of the items picked so far.
            value: Parent weight of the items picked so far.

        Returns:
            bool: False when the node limit was reached.
        """
        nonlocal visited
        visited += 1
        if visited > node_limit:
            return False
        if value > best[0] or (value == best[0] and size < best[1]):
            best[:] = [value, size, list(pick)]
        if index == len(items) or value + bound(index, room - size) <= best[0]:
            return True
        if size + sizes[index] <= room:
            pick.append(index)
            finished = search(index + 1, size + sizes[index], value + values[index])
            pick.pop()
            if not finished:
                return False
        return search(index + 1, size, value)

    if not search(0, 0.0, 0.0):
        return None
    return [items[index] for index in best[2]]


def _greedy_share(items: list[Edge], leaf: int, capacity: float) -> list[Edge]:
    """Fitting prefix of ``items`` plus the next edge. The leaf stays within ``2T``.

    Args:
        items: Candidate edges in ratio order.
        leaf: Node that keeps the edges.
        capacity: Residual capacity of the leaf.

    Returns:
        list[Edge]: Edges oriented toward the leaf.
    """
    chosen: list[Edge] = []
    used = 0.0
    for edge in items:
        chosen.append(edge)
        used += edge.weight_toward(leaf)
        if used > capacity + _FIT:
            break
    return chosen


def _heavy_edge_rule(state: _TrialState) -> tuple[bool, str]:
    """Apply the one-heavy-edge-per-node rule.

    An edge is heavy at ``v`` when it weighs more than half of ``v``'s
    residual capacity, so ``v`` can hold at most one such edge. Edges heavy
    at both ends must each go to a different endpoint:

    * a component of them with more edges than nodes is infeasible;
    * a component with one cycle gives every node exactly one of its edges,
      so its pendant edges are forced and every other edge heavy at one of
      its nodes must go to its other endpoint.

    Args:
        state: Trial state, updated in place.

    Returns:
        tuple[bool, str]: Whether an edge was fixed, and a reject reason.
    """
    room = dict(state.cap)

    def heavy(edge: Edge, node: int) -> bool:
        """Whether ``edge`` weighs more than half of ``node``'s capacity.

        Args:
            edge: Free edge.
            node: One of its endpoints.

        Returns:
            bool: True when two such edges cannot both fit on ``node``.
        """
        return edge.weight_toward(node) > room[node] / 2.0 + _FIT

    adjacency: dict[int, list[Edge]] = {}
    for edge_id in sorted(state.free):
        edge = state.instance.edge(edge_id)
        if heavy(edge, edge.u) and heavy(edge, edge.v):
            adjacency.setdefault(edge.u, []).append(edge)
            adjacency.setdefault(edge.v, []).append(edge)
    changed = False
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        nodes, edge_ids = _heavy_component(adjacency, start)
        seen |= nodes
        if len(edge_ids) > len(nodes):
            return changed, "heavy-edges-overfull"
        if len(edge_ids) < len(nodes):
            continue
        degree = {node: len(adjacency[node]) for node in nodes}
        remaining = set(edge_ids)
        pending = deque(sorted(node for node in nodes if degree[node] == 1))
        while pending:
            node = pending.popleft()
            if degree[node] != 1:
                continue
            edge = next(edge for edge in adjacency[node] if edge.id in remaining)
            if not state.fits(edge, node):
                return changed, "heavy-edges-overfull"
            state.lock(edge, node)
            changed = True
            remaining.discard(edge.id)
            degree[node] = 0
            other = edge.other(node)
            degree[other] -= 1
            if degree[other] == 1:
                pending.append(other)
        for node in sorted(nodes):
            for edge in state.free_at(node):
                if edge.id in edge_ids or not heavy(edge, node):
                    continue
                other = edge.other(node)
                if not state.fits(edge, other):
                    return changed, "heavy-slot-conflict"
                state.lock(edge, other)
                changed = True
    return changed, ""


def _heavy_component(
    adjacency: dict[int, list[Edge]], start: int
) -> tuple[set[int], set[int]]:
    """Nodes and edge ids of the component of ``start`` in the heavy graph.

    Args:
        adjacency: Edges heavy at both ends, listed at each endpoint.
        start: Node whose component is collected.

    Returns:
        tuple[set[int], set[int]]: Node set and edge-id set of the component.
    """
    nodes = {start}
    edge_ids: set[int] = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for edge in adjacency[node]:
            edge_ids.add(edge.id)
            other = edge.other(node)
            if other not in nodes:
                nodes.add(other)
                queue.append(other)
    return nodes, edge_ids


def _rotate_plain_cycles(state: _TrialState) -> None:
    """Orient every free component that is a plain cycle by its better rotation.

    After the reductions every free edge fits on both ends and no node can
    take both of its cycle edges, so each node takes exactly one and both
    rotations stay within ``T``.

    Args:
        state: Trial state, updated in place.
    """
    seen: set[int] = set()
    for start in state.instance.nodes:
        if start in seen or not state.incident[start]:
            continue
        component = _free_component(state, start)
        seen |= component
        if not all(_has_two_neighbours(state, node) for node in component):
            continue
        walk = _cycle_walk(state, start)
        forward = max(state.load(head) + edge.weight_toward(head) for edge, _tail, head in walk)
        backward = max(state.load(tail) + edge.weight_toward(tail) for edge, tail, _head in walk)
        for edge, tail, head in walk:
            state.lock(edge, head if forward <= backward else tail)


def _free_component(state: _TrialState, start: int) -> set[int]:
    """Nodes joined to ``start`` by free edges.

    Args:
        state: Trial state to read.
        start: First node of the component.

    Returns:
        set[int]: Node set of the component.
    """
    nodes = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for edge_id in state.incident[node]:
            other = state.instance.edge(edge_id).other(node)
            if other not in nodes:
                nodes.add(other)
                queue.append(other)
    return nodes


def _has_two_neighbours(state: _TrialState, node: int) -> bool:
    """Whether ``node`` has exactly two free edges, to two different nodes.

    Args:
        state: Trial state to read.
        node: Node to test.

    Returns:
        bool: True for a node of a plain cycle.
    """
    edge_ids = state.incident[node]
    if len(edge_ids) != 2:
        return False
    first, second = (state.instance.edge(edge_id).other(node) for edge_id in edge_ids)
    return first != second


def _cycle_walk(state: _TrialState, start: int) -> list[tuple[Edge, int, int]]:
    """Walk a plain cycle once from ``start``.

    Args:
        state: Trial state to read.
        start: Node of the cycle where the walk begins.

    Returns:
        list[tuple[Edge, int, int]]: ``(edge, from, to)`` in walking order.
    """
    walk: list[tuple[Edge, int, int]] = []
    node = start
    edge = state.instance.edge(min(state.incident[start]))
    while True:
        following = edge.other(node)
        walk.append((edge, node, following))
        node = following
        if node == start:
            return walk
        edge = state.instance.edge(
            next(edge_id for edge_id in state.incident[node] if edge_id != edge.id)
        )


def certified_push(
    edges: list[Edge],
    base: dict[int, float],
    threshold: float,
    max_steps: int | None = None,
) -> PushResult:
    """Orient ``edges`` with every load at most ``2T``, or prove ``T`` infeasible.

    Every edge must weigh more than 0 and fit on both endpoints on top of
    ``base``. Nodes carry prices ``z``. Each edge sits on the endpoint where
    ``z * weight`` is smaller, and a tie makes the edge tight: it may flip
    without breaking that rule. Loads fall in three bands: Under (at most
    ``T``), Middle, and Over (above ``2T``).

    * Push: while an Over node reaches an Under node through tight flips,
      take the lowest helpful node (one that stays above ``T`` after giving
      its lightest downhill edge) and pass edges down one level at a time.
      Each node on the way keeps what it receives if it stays within ``2T``,
      otherwise it passes one of its own edges on. No node ever enters the
      Under or Over band.
    * Price: otherwise, let ``R`` be the nodes Over nodes reach by tight
      flips. If no edge leads out of ``R``, ``R`` and its prices are the
      witness. If one does, raise the prices of ``R`` until the cheapest such
      edge becomes tight.

    Args:
        edges: Free edges to orient.
        base: Load already fixed at each node.
        threshold: Candidate makespan ``T``.
        max_steps: Push and price steps allowed, or ``None`` for
            ``50 * (nodes + edges) ** 2``.

    Returns:
        PushResult: Balanced orientation, witness, or the orientation reached
        at the step limit.
    """
    pusher = _Pusher(edges, base, float(threshold))
    if max_steps is None:
        max_steps = 50 * (len(pusher.nodes) + len(edges)) ** 2
    return pusher.run(max_steps)


class _Pusher:
    """State of :func:`certified_push`: heads, loads, and node prices.

    Args:
        edges: Free edges to orient.
        base: Load already fixed at each node.
        threshold: Candidate makespan ``T``.
    """

    def __init__(self, edges: list[Edge], base: dict[int, float], threshold: float):
        """Put every edge on its cheaper endpoint, with every price at 1.

        Args:
            edges: Free edges to orient.
            base: Load already fixed at each node.
            threshold: Candidate makespan ``T``.
        """
        self.edges = list(edges)
        self.threshold = threshold
        self.top = 2.0 * threshold + _FIT
        self.incident: dict[int, list[Edge]] = {}
        for edge in self.edges:
            self.incident.setdefault(edge.u, []).append(edge)
            self.incident.setdefault(edge.v, []).append(edge)
        self.nodes = sorted(self.incident)
        self.price = {node: 1.0 for node in self.nodes}
        self.head = {edge.id: edge.lighter_head() for edge in self.edges}
        self.load = {node: float(base.get(node, 0.0)) for node in self.nodes}
        for edge in self.edges:
            head = self.head[edge.id]
            self.load[head] += edge.weight_toward(head)

    def run(self, max_steps: int) -> PushResult:
        """Alternate pushes and price raises until a stopping state.

        Args:
            max_steps: Push and price steps allowed.

        Returns:
            PushResult: Balanced orientation, witness, or step-limit state.
        """
        steps = 0
        while True:
            over = [node for node in self.nodes if self.load[node] > self.top]
            if not over:
                return PushResult("balanced", dict(self.head))
            if steps >= max_steps:
                return PushResult("step-limit", dict(self.head))
            steps += 1
            height = self._heights()
            if any(height[node] is not None for node in over) and self._push(height):
                continue
            region = self._reachable(over)
            crossing = [
                edge
                for edge in self.edges
                if self.head[edge.id] in region and edge.other(self.head[edge.id]) not in region
            ]
            if not crossing:
                prices = {node: self.price[node] for node in region}
                return PushResult(
                    "certificate",
                    dict(self.head),
                    PriceCertificate(frozenset(region), prices),
                )
            factor = min(self._ratio(edge) for edge in crossing)
            for node in region:
                self.price[node] *= factor
            largest = max(self.price.values())
            if largest > _PRICE_CEILING:
                for node in self.nodes:
                    self.price[node] /= largest

    def _priced(self, edge: Edge, node: int) -> float:
        """Price of ``node`` times the weight of ``edge`` there.

        Args:
            edge: Edge being priced.
            node: One of its endpoints.

        Returns:
            float: Priced weight.
        """
        return self.price[node] * edge.weight_toward(node)

    def _ratio(self, edge: Edge) -> float:
        """Priced weight at the tail over priced weight at the head.

        Args:
            edge: Edge whose two priced weights are compared.

        Returns:
            float: At least 1 while the edge sits on its cheaper end.
        """
        head = self.head[edge.id]
        here = self._priced(edge, head)
        if here <= 0.0:
            return inf
        return self._priced(edge, edge.other(head)) / here

    def _tight(self, edge: Edge) -> bool:
        """Whether ``edge`` costs the same priced weight on both ends.

        Args:
            edge: Edge to test.

        Returns:
            bool: True when flipping it keeps every edge on a cheaper end.
        """
        head = self.head[edge.id]
        here = self._priced(edge, head)
        return self._priced(edge, edge.other(head)) <= here * (1.0 + _TIGHT)

    def _heights(self) -> dict[int, int | None]:
        """Number of tight flips from each node to the nearest Under node.

        Returns:
            dict[int, int | None]: Height of each node, ``None`` if unreachable.
        """
        height: dict[int, int | None] = {node: None for node in self.nodes}
        queue: deque[int] = deque()
        for node in self.nodes:
            if self.load[node] <= self.threshold + _FIT:
                height[node] = 0
                queue.append(node)
        while queue:
            tail = queue.popleft()
            level = height[tail]
            for edge in self.incident[tail]:
                head = self.head[edge.id]
                if head == tail or height[head] is not None or not self._tight(edge):
                    continue
                height[head] = level + 1
                queue.append(head)
        return height

    def _downhill(self, node: int, height: dict[int, int | None]) -> Edge | None:
        """Lightest tight edge at ``node`` whose other end is one level lower.

        Args:
            node: Node that would give the edge away.
            height: Heights from :meth:`_heights`.

        Returns:
            Edge | None: The edge, or ``None`` when there is none.
        """
        level = height[node]
        if not level:
            return None
        best: tuple[tuple[float, int], Edge] | None = None
        for edge in self.incident[node]:
            if self.head[edge.id] != node or not self._tight(edge):
                continue
            if height[edge.other(node)] != level - 1:
                continue
            key = (edge.weight_toward(node), edge.id)
            if best is None or key < best[0]:
                best = (key, edge)
        return None if best is None else best[1]

    def _push(self, height: dict[int, int | None]) -> bool:
        """Push edges down one helpful path.

        The path starts at the lowest helpful node. Each later node keeps the
        edge it receives if its load stays within ``2T``, otherwise it gives
        its own lightest downhill edge one level down.

        Args:
            height: Heights from :meth:`_heights`.

        Returns:
            bool: False when no helpful node exists.
        """
        helpful = []
        for node in self.nodes:
            edge = self._downhill(node, height)
            if edge is not None and self.load[node] - edge.weight_toward(node) > self.threshold + _FIT:
                helpful.append((height[node], node, edge))
        if not helpful:
            return False
        _level, node, edge = min(helpful, key=lambda item: (item[0], item[1]))
        moves: list[tuple[Edge, int]] = []
        while edge is not None:
            receiver = edge.other(node)
            moves.append((edge, receiver))
            if self.load[receiver] + edge.weight_toward(receiver) <= self.top:
                break
            node = receiver
            edge = self._downhill(node, height)
        for edge, receiver in moves:
            giver = self.head[edge.id]
            self.load[giver] -= edge.weight_toward(giver)
            self.load[receiver] += edge.weight_toward(receiver)
            self.head[edge.id] = receiver
        return True

    def _reachable(self, sources: list[int]) -> set[int]:
        """Nodes that ``sources`` reach by flipping tight edges.

        Args:
            sources: Starting nodes.

        Returns:
            set[int]: Reached nodes, including ``sources``.
        """
        region = set(sources)
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            for edge in self.incident[node]:
                if self.head[edge.id] != node or not self._tight(edge):
                    continue
                other = edge.other(node)
                if other not in region:
                    region.add(other)
                    queue.append(other)
        return region


class GraphBalanceSearchStep(Step):
    """Bisect the threshold with :func:`decide_graph` and keep the best orientation.

    ``lower`` moves on a reject, ``accepted`` on an accept, and the measured
    incumbent is kept on the context. An unknown trial is considered as an
    orientation and ends the search, because it moves no fence.

    Args:
        eps: Relative gap between ``accepted`` and ``lower`` at which to stop.
        leaf_node_limit: Branch-and-bound nodes per leaf bundle.
        max_push_steps: Push and price steps per trial, or ``None`` for the
            default.
    """

    def __init__(
        self,
        eps: float = 0.25,
        leaf_node_limit: int = _LEAF_NODE_LIMIT,
        max_push_steps: int | None = None,
    ):
        """Store the stopping gap and the per-trial limits.

        Args:
            eps: Relative gap between ``accepted`` and ``lower`` at which to stop.
            leaf_node_limit: Branch-and-bound nodes per leaf bundle.
            max_push_steps: Push and price steps per trial, or ``None``.
        """
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.eps = eps
        self.leaf_node_limit = leaf_node_limit
        self.max_push_steps = max_push_steps

    def run(self, ctx: SolveContext) -> None:
        """Search thresholds and record the incumbent, the fences, and the trials.

        Args:
            ctx: Solve state that receives the incumbent orientation and bounds.
        """
        instance = ctx.instance
        if ctx.best_so_far_assignment is None:
            ctx.consider({edge.id: edge.lighter_head() for edge in instance.edges})
        if ctx.best_so_far_assignment is None:
            raise RuntimeError("graph balance search has no incumbent orientation")
        integral = instance.weights_are_integral()
        lower = ctx.analytic_lower_bound()
        if integral:
            lower = float(ceil(lower - 1e-9))
        accepted = ctx.best_so_far_makespan
        reasons: Counter[str] = Counter()
        trials = 0
        certified = True
        while self._open(lower, accepted, ctx.best_so_far_makespan, integral):
            if integral:
                threshold = float(floor((lower + accepted) / 2.0))
            else:
                threshold = (lower + accepted) / 2.0
            trial = decide_graph(instance, threshold, self.leaf_node_limit, self.max_push_steps)
            trials += 1
            if trial.status == "reject":
                reasons[trial.reason] += 1
                lower = threshold + 1.0 if integral else threshold
                continue
            if trial.heads is None:
                raise RuntimeError("graph balance trial returned no orientation")
            ctx.consider(trial.heads)
            if trial.status == "accept":
                accepted = threshold
            else:
                certified = False
                ctx.metadata["graph_unknown_reason"] = trial.reason
                break
        ctx.raise_lower_bound(lower)
        ctx.metadata["graph_lower"] = lower
        ctx.metadata["graph_accepted"] = accepted
        ctx.metadata["graph_upper"] = ctx.best_so_far_makespan
        ctx.metadata["graph_trials"] = trials
        ctx.metadata["graph_reject_reasons"] = dict(reasons)
        ctx.metadata["graph_certified"] = certified

    def _open(self, lower: float, accepted: float, upper: float, integral: bool) -> bool:
        """Whether the fences are still too far apart.

        Args:
            lower: Proven lower bound.
            accepted: Smallest accepted threshold.
            upper: Measured incumbent makespan.
            integral: Whether thresholds are integers.

        Returns:
            bool: True when another trial should run.
        """
        if accepted <= lower or upper <= lower:
            return False
        gap = (1.0 + self.eps) * lower
        if accepted <= gap or upper <= gap:
            return False
        if not integral and accepted - lower <= 1e-9 * max(1.0, accepted):
            return False
        return True


def polish(instance: Instance, heads: dict[int, int], depth: int = 3) -> dict[int, int]:
    """Lower the makespan with chains of edge flips. Never raises it.

    From a node at the current makespan ``M``, a chain gives one edge to a
    neighbour, which keeps it or gives one of its own edges on, up to
    ``depth`` flips. A chain is applied only when every node it touches ends
    below ``M``. Polishing stops when no node at ``M`` has such a chain.

    Args:
        instance: Graph whose orientation is polished.
        heads: Complete orientation to start from.
        depth: Longest chain of flips tried.

    Returns:
        dict[int, int]: Orientation with makespan at most the starting one.
    """
    heads = dict(heads)
    load = loads(instance, heads)
    held: dict[int, set[int]] = {node: set() for node in instance.nodes}
    for edge in instance.edges:
        held[heads[edge.id]].add(edge.id)
    for _round in range(_POLISH_ROUNDS):
        top = max(load.values(), default=0.0)
        chain = None
        for node in instance.nodes:
            if load[node] >= top - _FIT:
                chain = _flip_chain(instance, node, top, load, held, depth)
                if chain:
                    break
        if not chain:
            break
        for edge, receiver in chain:
            giver = heads[edge.id]
            load[giver] -= edge.weight_toward(giver)
            load[receiver] += edge.weight_toward(receiver)
            held[giver].discard(edge.id)
            held[receiver].add(edge.id)
            heads[edge.id] = receiver
    return heads


def _flip_chain(
    instance: Instance,
    start: int,
    top: float,
    load: dict[int, float],
    held: dict[int, set[int]],
    depth: int,
) -> list[tuple[Edge, int]] | None:
    """Chain of flips from ``start`` after which every touched node is below ``top``.

    Args:
        instance: Graph being polished.
        start: Node at the current makespan.
        top: Current makespan.
        load: Load of every node.
        held: Edge ids currently oriented toward each node.
        depth: Longest chain of flips tried.

    Returns:
        list[tuple[Edge, int]] | None: ``(edge, receiver)`` flips, or ``None``.
    """

    def extend(
        node: int, incoming: float, budget: int, visited: frozenset[int]
    ) -> list[tuple[Edge, int]] | None:
        """Make ``node`` end below ``top`` after receiving ``incoming``.

        Args:
            node: Node on the chain.
            incoming: Weight it receives from the previous flip.
            budget: Flips still allowed.
            visited: Nodes already on the chain.

        Returns:
            list[tuple[Edge, int]] | None: Remaining flips, or ``None``.
        """
        if load[node] + incoming < top - _FIT:
            return []
        if budget == 0:
            return None
        candidates = sorted(
            (instance.edge(edge_id) for edge_id in held[node]),
            key=lambda edge: (-edge.weight_toward(node), edge.id),
        )
        for edge in candidates:
            receiver = edge.other(node)
            if receiver in visited:
                continue
            if load[node] + incoming - edge.weight_toward(node) >= top - _FIT:
                continue
            rest = extend(
                receiver, edge.weight_toward(receiver), budget - 1, visited | {receiver}
            )
            if rest is not None:
                return [(edge, receiver)] + rest
        return None

    return extend(start, 0.0, depth, frozenset({start}))


class PolishStep(Step):
    """Polish the incumbent with :func:`polish` and keep it if it improves.

    Args:
        depth: Longest chain of flips tried.
    """

    def __init__(self, depth: int = 3):
        """Store the chain depth.

        Args:
            depth: Longest chain of flips tried.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth

    def run(self, ctx: SolveContext) -> None:
        """Polish the incumbent orientation, if there is one.

        Args:
            ctx: Solve state whose incumbent is polished.
        """
        if ctx.best_so_far_assignment is None:
            return
        start = ctx.best_so_far_assignment.edge_node_map
        ctx.consider(polish(ctx.instance, start, self.depth))
