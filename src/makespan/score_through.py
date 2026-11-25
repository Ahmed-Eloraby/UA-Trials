"""Through-flow threshold search.

A trial at a threshold ``T`` works in four steps. When the name asks for
it, leaves are peeled before any node is given a group of edges. A parent
that drops to one neighbour is peeled on the next scan. A node is then given
every free edge when the whole group fits, and peeling runs again if that
creates a leaf. The trial scores each node that still has a free edge,
without pointing anything, then points the remaining edges by the flow
through each node.

The score imagines two loads. The in-load is the weight of pointing every
free edge at the node. The out-load is the weight of pointing them all away.
The metric is the ratio of those loads, or their difference.

Nodes and edges are ordered from the weighted graph before a trial. A node's
place comes from its incident weights and its neighbours' places, repeated
until the places settle. A tie gives one node its own place and repeats.
The kept order is the one whose edges, written in that order, are greatest.
Scans, ties, and the arcs of the flow network follow that order.

Each bundle gets one direction, from the higher score to the lower score.
Equal scores use the free component of that bundle. The fraction is the hop
distance from that component's highest score, divided by the same distance
plus the hop distance from its lowest score. The smaller fraction is the
start, so the bundle points toward the low score. An equal fraction goes
from the earlier place to the later place. The node at the start of that
direction is the one whose turn it is. The other end is its neighbour. Whose turn comes first is the variant. ``flow-desc`` takes the
largest flow through a node first, and that is the plain ``through`` name.
``flow-asc`` takes the smallest flow first. ``flow-room-asc`` and
``flow-room-desc`` use the flow divided by the node's room, smallest first
or largest first. The room is ``T`` minus the load already given to the node.
On its turn, the current node keeps an edge while the edge fits, and
otherwise gives the edge to the neighbour. ``quota`` spends the flow on each
bundle as weight delivered to the lower-score end. ``topo`` skips the flow
and takes turns in topological order, a node before the neighbours it points
to. That pass ends the trial.

The search keeps the finished orientation with the smaller makespan.
A failed trial leaves the stored lower bound unchanged.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from math import inf

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import ABS_TOL, Instance, makespan, nearly_leq

METRICS = ("ratio", "diff")
PRUNE_NAMES = (("peel", True), ("raw", False))
ORDERS = (
    "flow-desc",
    "flow-asc",
    "flow-room-asc",
    "flow-room-desc",
    "quota",
    "topo",
)
# Print load, free edges, and orientation after each step.
DEBUG = True

_MetricKey = Fraction
_SOURCE = 0
_SINK = 1
_trace: list[str] = []
_trace_depth = 0
_free_label: dict[int, str] = {}
_arrow: dict[int, str] = {}


def through_names() -> list[str]:
    """Command-line names of the twenty-four through algorithms.

    The order is metric, then pruning, then turn order. ``flow-desc`` keeps
    the plain ``through`` name. The other orders add a suffix. There is no
    early name.

    Returns:
        list[str]: ``score-{ratio|diff}-{peel|raw}-through[-{order}]``.
    """
    names = []
    for metric in METRICS:
        for prune_name, _prune in PRUNE_NAMES:
            for order in ORDERS:
                suffix = "" if order == "flow-desc" else f"-{order}"
                names.append(f"score-{metric}-{prune_name}-through{suffix}")
    return names


def _ratio(numerator: float, denominator: float) -> _MetricKey:
    """Exact quotient, used as an increasing sort key.

    Every edge weight is strictly positive, so a scored load and an endpoint
    weight used as a denominator are strictly positive. The quotient is finite.

    Args:
        numerator: Load or weight on the side being compared.
        denominator: Load or weight it is compared with.

    Returns:
        _MetricKey: ``numerator / denominator``.
    """
    return Fraction(numerator) / Fraction(denominator)


def _edge_key(metric: str, at_node: float, at_neighbour: float) -> _MetricKey:
    """Increasing key for one edge, from the node toward its neighbour.

    A smaller key means the edge is cheaper for the node than for the neighbour.
    Ratio uses the quotient of the two weights. Difference uses their subtraction.

    Args:
        metric: ``"ratio"`` or ``"diff"``.
        at_node: Weight of the edge at the node.
        at_neighbour: Weight of the edge at the neighbour.

    Returns:
        _MetricKey: The quotient, or the difference when ``metric`` is ``"diff"``.
    """
    if metric == "ratio":
        return _ratio(at_node, at_neighbour)
    return Fraction(at_node) - Fraction(at_neighbour)


@dataclass(frozen=True)
class _Rank:
    """Order of the nodes and edges taken from the weighted graph.

    A smaller node rank is handled sooner. An edge key orders an edge by the
    ranks of its ends and the weights at those ends.

    Args:
        node: Rank of each node.
        edge: Sort key of each edge id.
    """

    node: dict[int, int]
    edge: dict[int, tuple]


def _rank_graph(instance: Instance) -> _Rank:
    """Order nodes and edges from their weights and their connections.

    Refinement gives every node a signature from its incident weights and the
    signatures of its neighbours, and repeats until the signatures settle.
    While a signature is shared, one of those nodes is given a signature of
    its own and refinement runs again. The order that is kept is the one
    whose list of edges, written from earlier nodes to later nodes, is the
    greatest. An edge's key puts its earlier end first.

    Args:
        instance: Graph being oriented.

    Returns:
        _Rank: Rank of every node and a sort key for every edge.
    """
    nodes = list(instance.nodes)
    locate = {node: index for index, node in enumerate(nodes)}
    adjacent: list[list[tuple[int, Fraction, Fraction]]] = [[] for _ in nodes]
    for edge in instance.edges:
        left = locate[edge.u]
        right = locate[edge.v]
        adjacent[left].append((right, Fraction(edge.w_u), Fraction(edge.w_v)))
        adjacent[right].append((left, Fraction(edge.w_v), Fraction(edge.w_u)))

    def refine(color: list[int]) -> list[int]:
        """Replace each colour by the colours and weights around that node.

        Args:
            color: Current colour of each node, in ``nodes`` order.

        Returns:
            list[int]: Stable colouring. Equal colours mark nodes that still tie.
        """
        current = color
        while True:
            signature = []
            for index, tint in enumerate(current):
                links = tuple(
                    sorted(
                        (here, there, current[other])
                        for other, here, there in adjacent[index]
                    )
                )
                signature.append((tint, links))
            compact = {item: rank for rank, item in enumerate(sorted(set(signature)))}
            nxt = [compact[item] for item in signature]
            if nxt == current:
                return current
            current = nxt

    def certificate(order: list[int]) -> tuple:
        """Edges written from earlier nodes toward later nodes.

        Args:
            order: Nodes, earliest first. Colours in this order are discrete.

        Returns:
            tuple: One sorted row of ``(later slot, weight here, weight there)``
            for each node.
        """
        place = {node: index for index, node in enumerate(order)}
        rows = []
        for node in order:
            links = []
            for other, here, there in adjacent[locate[node]]:
                slot = place[nodes[other]]
                if slot > place[node]:
                    links.append((slot, here, there))
            links.sort()
            rows.append(tuple(links))
        return tuple(rows)

    best: tuple | None = None
    best_order: list[int] | None = None

    def consider(color: list[int]) -> None:
        """Keep the discrete order with the greatest edge certificate.

        Args:
            color: Colouring to refine. A shared colour is split one node at a time.
        """
        nonlocal best, best_order
        color = refine(color)
        groups: dict[int, list[int]] = defaultdict(list)
        for index, tint in enumerate(color):
            groups[tint].append(nodes[index])
        tied = [(tint, members) for tint, members in groups.items() if len(members) > 1]
        if not tied:
            order = sorted(nodes, key=lambda node: color[locate[node]])
            cert = certificate(order)
            if best is None or cert > best:
                best = cert
                best_order = order
            return
        _tint, members = min(tied, key=lambda item: (len(item[1]), item[0]))
        for chosen in members:
            nxt = color[:]
            nxt[locate[chosen]] = max(color) + 1
            consider(nxt)

    if nodes:
        consider([0] * len(nodes))
    order = [] if best_order is None else best_order
    node_rank = {node: index for index, node in enumerate(order)}
    edge_rank: dict[int, tuple] = {}
    for edge in instance.edges:
        left = node_rank[edge.u]
        right = node_rank[edge.v]
        at_left = Fraction(edge.w_u)
        at_right = Fraction(edge.w_v)
        if (right, at_right) < (left, at_left):
            edge_rank[edge.id] = (right, left, at_right, at_left)
        else:
            edge_rank[edge.id] = (left, right, at_left, at_right)
    return _Rank(node_rank, edge_rank)


class ThroughStep(Step):
    """Bisect a threshold and keep the best orientation one through trial finds.

    ``metric`` is ``"ratio"`` or ``"diff"``. ``prune`` peels leaves before
    scoring. ``order`` chooses whose turn comes first.

    Args:
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when pendant bundles are peeled before scoring.
        order: One of :data:`ORDERS`. ``"quota"`` spends bundle flow. ``"topo"`` skips the flow.
        tolerance: Width at which a fractional threshold search stops.
    """

    def __init__(
        self,
        metric: str = "ratio",
        prune: bool = True,
        order: str = "flow-desc",
        tolerance: float = 1e-7,
    ):
        """Store the metric, the leaf switch, the turn order, and the stopping width.

        Args:
            metric: ``"ratio"`` or ``"diff"``.
            prune: True when pendant bundles are peeled before scoring.
            order: Whose turn comes first.
            tolerance: Width at which a fractional threshold search stops.
        """
        if metric not in METRICS:
            raise ValueError("metric must be 'ratio' or 'diff'")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.metric = metric
        self.prune = prune
        self.order = order
        self.tolerance = tolerance

    def run(self, ctx: SolveContext) -> None:
        """Search thresholds and record the best orientation that respected one.

        The search orients every edge from scratch. The result is written onto
        edges that are not locked.

        Args:
            ctx: Solve state that receives the incumbent orientation.
        """
        heads, trials = search_heads(
            ctx.instance,
            metric=self.metric,
            prune=self.prune,
            order=self.order,
            tolerance=self.tolerance,
        )
        for edge_id, head in heads.items():
            if edge_id not in ctx.locked:
                ctx.orientation[edge_id] = head
        ctx.consider(heads)
        ctx.metadata["score_trials"] = trials
        ctx.metadata["score_metric"] = self.metric
        ctx.metadata["score_prune"] = self.prune
        ctx.metadata["score_mode"] = "through"
        ctx.metadata["score_order"] = self.order
        ctx.metadata["score_same_first"] = False


def search_heads(
    instance: Instance,
    metric: str = "ratio",
    prune: bool = False,
    order: str = "flow-desc",
    tolerance: float = 1e-7,
) -> tuple[dict[int, int], int]:
    """Return the best orientation found by halving the threshold range.

    Whole-number weights use whole-number thresholds. Other weights stop when
    the open range is at most ``tolerance``. Each success is measured by its
    makespan. The search keeps the smaller one.

    Args:
        instance: Graph to orient.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when leaves are peeled before scoring.
        order: Whose turn comes first.
        tolerance: Width at which a fractional search stops.

    Returns:
        tuple[dict[int, int], int]: The end each edge points to, and the trial count.

    Raises:
        RuntimeError: No trial produced an orientation.
        ValueError: ``metric`` or ``order`` is unknown.
    """
    if metric not in METRICS:
        raise ValueError("metric must be 'ratio' or 'diff'")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if instance.weights_are_integral():
        return _search_integral(instance, metric, prune, order)
    return _search_real(instance, metric, prune, order, tolerance)


def attempt(
    instance: Instance,
    threshold: float,
    metric: str = "ratio",
    prune: bool = False,
    order: str = "flow-desc",
) -> dict[int, int] | None:
    """Orient the graph at one threshold, or return ``None``.

    When pruning is on, peel every leaf before a node may take a group of
    edges. The peel scan repeats, so a parent that drops to one neighbour is
    peeled before absorb runs. Absorb then gives a remaining node every free
    edge when the group fits. If that creates a leaf, peel again. Without
    pruning, absorb once. Score whatever is still free. ``quota`` claims
    incoming bundles up to the flow on each one. ``topo`` walks the score
    DAG from upstream to downstream and keeps a
    leaving edge while it fits. The other orders do that keep-or-give in the
    chosen flow order. An edge that is still free, or a load above the
    threshold, rejects the trial.

    Args:
        instance: Graph to orient at this threshold.
        threshold: Candidate makespan ``T``.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when leaves are peeled before scoring.
        order: Whose turn comes first.

    Returns:
        dict[int, int] | None: The end each edge points to, or ``None`` when the trial fails.

    Raises:
        ValueError: ``metric`` or ``order`` is unknown.
    """
    if metric not in METRICS:
        raise ValueError("metric must be 'ratio' or 'diff'")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    owned = _trace_depth == 0
    _push_trace()
    try:
        locked: dict[int, int] = {}
        load = {node: 0.0 for node in instance.nodes}
        orientation: dict[int, int] = {}
        free = {edge.id for edge in instance.edges}
        rank = _rank_graph(instance)
        _prepare_trace(instance)
        _show_state("start", instance, load, orientation, free)
        # _absorb(instance, threshold, load, orientation, free, rank)
        if prune:
            # Peel while a leaf exists. Absorb only sees a graph with no leaves,
            # so a node cannot take the edge of a parent that just became a leaf.
            while True:
                if not _peel(
                    instance, threshold, load, orientation, free, metric, rank
                ):
                    _emit("trial rejected: peel")
                    return None
                pending = len(free)
                # _absorb(instance, threshold, load, orientation, free, rank)
                if len(free) == pending:
                    break

        _absorb(instance, threshold, load, orientation, free, rank)
        if free:
            score, out_load, in_load = _scores(instance, free, metric)
            if DEBUG:
                active = sorted(score)
                _show_state(
                    "scores",
                    instance,
                    load,
                    orientation,
                    free,
                    extra=[
                        _map_line("in-load", in_load, active),
                        _map_line("out-load", out_load, active),
                        _map_line("score", score, active),
                    ],
                )
            if order == "quota":
                placed = _assign_quota(
                    instance,
                    threshold,
                    load,
                    orientation,
                    free,
                    metric,
                    score,
                    out_load,
                    rank,
                )
            elif order == "topo":
                placed = _assign_topo(
                    instance, threshold, load, orientation, free, metric, score, rank
                )
            else:
                placed = _assign(
                    instance,
                    threshold,
                    load,
                    orientation,
                    free,
                    metric,
                    score,
                    out_load,
                    order,
                    rank,
                )
            if not placed:
                _emit("trial rejected: assign")
                return None
            if free:
                _show_state("edges still free", instance, load, orientation, free)
                _emit("trial rejected: edges still free")
                return None
        if any(not nearly_leq(value, threshold) for value in load.values()):
            _show_state("load above threshold", instance, load, orientation, free)
            _emit("trial rejected: load above threshold")
            return None
        _show_state("trial finished", instance, load, orientation, free)
        return _complete(instance, locked, orientation)
    finally:
        _pop_trace(owned)


def _search_integral(
    instance: Instance,
    metric: str,
    prune: bool,
    order: str,
) -> tuple[dict[int, int], int]:
    """Halve a whole-number range. Success lowers the high end. Failure raises the low end.

    Args:
        instance: Graph to orient.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when leaves are peeled before scoring.
        order: Whose turn comes first.

    Returns:
        tuple[dict[int, int], int]: Best orientation and the number of trials.

    Raises:
        RuntimeError: No whole-number threshold produced an orientation.
    """
    edges = [edge for edge in instance.edges]
    low = max(min(edge.w_u, edge.w_v) for edge in edges)
    high = sum(max(edge.w_u, edge.w_v) for edge in edges)
    if low > high:
        low = high
    best_orientation: dict[int, int] | None = None
    best_makespan: float | None = None
    trials = 0
    _push_trace()
    try:
        while low <= high:
            threshold = (low + high) // 2
            trials += 1
            _print_trial(trials, threshold, metric, prune, order)
            orientation = attempt(instance, float(threshold), metric, prune, order)
            if orientation is None:
                low = threshold + 1
                continue
            measured = makespan(instance, orientation)
            _emit(f"trial accepted, makespan {_fmt_num(measured)}")
            if best_makespan is None or measured < best_makespan:
                best_makespan = measured
                best_orientation = orientation
            high = threshold - 1
    finally:
        _pop_trace(True)
    if best_orientation is None:
        raise RuntimeError("score-through found no orientation")
    return best_orientation, trials


def _search_real(
    instance: Instance,
    metric: str,
    prune: bool,
    order: str,
    tolerance: float,
) -> tuple[dict[int, int], int]:
    """Halve a real range down to ``tolerance``, after a trial at the high end.

    Args:
        instance: Graph to orient.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when leaves are peeled before scoring.
        order: Whose turn comes first.
        tolerance: Width at which the search stops.

    Returns:
        tuple[dict[int, int], int]: Best orientation and the number of trials.

    Raises:
        RuntimeError: The trial at the high end produced no orientation.
    """
    edges = [edge for edge in instance.edges]
    low = max(min(edge.w_u, edge.w_v) for edge in edges)
    high = sum(max(edge.w_u, edge.w_v) for edge in edges)
    trials = 1
    _push_trace()
    try:
        _print_trial(trials, high, metric, prune, order)
        orientation = attempt(instance, high, metric, prune, order)
        if orientation is None:
            raise RuntimeError(
                "score-through found no orientation at the weight ceiling"
            )
        best_orientation = orientation
        best_makespan = makespan(instance, orientation)
        _emit(f"trial accepted, makespan {_fmt_num(best_makespan)}")
        while high - low > tolerance:
            threshold = (low + high) / 2.0
            if threshold <= low or threshold >= high:
                break
            trials += 1
            _print_trial(trials, threshold, metric, prune, order)
            orientation = attempt(instance, threshold, metric, prune, order)
            if orientation is None:
                low = threshold
                continue
            measured = makespan(instance, orientation)
            _emit(f"trial accepted, makespan {_fmt_num(measured)}")
            if measured < best_makespan:
                best_makespan = measured
                best_orientation = orientation
            high = threshold
    finally:
        _pop_trace(True)
    return best_orientation, trials


def _absorb(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    rank: _Rank,
) -> None:
    """Give a node all of its free edges when the group fits under ``T``.

    Nodes are scanned in canonical order. A group that fits is pointed at once.
    The scan repeats until a full pass points nothing.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset. Pointed edges are removed.
        rank: Canonical order of the nodes.
    """
    pointed_any = False
    changed = True
    while changed:
        changed = False
        for node in sorted(instance.nodes, key=lambda item: rank.node[item]):
            touching = [edge for edge in instance.incident(node) if edge.id in free]
            if not touching:
                continue
            total = sum(edge.weight_toward(node) for edge in touching)
            if not nearly_leq(load[node] + total, threshold):
                continue
            for edge in touching:
                _point(edge, node, load, orientation, free)
                _show_state(
                    f"absorb {node} takes {edge.id} "
                    f"(weight {_fmt_num(edge.weight_toward(node))})",
                    instance,
                    load,
                    orientation,
                    free,
                )
            changed = True
            pointed_any = True
    if not pointed_any:
        _emit("[absorb] nothing fit")


def _peel(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    metric: str,
    rank: _Rank,
) -> bool:
    """Remove leaves until none remain. Fail when a peel puts a load above ``T``.

    A leaf keeps edges while they fit. The metric from the leaf to its parent
    sets the order: a smaller key comes first, and an equal key uses the
    edge's canonical key. The other edges go to the parent. The scan repeats.
    A parent later in the canonical order is pruned in the same scan. A parent
    already passed is pruned when the scan repeats.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset. Peeled edges are removed.
        metric: ``"ratio"`` or ``"diff"``.
        rank: Canonical order of the nodes and edges.

    Returns:
        bool: False when the leaf or its parent ends above ``T``.
    """
    peeled_any = False
    changed = True
    while changed:
        changed = False
        for node in sorted(instance.nodes, key=lambda item: rank.node[item]):
            neighbours = list(_free_neighbours(instance, node, free))
            if len(neighbours) != 1:
                continue
            parent = neighbours[0]
            bundle = [
                edge
                for edge in instance.edges
                if edge.id in free and node in edge.endpoints()
            ]
            bundle.sort(
                key=lambda edge: (
                    _edge_key(
                        metric,
                        edge.weight_toward(node),
                        edge.weight_toward(parent),
                    ),
                    rank.edge[edge.id],
                )
            )
            for edge in bundle:
                if _fits(load, node, edge, threshold):
                    _point(edge, node, load, orientation, free)
                    _show_state(
                        f"peel {node} keeps {edge.id} "
                        f"(weight {_fmt_num(edge.weight_toward(node))})",
                        instance,
                        load,
                        orientation,
                        free,
                    )
                else:
                    _point(edge, parent, load, orientation, free)
                    _show_state(
                        f"peel {node} gives {edge.id} to {parent} "
                        f"(weight {_fmt_num(edge.weight_toward(parent))})",
                        instance,
                        load,
                        orientation,
                        free,
                    )
            if not nearly_leq(load[parent], threshold) or not nearly_leq(
                load[node], threshold
            ):
                _emit(
                    f"  rejected: load {node}={_fmt_num(load[node])} "
                    f"or parent {parent}={_fmt_num(load[parent])} "
                    f"above {_fmt_num(threshold)}"
                )
                return False
            changed = True
            peeled_any = True
    if not peeled_any:
        _emit("[peel] no leaf")
    return True


def _scores(
    instance: Instance,
    free: set[int],
    metric: str,
) -> tuple[dict[int, _MetricKey], dict[int, float], dict[int, float]]:
    """Score every node that still has a free edge. Point nothing.

    The in-load is the weight of every free edge pointed at the node.
    The out-load is the weight of those edges pointed away.
    The score is the metric of the in-load against the out-load.

    Args:
        instance: Graph being oriented.
        free: Edges that are still unset.
        metric: ``"ratio"`` or ``"diff"``.

    Returns:
        tuple[dict[int, _MetricKey], dict[int, float], dict[int, float]]: Score
        of each active node, that node's out-load, and that node's in-load.
    """
    in_load = {node: 0.0 for node in instance.nodes}
    out_load = {node: 0.0 for node in instance.nodes}
    active: set[int] = set()
    for edge in instance.edges:
        if edge.id not in free:
            continue
        in_load[edge.u] += edge.weight_toward(edge.u)
        in_load[edge.v] += edge.weight_toward(edge.v)
        out_load[edge.u] += edge.weight_toward(edge.v)
        out_load[edge.v] += edge.weight_toward(edge.u)
        active.add(edge.u)
        active.add(edge.v)
    score = {node: _edge_key(metric, in_load[node], out_load[node]) for node in active}
    return score, out_load, in_load


def _assign(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    metric: str,
    score: dict[int, _MetricKey],
    out_load: dict[int, float],
    order: str,
    rank: _Rank,
) -> bool:
    """Point every remaining free edge once, in the chosen turn order.

    A tie uses the larger score, then the earlier canonical place. The current
    node handles only the edges that leave it. It keeps an edge while the edge fits.
    Otherwise it gives the edge to the neighbour. The metric from the current
    node to the neighbour sets the order inside that turn.

    The flow and the room decide whose turn comes first. They do not limit
    how much weight the current node may keep. The room is ``T`` minus the
    load already given to the node. A node with no room and some flow sorts
    as an infinite flow-to-room ratio.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset. Pointed edges are removed.
        metric: ``"ratio"`` or ``"diff"``.
        score: Score of each node that still had a free edge.
        out_load: Out-load of every node.
        order: Whose turn comes first.
        rank: Canonical order of the nodes and edges.

    Returns:
        bool: False when an edge fits on neither the current node nor its neighbour.
    """
    bundles = _direct_bundles(instance, free, score, rank)
    through, _bundle_flow = _flow_through(
        threshold, load, score, out_load, bundles, rank
    )
    room = {node: _room(threshold, load, node) for node in score}
    turns = _turn_order(score, through, room, order, rank)
    if DEBUG:
        _show_state(
            f"assign {order}",
            instance,
            load,
            orientation,
            free,
            extra=[
                *_bundle_lines(bundles),
                _map_line("flow", through, turns),
                _map_line("room", room, turns),
                "turns: " + (" ".join(str(node) for node in turns) or "(none)"),
            ],
        )
    leaving: dict[int, list[tuple]] = defaultdict(list)
    for node, neighbour, _weight_at_neighbour, edges in bundles:
        for edge in edges:
            leaving[node].append((edge, neighbour))
    for current in turns:
        batch = [
            (edge, current, neighbour)
            for edge, neighbour in leaving[current]
            if edge.id in free
        ]
        if not _keep_or_give(
            instance, threshold, load, orientation, free, metric, batch, rank
        ):
            return False
    return True


def _room(threshold: float, load: dict[int, float], node: int) -> float:
    """Spare load under ``T``. A node already above ``T`` has no room.

    Args:
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far.
        node: Node whose spare load is requested.

    Returns:
        float: ``T`` minus the load, or zero when that difference is negative.
    """
    spare = threshold - load[node]
    if spare < 0:
        return 0.0
    return spare


def _flow_room_key(flow: float, room: float) -> tuple[int, Fraction]:
    """Increasing key for the flow divided by the room.

    A node with no room and some flow sorts after every finite ratio. A node
    with neither flow nor room sorts as zero.

    Args:
        flow: Flow through the node.
        room: Spare load under ``T``.

    Returns:
        tuple[int, Fraction]: ``(0, ratio)`` when the room is usable, or
        ``(1, 0)`` when the ratio is treated as infinite.
    """
    if room <= ABS_TOL:
        if flow > ABS_TOL:
            return (1, Fraction(0))
        return (0, Fraction(0))
    return (0, Fraction(flow) / Fraction(room))


def _turn_order(
    score: dict[int, _MetricKey],
    through: dict[int, float],
    room: dict[int, float],
    order: str,
    rank: _Rank,
) -> list[int]:
    """Nodes in the order their turns are taken.

    ``flow-desc`` takes the largest flow first. ``flow-asc`` takes the
    smallest flow first. ``flow-room-asc`` and ``flow-room-desc`` use the
    flow divided by the room. A tie takes the larger score, then the earlier
    canonical place.

    Args:
        score: Score of each active node.
        through: Flow through each active node.
        room: Spare load of each active node.
        order: Whose turn comes first.
        rank: Canonical order of the nodes.

    Returns:
        list[int]: Active nodes, first turn first.
    """

    def turn_key(node: int) -> tuple:
        """Sort key for one node. Smaller key takes its turn sooner.

        Args:
            node: Active node.

        Returns:
            tuple: Flow or flow-over-room, then the negated score, then the place.
        """
        flow = through[node]
        if order == "flow-desc":
            return (-flow, -score[node], rank.node[node])
        if order == "flow-asc":
            return (flow, -score[node], rank.node[node])
        band, ratio = _flow_room_key(flow, room[node])
        if order == "flow-room-asc":
            return (band, ratio, -score[node], rank.node[node])
        return (-band, -ratio, -score[node], rank.node[node])

    return sorted(score, key=turn_key)


def _assign_quota(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    metric: str,
    score: dict[int, _MetricKey],
    out_load: dict[int, float],
    rank: _Rank,
) -> bool:
    """Claim incoming bundles up to the flow each one carries.

    Turns run largest flow-through first, then larger score, then the earlier
    canonical place. A turn claims only bundles that point at the current node.
    A lower-score neighbour claims its own incoming flow on its own turn.
    Incoming bundles are taken largest flow first, then the earlier place of
    the higher-score neighbour.
    Inside a bundle, the metric at the current node against that neighbour
    decides which edge uses the flow first.

    An edge is pointed at the current node while its weight there fits under
    ``T`` and in the flow still left. The edge is atomic. The higher-score
    neighbour keeps it when it does not, if the edge fits there.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset. Pointed edges are removed.
        metric: ``"ratio"`` or ``"diff"``.
        score: Score of each node that still had a free edge.
        out_load: Out-load of every node.
        rank: Canonical order of the nodes and edges.

    Returns:
        bool: False when an edge fits on neither end.
    """
    bundles = _direct_bundles(instance, free, score, rank)
    through, bundle_flow = _flow_through(
        threshold, load, score, out_load, bundles, rank
    )
    room = {node: _room(threshold, load, node) for node in score}
    turns = _turn_order(score, through, room, "flow-desc", rank)
    if DEBUG:
        _show_state(
            "assign quota",
            instance,
            load,
            orientation,
            free,
            extra=[
                *_bundle_lines(bundles),
                _map_line("flow", through, turns),
                _map_line("room", room, turns),
                "bundle-flow: "
                + (
                    ", ".join(
                        f"{src}->{dst}:{_fmt_num(amount)}"
                        for (src, dst), amount in sorted(bundle_flow.items())
                    )
                    or "(none)"
                ),
                "turns: " + (" ".join(str(node) for node in turns) or "(none)"),
            ],
        )
    incoming: dict[int, list[tuple]] = defaultdict(list)
    for higher, current, _weight_at_current, edges in bundles:
        incoming[current].append(
            (higher, edges, bundle_flow.get((higher, current), 0.0))
        )
    for current in turns:
        channels = sorted(
            incoming[current],
            key=lambda item: (-item[2], rank.node[item[0]]),
        )
        for higher, edges, quota in channels:
            if not _spend_quota(
                instance,
                threshold,
                load,
                orientation,
                free,
                metric,
                current,
                higher,
                edges,
                quota,
                rank,
            ):
                return False
    return True


def _spend_quota(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    metric: str,
    current: int,
    higher: int,
    edges: list,
    quota: float,
    rank: _Rank,
) -> bool:
    """Point edges at ``current`` until the bundle's flow is spent.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset.
        metric: ``"ratio"`` or ``"diff"``.
        current: Lower-score end that receives the flow.
        higher: Higher-score neighbour that the bundle comes from.
        edges: Free edges in that bundle.
        quota: Flow on the bundle, measured as weight at ``current``.
        rank: Canonical order of the edges.

    Returns:
        bool: False when an edge fits on neither end.
    """
    ordered = sorted(
        edges,
        key=lambda edge: (
            _edge_key(
                metric,
                edge.weight_toward(current),
                edge.weight_toward(higher),
            ),
            rank.edge[edge.id],
        ),
    )
    remaining = quota
    for edge in ordered:
        if edge.id not in free:
            continue
        weight = edge.weight_toward(current)
        if _fits(load, current, edge, threshold) and nearly_leq(weight, remaining):
            _point(edge, current, load, orientation, free)
            remaining -= weight
            _show_state(
                f"quota {current} takes {edge.id} from {higher} "
                f"(weight {_fmt_num(weight)}, remaining {_fmt_num(remaining)})",
                instance,
                load,
                orientation,
                free,
            )
        elif _fits(load, higher, edge, threshold):
            _point(edge, higher, load, orientation, free)
            _show_state(
                f"quota {higher} keeps {edge.id} "
                f"(weight {_fmt_num(edge.weight_toward(higher))})",
                instance,
                load,
                orientation,
                free,
            )
        else:
            _show_state(
                f"quota cannot place {edge.id} between {current} and {higher}",
                instance,
                load,
                orientation,
                free,
            )
            return False
    return True


def _assign_topo(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    metric: str,
    score: dict[int, _MetricKey],
    rank: _Rank,
) -> bool:
    """Point leaving edges in topological order. Do not push a flow.

    A node takes its turn before the neighbours its bundles point to. The turn
    is the usual keep-or-give: the current node keeps a leaving edge while it
    fits, and otherwise gives the edge to that neighbour.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset. Pointed edges are removed.
        metric: ``"ratio"`` or ``"diff"``.
        score: Score of each node that still had a free edge.
        rank: Canonical order of the nodes and edges.

    Returns:
        bool: False when an edge fits on neither end.
    """
    bundles = _direct_bundles(instance, free, score, rank)
    turns = _topo_turns(score, bundles, rank)
    if DEBUG:
        _show_state(
            "assign topo",
            instance,
            load,
            orientation,
            free,
            extra=[
                *_bundle_lines(bundles),
                "turns: " + (" ".join(str(node) for node in turns) or "(none)"),
            ],
        )
    leaving: dict[int, list[tuple]] = defaultdict(list)
    for node, neighbour, _weight_at_neighbour, edges in bundles:
        for edge in edges:
            leaving[node].append((edge, neighbour))
    for current in turns:
        batch = [
            (edge, current, neighbour)
            for edge, neighbour in leaving[current]
            if edge.id in free
        ]
        if not _keep_or_give(
            instance, threshold, load, orientation, free, metric, batch, rank
        ):
            return False
    return True


def _topo_turns(
    score: dict[int, _MetricKey],
    bundles: list[tuple[int, int, float, list]],
    rank: _Rank,
) -> list[int]:
    """Kahn order of the score DAG. A node comes before the neighbours it points to.

    When several nodes are ready, the larger score goes first, then the earlier
    canonical place. A node left by a cycle is appended in that same order.

    Args:
        score: Score of each active node.
        bundles: Directed bundles, ``(node, neighbour, weight at the neighbour, edges)``.
        rank: Canonical order of the nodes.

    Returns:
        list[int]: Active nodes, first turn first.
    """
    indeg = {node: 0 for node in score}
    succ: dict[int, list[int]] = {node: [] for node in score}
    for node, neighbour, _weight, _edges in bundles:
        if node not in indeg or neighbour not in indeg or node == neighbour:
            continue
        indeg[neighbour] += 1
        succ[node].append(neighbour)
    ready = [node for node in score if indeg[node] == 0]
    turns: list[int] = []
    seen: set[int] = set()
    while ready:
        current = min(ready, key=lambda node: (-score[node], rank.node[node]))
        ready.remove(current)
        turns.append(current)
        seen.add(current)
        for neighbour in succ[current]:
            indeg[neighbour] -= 1
            if indeg[neighbour] == 0 and neighbour not in seen:
                ready.append(neighbour)
    if len(turns) < len(score):
        rest = [node for node in score if node not in seen]
        rest.sort(key=lambda node: (-score[node], rank.node[node]))
        turns.extend(rest)
    return turns


def _hop_distance(adjacent: dict[int, set[int]], sources: list[int]) -> dict[int, int]:
    """Hop distance from a set of nodes, along free edges.

    Args:
        adjacent: Free neighbours of each scored node.
        sources: Nodes that sit at distance zero.

    Returns:
        dict[int, int]: Hop distance of every node reached from ``sources``.
    """
    distance = {node: 0 for node in sources}
    queue: deque[int] = deque(sources)
    while queue:
        node = queue.popleft()
        for neighbour in adjacent[node]:
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance


def _score_progress(
    instance: Instance,
    free: set[int],
    score: dict[int, _MetricKey],
) -> dict[int, tuple[int, int]]:
    """Hop distances from each free component's highest and lowest score.

    A connected piece of the free graph uses its own extremes. Every node that
    holds the highest score in its piece is a source at distance zero, and
    every node that holds the lowest score is a sink at distance zero.

    Args:
        instance: Graph being oriented.
        free: Edges that are still unset.
        score: Score of each active node.

    Returns:
        dict[int, tuple[int, int]]: ``(hops from the high score, hops from the
        low score)`` for every scored node.
    """
    adjacent: dict[int, set[int]] = {node: set() for node in score}
    for edge in instance.edges:
        if edge.id not in free:
            continue
        left, right = edge.endpoints()
        if left == right or left not in adjacent or right not in adjacent:
            continue
        adjacent[left].add(right)
        adjacent[right].add(left)
    progress: dict[int, tuple[int, int]] = {}
    seen: set[int] = set()
    for start in score:
        if start in seen:
            continue
        component: list[int] = []
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in adjacent[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        highest = max(score[node] for node in component)
        lowest = min(score[node] for node in component)
        from_high = _hop_distance(
            adjacent, [node for node in component if score[node] == highest]
        )
        from_low = _hop_distance(
            adjacent, [node for node in component if score[node] == lowest]
        )
        for node in component:
            progress[node] = (from_high[node], from_low[node])
    return progress


def _direct_bundles(
    instance: Instance,
    free: set[int],
    score: dict[int, _MetricKey],
    rank: _Rank,
) -> list[tuple[int, int, float, list]]:
    """Give every remaining pair one direction.

    The direction goes from the higher score to the lower score. Equal scores
    go from the smaller progress fraction toward the larger one, and an equal
    fraction goes from the earlier canonical place to the later place. The
    first node is the one that will try to keep the edges. The second is its
    neighbour. The weight is the total weight of the bundle at the neighbour.

    Args:
        instance: Graph being oriented.
        free: Edges that are still unset.
        score: Score of each active node.
        rank: Canonical order of the nodes and edges.

    Returns:
        list[tuple[int, int, float, list]]: ``(node, neighbour, weight at the neighbour, edges)``.
    """
    progress = _score_progress(instance, free, score)
    edges_between: dict[tuple[int, int], list] = defaultdict(list)
    for edge in instance.edges:
        if edge.id not in free:
            continue
        edges_between[tuple(sorted(edge.endpoints()))].append(edge)
    bundles = []
    for (left, right), edges in edges_between.items():
        edges.sort(key=lambda edge: rank.edge[edge.id])
        if _tries_first(score, left, right, rank, progress):
            node, neighbour = left, right
        else:
            node, neighbour = right, left
        weight_at_neighbour = sum(edge.weight_toward(neighbour) for edge in edges)
        bundles.append((node, neighbour, weight_at_neighbour, edges))
    bundles.sort(
        key=lambda item: (
            rank.node[item[0]],
            rank.node[item[1]],
            tuple(rank.edge[edge.id] for edge in item[3]),
        )
    )
    return bundles


def _tries_first(
    score: dict[int, _MetricKey],
    left: int,
    right: int,
    rank: _Rank,
    progress: dict[int, tuple[int, int]],
) -> bool:
    """True when ``left`` keeps an edge before ``right`` does.

    The higher score tries first. Equal scores let the smaller progress
    fraction try first. The fraction is the hop distance from the component's
    highest score divided by the sum of the distances from its highest and
    lowest scores. An equal fraction lets the earlier place try first.

    Args:
        score: Score of each active node.
        left: One end of a bundle.
        right: The other end of that bundle.
        rank: Canonical order of the nodes.
        progress: ``(hops from the high score, hops from the low score)`` of
            each scored node.

    Returns:
        bool: True when ``left`` is the node whose turn owns the bundle.
    """
    if score[left] != score[right]:
        return score[left] > score[right]
    high_left, low_left = progress[left]
    high_right, low_right = progress[right]
    # t(left) < t(right) exactly when this product is smaller.
    if high_left * low_right != high_right * low_left:
        return high_left * low_right < high_right * low_left
    return rank.node[left] < rank.node[right]


def _flow_through(
    threshold: float,
    load: dict[int, float],
    score: dict[int, _MetricKey],
    out_load: dict[int, float],
    bundles: list[tuple[int, int, float, list]],
    rank: _Rank,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Rank each scored node by the flow that passes through it. Point nothing.

    Each scored node is split into an entry and an exit. The arc between them
    holds the room under ``T``, which is ``T`` minus the node's load. A bundle
    runs from the exit of the node that tries first to the entry of its
    neighbour, and holds the weight of the bundle at the neighbour.

    A node with no bundle coming in is fed at its exit, up to its out-load.
    A node with no bundle leaving it sends its exit on to the sink, up to its
    room. The flow through the node is the flow on its entry-exit arc plus
    the flow on its feed. The feed is part of the rank because that node is
    fed at the exit, so the entry-exit arc alone would read zero. The flow on
    a bundle arc is its capacity minus the room left after the push.

    Args:
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far.
        score: Score of each active node. The keys are the nodes that are ranked.
        out_load: Out-load of every node.
        bundles: Directed bundles, ``(node, neighbour, weight at the neighbour, edges)``.
        rank: Canonical order of the nodes. Arcs are inserted in that order.

    Returns:
        tuple[dict[int, float], dict[tuple[int, int], float]]: Flow through each
        scored node, and the flow on each bundle ``(node, neighbour)``.
    """
    nodes = sorted(score, key=lambda node: rank.node[node])
    bundles = sorted(
        bundles,
        key=lambda item: (
            rank.node[item[0]],
            rank.node[item[1]],
            tuple(rank.edge[edge.id] for edge in item[3]),
        ),
    )
    network: list[list[list]] = [[] for _ in range(2 + 2 * len(nodes))]
    entry_of = {node: 2 + 2 * index for index, node in enumerate(nodes)}
    exit_of = {node: 3 + 2 * index for index, node in enumerate(nodes)}
    bundles_in = {node: 0 for node in nodes}
    bundles_out = {node: 0 for node in nodes}
    bundle_arcs: list[tuple[int, int, list | None, float]] = []
    for node, neighbour, weight_at_neighbour, _edges in bundles:
        bundles_in[neighbour] += 1
        bundles_out[node] += 1
        bundle_arcs.append(
            (
                node,
                neighbour,
                _add_arc(
                    network, exit_of[node], entry_of[neighbour], weight_at_neighbour
                ),
                weight_at_neighbour,
            )
        )
    entry_exit: list[tuple[int, list | None, float]] = []
    feeds: list[tuple[int, list | None, float]] = []
    for node in nodes:
        room = threshold - load[node]
        if room < 0:
            room = 0.0
        entry_exit.append(
            (node, _add_arc(network, entry_of[node], exit_of[node], room), room)
        )
        if bundles_in[node] == 0:
            feed = out_load.get(node, 0.0)
            feeds.append((node, _add_arc(network, _SOURCE, exit_of[node], feed), feed))
        if bundles_out[node] == 0:
            _add_arc(network, exit_of[node], _SINK, room)
    _push_flow(network, source=_SOURCE, sink=_SINK)
    through = {node: 0.0 for node in nodes}
    for node, arc, capacity in entry_exit + feeds:
        if arc is not None:
            room_left = arc[1]
            through[node] += max(0.0, capacity - room_left)
    bundle_flow: dict[tuple[int, int], float] = {}
    for node, neighbour, arc, capacity in bundle_arcs:
        if arc is None:
            bundle_flow[(node, neighbour)] = 0.0
        else:
            bundle_flow[(node, neighbour)] = max(0.0, capacity - arc[1])
    return through, bundle_flow


def _keep_or_give(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    metric: str,
    batch: list[tuple],
    rank: _Rank,
) -> bool:
    """Keep each edge on the current node while it fits, otherwise give it to the neighbour.

    The batch is sorted by the metric from the current node to the neighbour,
    then by the edge's canonical key.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset.
        metric: ``"ratio"`` or ``"diff"``.
        batch: ``(edge, current, neighbour)`` triples.
        rank: Canonical order of the edges.

    Returns:
        bool: False when an edge fits on neither end.
    """
    ordered = sorted(batch, key=lambda choice: _choice_key(metric, choice, rank))
    for edge, current, neighbour in ordered:
        if edge.id not in free:
            continue
        if _fits(load, current, edge, threshold):
            _point(edge, current, load, orientation, free)
            _show_state(
                f"turn {current} keeps {edge.id} "
                f"(weight {_fmt_num(edge.weight_toward(current))})",
                instance,
                load,
                orientation,
                free,
            )
        elif _fits(load, neighbour, edge, threshold):
            _point(edge, neighbour, load, orientation, free)
            _show_state(
                f"turn {current} gives {edge.id} to {neighbour} "
                f"(weight {_fmt_num(edge.weight_toward(neighbour))})",
                instance,
                load,
                orientation,
                free,
            )
        else:
            _show_state(
                f"turn {current} cannot place {edge.id}",
                instance,
                load,
                orientation,
                free,
            )
            return False
    return True


def _choice_key(metric: str, choice: tuple, rank: _Rank) -> tuple[_MetricKey, tuple]:
    """Sort key for one edge on the current node's turn.

    Args:
        metric: ``"ratio"`` or ``"diff"``.
        choice: ``(edge, current, neighbour)``.
        rank: Canonical order of the edges.

    Returns:
        tuple[_MetricKey, tuple]: Metric from the current node to the neighbour,
        then the edge's canonical key.
    """
    edge, current, neighbour = choice
    return (
        _edge_key(
            metric,
            edge.weight_toward(current),
            edge.weight_toward(neighbour),
        ),
        rank.edge[edge.id],
    )


def _add_arc(
    network: list[list[list]], start: int, end: int, capacity: float
) -> list | None:
    """Add an arc when ``capacity`` is positive, plus a reverse arc that starts empty.

    Each arc is ``[end, room, index of the reverse arc]``. The reverse arc is
    how a later pass can undo a send.

    Args:
        network: Adjacency of the flow network. Updated in place.
        start: Where this arc begins.
        end: Where this arc ends.
        capacity: How much flow the arc can carry.

    Returns:
        list | None: The forward arc, or ``None`` when ``capacity`` is negligible.
    """
    if capacity <= ABS_TOL:
        return None
    forward = [end, float(capacity), len(network[end])]
    backward = [start, 0.0, len(network[start])]
    network[start].append(forward)
    network[end].append(backward)
    return forward


def _push_flow(network: list[list[list]], source: int, sink: int) -> float:
    """Send as much flow as the network allows, along the shortest path that still has room.

    Each pass finds a source-to-sink path with room on every arc and the fewest
    arcs. The path carries the smallest room on its arcs. That room moves to
    the reverse arcs, so a later pass can send flow back.

    Args:
        network: Adjacency of the flow network. Room is updated in place.
        source: Where flow enters.
        sink: Where flow leaves.

    Returns:
        float: Total flow pushed from ``source`` to ``sink``.
    """
    sent = 0.0
    while True:
        came_from = [-1] * len(network)
        arc_used = [-1] * len(network)
        came_from[source] = source
        queue: deque[int] = deque([source])
        while queue and came_from[sink] == -1:
            node = queue.popleft()
            for index, arc in enumerate(network[node]):
                end, room, _reverse = arc
                if room > ABS_TOL and came_from[end] == -1:
                    came_from[end] = node
                    arc_used[end] = index
                    queue.append(end)
        if came_from[sink] == -1:
            break
        amount = inf
        node = sink
        while node != source:
            previous = came_from[node]
            amount = min(amount, network[previous][arc_used[node]][1])
            node = previous
        if amount <= ABS_TOL:
            break
        node = sink
        while node != source:
            previous = came_from[node]
            arc = network[previous][arc_used[node]]
            arc[1] -= amount
            network[node][arc[2]][1] += amount
            node = previous
        sent += amount
    return sent


def _free_neighbours(instance: Instance, node: int, free: set[int]) -> set[int]:
    """Neighbours reached from ``node`` by an edge that is still free.

    Args:
        instance: Graph being oriented.
        node: Node whose free neighbourhood is requested.
        free: Edges that have not been pointed yet.

    Returns:
        set[int]: Distinct free neighbours of ``node``.
    """
    found = set()
    for edge in instance.incident(node):
        if edge.id in free:
            found.add(edge.other(node))
    return found


def _fits(load: dict[int, float], node: int, edge, threshold: float) -> bool:
    """True when the edge can be pointed at ``node`` without passing ``T``.

    Args:
        load: Weight given to each node so far.
        node: End the edge would point to.
        edge: Edge being tested.
        threshold: Candidate makespan ``T``.

    Returns:
        bool: True when the resulting load is at most ``T``.
    """
    return nearly_leq(load[node] + edge.weight_toward(node), threshold)


def _push_trace() -> None:
    """Hold the trace so a nested trial does not write it early."""
    global _trace_depth
    if not DEBUG:
        return
    _trace_depth += 1


def _pop_trace(flush: bool) -> None:
    """Release one hold. Write the trace when ``flush`` is true.

    Args:
        flush: True when this caller owns the trace and should write it.
    """
    global _trace_depth
    if not DEBUG:
        return
    _trace_depth -= 1
    if flush:
        _flush_trace()


def _emit(line: str) -> None:
    """Append one trace line. The console is written later, in one piece.

    Args:
        line: One line, without a trailing newline.
    """
    if DEBUG:
        _trace.append(line)


def _flush_trace() -> None:
    """Write every buffered line and empty the buffer."""
    if not DEBUG or not _trace:
        return
    sys.stdout.write("\n".join(_trace))
    sys.stdout.write("\n")
    _trace.clear()


def _prepare_trace(instance: Instance) -> None:
    """Build the unchanging edge text once for this trial.

    Args:
        instance: Graph being oriented.
    """
    global _free_label, _arrow
    if not DEBUG:
        return
    free_label: dict[int, str] = {}
    arrow: dict[int, str] = {}
    for edge in instance.edges:
        free_label[edge.id] = (
            f"{edge.id}:{edge.u}-{edge.v}"
            f"({_fmt_num(edge.w_u)}/{_fmt_num(edge.w_v)})"
        )
        arrow[edge.id] = f"{edge.id}->"
    _free_label = free_label
    _arrow = arrow


def _print_trial(
    trials: int, threshold: float, metric: str, prune: bool, order: str
) -> None:
    """Buffer the banner for one threshold trial.

    Args:
        trials: How many trials have started, including this one.
        threshold: Candidate makespan ``T``.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when leaves are peeled before scoring.
        order: Whose turn comes first.
    """
    if not DEBUG:
        return
    _emit("_" * 80)
    _emit(
        f"trial {trials}, threshold {_fmt_num(threshold)}, "
        f"metric {metric}, prune {prune}, order {order}"
    )
    _emit("_" * 80)


def _fmt_num(value: float) -> str:
    """Compact number. Whole values are printed without a decimal.

    Args:
        value: Load, weight, flow, or threshold.

    Returns:
        str: ``value`` as an integer when it is integral, otherwise a short float.
    """
    nearest = round(value)
    if abs(value - nearest) <= ABS_TOL:
        return str(int(nearest))
    return f"{value:.6g}"


def _fmt_value(value) -> str:
    """Print a load as a number and a score as its exact value.

    Args:
        value: A load, a flow, or a score.

    Returns:
        str: A compact number, or the score unchanged when it is not a float.
    """
    if isinstance(value, (int, float)):
        return _fmt_num(value)
    return str(value)


def _map_line(label: str, values: dict, nodes) -> str:
    """One labeled row of ``node:value`` pairs.

    Args:
        label: Name of the row.
        values: Number or score stored for each node.
        nodes: Nodes to print, in that order.

    Returns:
        str: ``label`` followed by the pairs.
    """
    body = "  ".join(f"{node}:{_fmt_value(values[node])}" for node in nodes)
    return f"{label}: {body}"


def _bundle_lines(bundles: list[tuple[int, int, float, list]]) -> list[str]:
    """One row per directed bundle.

    Args:
        bundles: Directed bundles, ``(node, neighbour, weight at the neighbour, edges)``.

    Returns:
        list[str]: A row for each bundle, or one row when there are none.
    """
    lines = []
    for node, neighbour, weight, edges in bundles:
        ids = ",".join(str(edge.id) for edge in edges)
        lines.append(
            f"bundle {node}->{neighbour} weight {_fmt_num(weight)} edges [{ids}]"
        )
    if not lines:
        return ["bundle: (none)"]
    return lines


def _show_state(
    note: str,
    instance: Instance,
    load: dict[int, float],
    orientation: dict[int, int],
    free: set[int],
    extra: list[str] | None = None,
) -> None:
    """Buffer loads, free edges, the orientation so far, and any extra rows.

    Args:
        note: What just happened.
        instance: Graph being oriented.
        load: Weight given to each node so far.
        orientation: Chosen end of each pointed edge.
        free: Edges that are still unset.
        extra: Optional rows printed after the orientation.
    """
    if not DEBUG:
        return
    lines = [f"[{note}]", "  " + _map_line("load", load, instance.nodes)]
    if free:
        lines.append(
            "  free: "
            + ", ".join(
                _free_label[edge.id] for edge in instance.edges if edge.id in free
            )
        )
    else:
        lines.append("  free: (none)")
    if orientation:
        lines.append(
            "  oriented: "
            + ", ".join(
                _arrow[edge.id] + str(orientation[edge.id])
                for edge in instance.edges
                if edge.id in orientation
            )
        )
    else:
        lines.append("  oriented: (none)")
    if extra:
        lines.extend(f"  {line}" for line in extra)
    _emit("\n".join(lines))


def _point(
    edge, end: int, load: dict[int, float], orientation: dict[int, int], free: set[int]
) -> None:
    """Point ``edge`` at ``end`` and add its weight there.

    Args:
        edge: Edge being pointed.
        end: End the edge points to.
        load: Weight given to each node so far. Updated in place.
        orientation: Chosen end of each pointed edge. Updated in place.
        free: Edges that are still unset. ``edge`` is removed.
    """
    orientation[edge.id] = end
    load[end] += edge.weight_toward(end)
    free.discard(edge.id)


def _complete(
    instance: Instance,
    locked: dict[int, int],
    orientation: dict[int, int],
) -> dict[int, int]:
    """Copy the chosen end of every edge.

    The trial's choice wins. ``locked`` fills an edge the trial did not write.

    Args:
        instance: Graph whose edges must all appear.
        locked: Fallback ends for edges the trial did not write.
        orientation: Ends written by the trial.

    Returns:
        dict[int, int]: The end each edge points to.

    Raises:
        RuntimeError: An edge has no chosen end.
    """
    chosen: dict[int, int] = {}
    for edge in instance.edges:
        end = orientation.get(edge.id, locked.get(edge.id))
        if end is None:
            raise RuntimeError(f"edge {edge.id} was left unoriented")
        chosen[edge.id] = end
    return chosen
