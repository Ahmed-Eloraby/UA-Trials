"""Score-family threshold search.

One trial at a threshold ``T`` absorbs any node that can take every free
incident edge, optionally peels leaves, then scores the edges that remain.
The score is hypothetical: in-load is the weight of pointing every free edge
at the node, and out-load is the weight of pointing them all away. Nothing is
oriented by the score itself.

The metric is the ratio of those loads, or their difference. Lowest metric in
a free component is a sink. Highest is a source. Stage 3 then orients free
edges under ``T``. The sender keeps an edge while it fits, and the receiver
takes it only when the sender cannot. ``inflow`` instead claims from the
largest flow inward, capping each channel by its share of that flow.

The outer search keeps the orientation with the smallest measured makespan
among trials that finished inside their threshold. A rejected trial is not a
proof that ``T`` is infeasible, so it does not raise the stored lower bound.
"""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from math import inf

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import ABS_TOL, Instance, is_integral_number, makespan, nearly_leq

METRICS = ("ratio", "diff")
PRUNE_NAMES = (("peel", True), ("raw", False))
MODES = ("sink", "source", "both", "flow", "topo", "inflow")

_MetricKey = Fraction


def family_names() -> list[str]:
    """Command-line names of the forty-eight score-family algorithms.

    The order is metric, then pruning, then Core mode. Each mode is listed
    twice: the plain name orients same-level edges after the cuts, and the
    ``early`` name orients them first.

    Returns:
        list[str]: ``score-{ratio|diff}-{peel|raw}-[early-]{mode}``.
    """
    names = []
    for metric in METRICS:
        for prune_name, _prune in PRUNE_NAMES:
            for mode in MODES:
                names.append(f"score-{metric}-{prune_name}-{mode}")
                names.append(f"score-{metric}-{prune_name}-early-{mode}")
    return names


def _ratio(numerator: float, denominator: float) -> _MetricKey:
    """Sort key for ``numerator / denominator``, in increasing order.

    Every edge weight is strictly positive, so a scored load and an endpoint
    weight used as a denominator are strictly positive. The ratio is finite.

    Args:
        numerator: Load or weight being compared.
        denominator: Load or weight it is compared with.

    Returns:
        _MetricKey: The ratio.
    """
    return Fraction(numerator) / Fraction(denominator)


def _edge_key(metric: str, toward_sender: float, toward_other: float) -> _MetricKey:
    """Increasing key for one metric between two endpoint weights.

    Args:
        metric: ``"ratio"`` or ``"diff"``.
        toward_sender: Weight or load at the node being filled.
        toward_other: Weight or load at the other endpoint.

    Returns:
        _MetricKey: The ratio, or the difference when ``metric`` is ``"diff"``.
    """
    if metric == "ratio":
        return _ratio(toward_sender, toward_other)
    return Fraction(toward_sender) - Fraction(toward_other)


def _negate(key: _MetricKey) -> _MetricKey:
    """Key that sorts a larger metric first.

    Args:
        key: Increasing metric key.

    Returns:
        _MetricKey: The negation of ``key``.
    """
    return -key


class ScoreStep(Step):
    """Bisect a threshold and keep the best orientation one score trial finds.

    ``metric`` is ``"ratio"`` or ``"diff"``. ``prune`` peels leaves before
    scoring. ``mode`` is ``"sink"``, ``"source"``, ``"both"``, ``"flow"``,
    ``"topo"``, or ``"inflow"``. ``same_first`` orients edges inside one level
    before the level-to-level pass. The default leaves those edges until the end.

    Args:
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when pendant bundles are peeled before scoring.
        mode: Core order.
        same_first: True when same-level edges are oriented before the cuts.
        tolerance: Width at which a fractional threshold search stops.
    """

    def __init__(
        self,
        metric: str = "ratio",
        prune: bool = True,
        mode: str = "sink",
        same_first: bool = False,
        tolerance: float = 1e-7,
    ):
        """Store the metric, the leaf switch, the order, and the stopping width.

        Args:
            metric: ``"ratio"`` or ``"diff"``.
            prune: True when pendant bundles are peeled before scoring.
            mode: Core order.
            same_first: True when same-level edges are oriented before the cuts.
            tolerance: Width at which a fractional threshold search stops.
        """
        if metric not in METRICS:
            raise ValueError("metric must be 'ratio' or 'diff'")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.metric = metric
        self.prune = prune
        self.mode = mode
        self.same_first = same_first
        self.tolerance = tolerance

    def run(self, ctx: SolveContext) -> None:
        """Search thresholds and record the best orientation that respected one.

        Locked edges stay as they are. Their weight is already in the load
        when a trial starts.

        Args:
            ctx: Solve state that receives the incumbent orientation.
        """
        heads, trials = search_heads(
            ctx.instance,
            metric=self.metric,
            prune=self.prune,
            mode=self.mode,
            same_first=self.same_first,
            tolerance=self.tolerance,
        )
        for edge_id, head in heads.items():
            if edge_id not in ctx.locked:
                ctx.orientation[edge_id] = head
        ctx.consider(heads)
        ctx.metadata["score_trials"] = trials
        ctx.metadata["score_metric"] = self.metric
        ctx.metadata["score_prune"] = self.prune
        ctx.metadata["score_mode"] = self.mode
        ctx.metadata["score_same_first"] = self.same_first


def search_heads(
    instance: Instance,
    metric: str = "ratio",
    prune: bool = False,
    mode: str = "sink",
    same_first: bool = False,
    tolerance: float = 1e-7,
) -> tuple[dict[int, int], int]:
    """Return the best orientation found by bisection, and the number of trials.

    Integral weights use the integer search. Other weights bisect until the
    bracket is narrower than ``tolerance``. Among trials that return an
    orientation, the one with the smallest measured makespan is kept.

    Args:
        instance: Graph to orient.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when pendant bundles are peeled before scoring.
        mode: Core order.
        same_first: True when same-level edges are oriented before the cuts.
        tolerance: Width at which a fractional search stops.

    Returns:
        tuple[dict[int, int], int]: Complete orientation, and the trial count.

    Raises:
        RuntimeError: No trial produced an orientation.
        ValueError: ``metric`` or ``mode`` is unknown.
    """
    if instance.weights_are_integral():
        return _search_integral(instance, metric, prune, mode, same_first)
    return _search_real(instance, metric, prune, mode, same_first, tolerance)


def attempt(
    instance: Instance,
    threshold: float,
    metric: str = "ratio",
    prune: bool = False,
    mode: str = "sink",
    same_first: bool = False,
) -> dict[int, int] | None:
    """Try one threshold. Return a complete orientation, or ``None`` on reject.

    The trial may orient same-level edges and then run a leftover pass.

    Args:
        instance: Graph to orient at this threshold.
        threshold: Candidate makespan ``T``.
        locked: Edges already oriented, mapping each such edge id to its head.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when pendant bundles are peeled before scoring.
        mode: Core order.
        same_first: True when same-level edges are oriented before the cuts.

    Returns:
        dict[int, int] | None: Complete orientation, or ``None`` when the trial rejects.

    Raises:
        ValueError: ``metric`` or ``mode`` is unknown.
    """
    locked = dict({})
    load = {node: 0.0 for node in instance.nodes}
    assign = dict({})
    free = {edge.id for edge in instance.edges}
    _absorb(instance, threshold, load, assign, free)
    if prune and not _peel(instance, threshold, load, assign, free, metric):
        return None
    if prune:
        _absorb(instance, threshold, load, assign, free)
    if free:
        keys, out_load = _scores(instance, free, metric)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        adj = _adjacency(instance, free)
        dist_sink, dist_source = _distances(adj, keys)
        if same_first and not _same_level(
            instance,
            threshold,
            load,
            assign,
            free,
            metric,
            mode,
            keys,
            dist_sink,
            dist_source,
        ):
            return None
        if same_first:
            _absorb(instance, threshold, load, assign, free)
        if not _core(
            instance,
            threshold,
            load,
            assign,
            free,
            metric,
            mode,
            keys,
            out_load,
            dist_sink,
            dist_source,
        ):
            return None
        _absorb(instance, threshold, load, assign, free)
        if not _leftover(
            instance,
            threshold,
            load,
            assign,
            free,
            metric,
            mode,
            keys,
            dist_sink,
            dist_source,
        ):
            return None
    if any(not nearly_leq(value, threshold) for value in load.values()):
        return None
    return _complete(instance, locked, assign)


def _search_integral(
    instance: Instance,
    metric: str,
    prune: bool,
    mode: str,
    same_first: bool,
) -> tuple[dict[int, int], int]:
    """Integer bisection. A success narrows the ceiling; a reject raises the floor.

    Args:
        instance: Graph to orient.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when pendant bundles are peeled before scoring.
        mode: Core order.
        same_first: True when same-level edges are oriented before the cuts.

    Returns:
        tuple[dict[int, int], int]: Best orientation and the number of trials.

    Raises:
        RuntimeError: No integer threshold produced an orientation.
    """
    free = [edge for edge in instance.edges]
    light = max(min(edge.w_u, edge.w_v) for edge in free)
    ceiling = sum(max(edge.w_u, edge.w_v) for edge in free)
    lo = light
    hi = ceiling
    if lo > hi:
        lo = hi
    best_heads: dict[int, int] | None = None
    best_score: float | None = None
    trials = 0
    while lo <= hi:
        threshold = (lo + hi) // 2
        trials += 1
        found = attempt(instance, float(threshold), metric, prune, mode, same_first)
        if found is None:
            lo = threshold + 1
            continue
        score = makespan(instance, found)
        if best_score is None or score < best_score:
            best_score = score
            best_heads = found
        hi = threshold - 1
    if best_heads is None:
        raise RuntimeError("score-family found no orientation")
    return best_heads, trials


def _search_real(
    instance: Instance,
    metric: str,
    prune: bool,
    mode: str,
    same_first: bool,
    tolerance: float,
) -> tuple[dict[int, int], int]:
    """Real bisection down to ``tolerance``, seeded by a trial at the ceiling.

    Args:
        instance: Graph to orient.
        locked: Edges already oriented, mapping each such edge id to its head.
        metric: ``"ratio"`` or ``"diff"``.
        prune: True when pendant bundles are peeled before scoring.
        mode: Core order.
        same_first: True when same-level edges are oriented before the cuts.
        tolerance: Width at which the search stops.

    Returns:
        tuple[dict[int, int], int]: Best orientation and the number of trials.

    Raises:
        RuntimeError: The ceiling trial produced no orientation.
    """
    free = [edge for edge in instance.edges]
    lo = max(min(edge.w_u, edge.w_v) for edge in free)
    hi = sum(max(edge.w_u, edge.w_v) for edge in free)
    trials = 1
    found = attempt(instance, hi, metric, prune, mode, same_first)
    if found is None:
        raise RuntimeError("score-family found no orientation at the weight ceiling")
    best_heads = found
    best_score = makespan(instance, found)
    while hi - lo > tolerance:
        threshold = (lo + hi) / 2.0
        if threshold <= lo or threshold >= hi:
            break
        trials += 1
        found = attempt(instance, threshold, metric, prune, mode, same_first)
        if found is None:
            lo = threshold
            continue
        score = makespan(instance, found)
        if score < best_score:
            best_score = score
            best_heads = found
        hi = threshold
    return best_heads, trials


def _absorb(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
) -> None:
    """Orient a free star toward its node whenever the whole star fits in ``T``.

    Nodes are scanned in id order. The scan repeats until no star fits.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented. Absorbed ids are removed.
    """
    changed = True
    while changed:
        changed = False
        for node in instance.nodes:
            star = [edge for edge in instance.incident(node) if edge.id in free]
            if not star:
                continue
            total = sum(edge.weight_toward(node) for edge in star)
            if not nearly_leq(load[node] + total, threshold):
                continue
            for edge in star:
                _orient(edge, node, load, assign, free)
            changed = True


def _peel(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
) -> bool:
    """Peel pendant nodes until none remain. Reject when a peel exceeds ``T``.

    A pendant keeps edges in increasing metric order for as long as they fit.
    The rest of the bundle is sent to the parent.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented. Peeled ids are removed.
        metric: ``"ratio"`` or ``"diff"``.

    Returns:
        bool: False when a peeled bundle pushes an endpoint above ``T``.
    """
    changed = True
    while changed:
        changed = False
        for node in instance.nodes:
            neighbors = list(_free_neighbors(instance, node, free))
            if len(neighbors) != 1:
                continue
            parent = neighbors[0]
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
                    edge.id,
                )
            )
            for edge in bundle:
                if _fits(load, node, edge, threshold):
                    _orient(edge, node, load, assign, free)
                else:
                    _orient(edge, parent, load, assign, free)
            if not nearly_leq(load[parent], threshold) or not nearly_leq(
                load[node], threshold
            ):
                return False
            changed = True
    return True


def _scores(
    instance: Instance,
    free: set[int],
    metric: str,
) -> tuple[dict[int, _MetricKey], dict[int, float]]:
    """Hypothetical in-load and out-load of every node that still has a free edge.

    In-load sums the weight of pointing each free edge at the node. Out-load
    sums the weight of pointing each free edge at the other end. Neither sum
    is written as an orientation.

    Args:
        instance: Graph being oriented.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.

    Returns:
        tuple[dict[int, _MetricKey], dict[int, float]]: Metric key of each
        active node, and that node's out-load.
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
    keys = {node: _edge_key(metric, in_load[node], out_load[node]) for node in active}
    return keys, out_load


def _core(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    mode: str,
    keys: dict[int, _MetricKey],
    out_load: dict[int, float],
    dist_sink: dict[int, int],
    dist_source: dict[int, int],
) -> bool:
    """Orient free edges by the selected Core order. Reject above ``T``.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented. Oriented ids are removed.
        metric: ``"ratio"`` or ``"diff"``.
        mode: Core order.
        keys: Metric key of each node that had a free edge when scoring began.
        out_load: Hypothetical out-load of every node, used by the flow.
        dist_sink: Distance to the nearest sink on the free graph.
        dist_source: Distance to the nearest source on the free graph.

    Returns:
        bool: False when an edge cannot be placed without exceeding ``T``.
    """
    if mode == "sink":
        return _offload_levels(
            instance, threshold, load, assign, free, metric, dist_sink, True
        )
    if mode == "source":
        return _offload_levels(
            instance, threshold, load, assign, free, metric, dist_source, False
        )
    if mode == "both":
        return _offload_both(
            instance,
            threshold,
            load,
            assign,
            free,
            metric,
            dist_sink,
            dist_source,
        )
    if mode == "flow":
        outflow = _score_flow(instance, threshold, load, free, keys, out_load)
        order = sorted(
            keys, key=lambda node: (-outflow[node], _negate(keys[node]), node)
        )
        return _concede(instance, threshold, load, assign, free, metric, keys, order)
    if mode == "inflow":
        inflow, channel, caps = _measure_inflow(
            instance, threshold, load, free, keys, out_load
        )
        return _assign_inflow(
            instance,
            threshold,
            load,
            assign,
            free,
            metric,
            keys,
            inflow,
            channel,
            caps,
        )
    if mode == "topo":
        arcs = _score_arcs(instance, free, keys)
        return _concede(
            instance,
            threshold,
            load,
            assign,
            free,
            metric,
            keys,
            _topo_order(keys, arcs),
        )


def _offload_levels(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    levels: dict[int, int],
    sender_is_higher: bool,
) -> bool:
    """Offload each consecutive BFS cut, starting at the seed layer.

    When ``sender_is_higher`` is true the farther node is the sender, so load
    overflows toward the seeds. Otherwise the node closer to the seeds is the
    sender, so load overflows away from them.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.
        levels: BFS distance of each active node from the seeds.
        sender_is_higher: True when the larger distance is the sender.

    Returns:
        bool: False when an edge on a cut fits on neither endpoint.
    """
    if not levels:
        return True
    last = max(levels.values())
    for low in range(last):
        batch: list[tuple] = []
        for edge in instance.edges:
            if edge.id not in free:
                continue
            left = levels.get(edge.u)
            right = levels.get(edge.v)
            if left is None or right is None:
                continue
            if abs(left - right) != 1 or min(left, right) != low:
                continue
            if sender_is_higher:
                sender = edge.u if left > right else edge.v
            else:
                sender = edge.u if left < right else edge.v
            batch.append((edge, sender, edge.other(sender)))
        if not _offload(threshold, load, assign, free, metric, batch):
            return False
    return True


def _offload_both(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    dist_sink: dict[int, int],
    dist_source: dict[int, int],
) -> bool:
    """Push toward sinks, one sender at a time, overflow toward better receivers.

    Senders closer to a sink go first. A tie goes to the sender farther from
    a source, then to the smaller node id. Edges that fit stay on the sender,
    cheapest metric first. Overflow edges then try receivers in that same
    order, so a nearer source is used only after an earlier receiver.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.
        dist_sink: Distance to the nearest sink.
        dist_source: Distance to the nearest source.

    Returns:
        bool: False when an overflow edge does not fit on its receiver.
    """
    if not dist_sink:
        return True
    last = max(dist_sink.values())
    for low in range(last):
        grouped: dict[int, list[tuple]] = defaultdict(list)
        for edge in instance.edges:
            if edge.id not in free:
                continue
            left = dist_sink.get(edge.u)
            right = dist_sink.get(edge.v)
            if left is None or right is None:
                continue
            if abs(left - right) != 1 or min(left, right) != low:
                continue
            sender = edge.u if left > right else edge.v
            grouped[sender].append((edge, edge.other(sender)))
        senders = sorted(
            grouped,
            key=lambda node: (dist_sink[node], -dist_source[node], node),
        )
        for sender in senders:
            items = grouped[sender]
            items.sort(
                key=lambda item: (
                    _edge_key(
                        metric,
                        item[0].weight_toward(sender),
                        item[0].weight_toward(item[1]),
                    ),
                    item[0].id,
                )
            )
            overflow: list[tuple] = []
            for edge, receiver in items:
                if edge.id not in free:
                    continue
                if _fits(load, sender, edge, threshold):
                    _orient(edge, sender, load, assign, free)
                else:
                    overflow.append((edge, receiver))
            overflow.sort(
                key=lambda item: (
                    dist_sink[item[1]],
                    -dist_source[item[1]],
                    item[1],
                    item[0].id,
                )
            )
            for edge, receiver in overflow:
                if _fits(load, receiver, edge, threshold):
                    _orient(edge, receiver, load, assign, free)
                else:
                    return False
    return True


def _concede(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    keys: dict[int, _MetricKey],
    order: list[int],
) -> bool:
    """Let each node, in ``order``, keep or hand off edges to a lower score.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.
        keys: Metric key of each active node.
        order: Nodes, highest priority first.

    Returns:
        bool: False when an edge fits on neither the node nor its lower neighbor.
    """
    for node in order:
        batch = []
        for edge in instance.incident(node):
            if edge.id not in free:
                continue
            other = edge.other(node)
            if other not in keys or not keys[other] < keys[node]:
                continue
            batch.append((edge, node, other))
        if not _offload(threshold, load, assign, free, metric, batch):
            return False
    return True


def _offload(
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    batch: list[tuple],
) -> bool:
    """Keep each edge on its sender while it fits, otherwise give it to the receiver.

    The batch is sorted by the edge metric ascending, then by edge id.

    Args:
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.
        batch: ``(edge, sender, receiver)`` triples.

    Returns:
        bool: False when an edge fits on neither endpoint.
    """
    ordered = sorted(
        batch,
        key=lambda item: (
            _edge_key(
                metric,
                item[0].weight_toward(item[1]),
                item[0].weight_toward(item[2]),
            ),
            item[0].id,
        ),
    )
    for edge, sender, receiver in ordered:
        if edge.id not in free:
            continue
        if _fits(load, sender, edge, threshold):
            _orient(edge, sender, load, assign, free)
        elif _fits(load, receiver, edge, threshold):
            _orient(edge, receiver, load, assign, free)
        else:
            return False
    return True


def _same_level(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    mode: str,
    keys: dict[int, _MetricKey],
    dist_sink: dict[int, int],
    dist_source: dict[int, int],
) -> bool:
    """Orient free edges whose endpoints share a level, before the cuts.

    ``source`` uses distance from the sources. The other modes use distance
    from the sinks. The batch is sorted by the chosen edge metric. The more
    sink-like endpoint is the receiver, as in the final leftover pass.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.
        mode: Core order, which chooses the level map and the receiver.
        keys: Metric key of each node scored before this pass.
        dist_sink: Distance to the nearest sink.
        dist_source: Distance to the nearest source.

    Returns:
        bool: False when a same-level edge fits on neither endpoint.
    """
    levels = dist_source if mode == "source" else dist_sink
    batch = []
    for edge in instance.edges:
        if edge.id not in free:
            continue
        left = levels.get(edge.u)
        right = levels.get(edge.v)
        if left is None or right is None or left != right:
            continue
        receiver = _preferred_receiver(
            edge.u, edge.v, mode, keys, dist_sink, dist_source
        )
        batch.append((edge, edge.other(receiver), receiver))
    return _offload(threshold, load, assign, free, metric, batch)


def _leftover(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    mode: str,
    keys: dict[int, _MetricKey],
    dist_sink: dict[int, int],
    dist_source: dict[int, int],
) -> bool:
    """Orient edges the layer pass left free. The more sink-like end receives.

    ``sink`` and ``topo`` prefer the lower metric. ``source`` prefers the node
    farther from a source. ``both``, ``flow``, and ``inflow`` prefer the smaller
    distance to a sink, then the larger distance to a source.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented.
        metric: ``"ratio"`` or ``"diff"``.
        mode: Core order, which chooses the preferred receiver.
        keys: Metric key of each node scored before stage 3.
        dist_sink: Distance to the nearest sink.
        dist_source: Distance to the nearest source.

    Returns:
        bool: False when an edge fits on neither endpoint.
    """
    batch = []
    for edge in instance.edges:
        if edge.id not in free:
            continue
        receiver = _preferred_receiver(
            edge.u, edge.v, mode, keys, dist_sink, dist_source
        )
        batch.append((edge, edge.other(receiver), receiver))
    return _offload(threshold, load, assign, free, metric, batch)


def _preferred_receiver(
    left: int,
    right: int,
    mode: str,
    keys: dict[int, _MetricKey],
    dist_sink: dict[int, int],
    dist_source: dict[int, int],
) -> int:
    """Endpoint that should receive a leftover edge when both choices fit.

    Args:
        left: One endpoint.
        right: The other endpoint.
        mode: Core order.
        keys: Metric key of each scored node.
        dist_sink: Distance to the nearest sink.
        dist_source: Distance to the nearest source.

    Returns:
        int: The more sink-like endpoint. Equal ranks select the smaller id
        through the last entry of the rank.
    """
    if mode == "source":
        return _lower_rank(
            left,
            right,
            lambda node: (-dist_source[node], keys[node], node),
        )
    if mode in ("both", "flow", "inflow"):
        return _lower_rank(
            left,
            right,
            lambda node: (dist_sink[node], -dist_source[node], keys[node], node),
        )
    return _lower_rank(left, right, lambda node: (keys[node], node))


def _lower_rank(left: int, right: int, rank) -> int:
    """Endpoint with the smaller rank.

    Args:
        left: One endpoint.
        right: The other endpoint.
        rank: Callable from a node id to a comparable rank. Smaller is preferred.

    Returns:
        int: ``left`` when its rank is smaller or equal, otherwise ``right``.
    """
    if rank(left) <= rank(right):
        return left
    return right


def _assign_inflow(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
    metric: str,
    keys: dict[int, _MetricKey],
    inflow: dict[int, float],
    channel: dict[tuple[int, int], float],
    caps: dict[tuple[int, int], float],
) -> bool:
    """Claim free edges from the largest in-flow, one channel at a time.

    A channel's allowance is its share of the node's in-flow times the frozen
    capacity of the arc into the node. Edges already sent to the node from
    that channel earlier in this pass count against the allowance. Within a
    channel the edge metric decides the order, as in a peel: the node keeps an
    edge while it fits under ``T`` and under the remaining allowance, and
    otherwise the neighbor takes it.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented. Oriented ids are removed.
        metric: ``"ratio"`` or ``"diff"``.
        keys: Metric key of each active node.
        inflow: Flow that arrived at each node across channel arcs.
        channel: Flow on each ``(tail, head)`` arc.
        caps: Capacity of each ``(tail, head)`` arc at flow time.

    Returns:
        bool: False when an edge fits on neither endpoint.
    """
    original = set(free)
    order = sorted(keys, key=lambda node: (-inflow.get(node, 0.0), node))
    for node in order:
        bundles: dict[int, list] = defaultdict(list)
        for edge in instance.incident(node):
            if edge.id not in free:
                continue
            bundles[edge.other(node)].append(edge)
        neighbors = sorted(
            bundles,
            key=lambda other: (-channel.get((other, node), 0.0), other),
        )
        node_in = inflow.get(node, 0.0)
        for other in neighbors:
            arrived = channel.get((other, node), 0.0)
            cap = caps.get((other, node), 0.0)
            share = arrived / node_in if node_in > ABS_TOL else 0.0
            allowance = share * cap
            used = 0.0
            for edge in instance.edges:
                if edge.id not in original or edge.id in free:
                    continue
                if assign.get(edge.id) != node:
                    continue
                if node not in edge.endpoints() or other not in edge.endpoints():
                    continue
                used += edge.weight_toward(node)
            remaining = allowance - used
            bundle = sorted(
                bundles[other],
                key=lambda edge: (
                    _edge_key(
                        metric,
                        edge.weight_toward(node),
                        edge.weight_toward(other),
                    ),
                    edge.id,
                ),
            )
            for edge in bundle:
                if edge.id not in free:
                    continue
                weight = edge.weight_toward(node)
                if _fits(load, node, edge, threshold) and nearly_leq(weight, remaining):
                    _orient(edge, node, load, assign, free)
                    remaining -= weight
                elif _fits(load, other, edge, threshold):
                    _orient(edge, other, load, assign, free)
                else:
                    return False
    return True


def _score_flow(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    free: set[int],
    keys: dict[int, _MetricKey],
    out_load: dict[int, float],
) -> dict[int, float]:
    """Push hypothetical out-load from high scores toward low scores.

    A simple arc goes from the strictly higher score to the strictly lower
    score. Its capacity is the free weight toward the lower node. Each node
    accepts at most ``T`` minus its current load. The super-source feeds each
    in-degree-zero node its out-load. Each out-degree-zero node drains into
    the super-sink up to its room.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node.
        free: Edge ids still unoriented.
        keys: Metric key of each active node.
        out_load: Hypothetical out-load of every node.

    Returns:
        dict[int, float]: Flow pushed across score arcs out of each active node.
    """
    outflow, _inflow, _channel, _caps = _push_score_flow(
        instance, threshold, load, free, keys, out_load, False
    )
    return outflow


def _measure_inflow(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    free: set[int],
    keys: dict[int, _MetricKey],
    out_load: dict[int, float],
) -> tuple[dict[int, float], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Push the score flow with an equal-score bundle open in both directions.

    Each direction counts toward indegree and outdegree. The super-source
    feeds a node only when no arc points at it, and the super-sink drains a
    node only when it has no arc out.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node.
        free: Edge ids still unoriented.
        keys: Metric key of each active node.
        out_load: Hypothetical out-load of every node.

    Returns:
        tuple[dict[int, float], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        In-flow of each active node, flow on each ``(tail, head)`` arc, and
        that arc's capacity. The super-source feed is not in-flow.
    """
    _outflow, inflow, channel, caps = _push_score_flow(
        instance, threshold, load, free, keys, out_load, True
    )
    return inflow, channel, caps


def _push_score_flow(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    free: set[int],
    keys: dict[int, _MetricKey],
    out_load: dict[int, float],
    both_ways: bool,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """Run one score-flow and read the flow off every positive arc.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node.
        free: Edge ids still unoriented.
        keys: Metric key of each active node.
        out_load: Hypothetical out-load of every node.
        both_ways: True when equal-score bundles contribute both directions.

    Returns:
        tuple[dict[int, float], dict[int, float], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        Out-flow of each node, in-flow of each node, flow on each
        ``(tail, head)`` arc, and that arc's capacity.
    """
    outflow = {node: 0.0 for node in keys}
    inflow = {node: 0.0 for node in keys}
    channel: dict[tuple[int, int], float] = {}
    caps: dict[tuple[int, int], float] = {}
    arcs = _bundle_arcs(instance, free, keys, both_ways)
    if not arcs:
        return outflow, inflow, channel, caps
    nodes = list(keys)
    count = 2 + 2 * len(nodes)
    graph: list[list[list]] = [[] for _ in range(count)]
    in_of = {node: 2 + 2 * index for index, node in enumerate(nodes)}
    out_of = {node: 3 + 2 * index for index, node in enumerate(nodes)}
    indeg = {node: 0 for node in nodes}
    outdeg = {node: 0 for node in nodes}
    recorded: list[tuple[int, int, list, float]] = []
    for tail, head, cap in arcs:
        indeg[head] += 1
        outdeg[tail] += 1
        forward = _add_arc(graph, out_of[tail], in_of[head], cap)
        if forward is not None:
            recorded.append((tail, head, forward, cap))
            caps[(tail, head)] = cap
    for node in nodes:
        room = threshold - load[node]
        if room < 0:
            room = 0.0
        _add_arc(graph, in_of[node], out_of[node], room)
        if indeg[node] == 0:
            _add_arc(graph, 0, out_of[node], out_load.get(node, 0.0))
        if outdeg[node] == 0:
            _add_arc(graph, out_of[node], 1, room)
    _edmonds_karp(graph, source=0, sink=1)
    for tail, head, forward, cap in recorded:
        pushed = max(0.0, cap - forward[1])
        outflow[tail] += pushed
        inflow[head] += pushed
        channel[(tail, head)] = pushed
    return outflow, inflow, channel, caps


def _directed(
    instance: Instance,
    free: set[int],
    keys: dict[int, _MetricKey],
) -> list[tuple[int, int, float]]:
    """One arc per adjacent pair, from the higher score to the lower score.

    Args:
        instance: Graph being oriented.
        free: Edge ids still unoriented.
        keys: Metric key of each active node.

    Returns:
        list[tuple[int, int, float]]: ``(higher, lower, weight toward lower)``.
        Equal scores produce no arc.
    """
    return _bundle_arcs(instance, free, keys, False)


def _bundle_arcs(
    instance: Instance,
    free: set[int],
    keys: dict[int, _MetricKey],
    both_ways: bool,
) -> list[tuple[int, int, float]]:
    """Channel arcs and the free weight toward each head.

    A strictly higher score points at the lower score. When ``both_ways`` is
    true, an equal-score bundle points both ways and each direction counts as
    its own arc.

    Args:
        instance: Graph being oriented.
        free: Edge ids still unoriented.
        keys: Metric key of each active node.
        both_ways: True when equal-score bundles contribute both directions.

    Returns:
        list[tuple[int, int, float]]: ``(tail, head, weight toward head)``.
    """
    grouped: dict[tuple[int, int], list] = defaultdict(list)
    for edge in instance.edges:
        if edge.id not in free:
            continue
        grouped[tuple(sorted(edge.endpoints()))].append(edge)
    arcs = []
    for (left, right), edges in grouped.items():
        toward_left = sum(edge.weight_toward(left) for edge in edges)
        toward_right = sum(edge.weight_toward(right) for edge in edges)
        if keys[left] == keys[right]:
            if not both_ways:
                continue
            arcs.append((left, right, toward_right))
            arcs.append((right, left, toward_left))
            continue
        if keys[left] > keys[right]:
            arcs.append((left, right, toward_right))
        else:
            arcs.append((right, left, toward_left))
    return arcs


def _score_arcs(
    instance: Instance,
    free: set[int],
    keys: dict[int, _MetricKey],
) -> list[tuple[int, int]]:
    """Higher-score to lower-score arcs, ignoring capacity.

    Args:
        instance: Graph being oriented.
        free: Edge ids still unoriented.
        keys: Metric key of each active node.

    Returns:
        list[tuple[int, int]]: Directed pairs. Equal scores produce no arc.
    """
    return [(high, low) for high, low, _cap in _directed(instance, free, keys)]


def _topo_order(
    keys: dict[int, _MetricKey],
    arcs: list[tuple[int, int]],
) -> list[int]:
    """Kahn order of the score DAG. Ready nodes break ties by metric, then id.

    A larger metric is emitted first. The score digraph is a DAG when arcs
    follow a strict metric decrease. Any node left by a cycle is appended in
    that same order.

    Args:
        keys: Metric key of each active node.
        arcs: Directed pairs from a higher score to a lower score.

    Returns:
        list[int]: Nodes, highest conceding priority first.
    """
    indeg = {node: 0 for node in keys}
    succ: dict[int, list[int]] = {node: [] for node in keys}
    for src, dst in arcs:
        if src not in indeg or dst not in indeg or src == dst:
            continue
        succ[src].append(dst)
        indeg[dst] += 1
    ready = [node for node, degree in indeg.items() if degree == 0]
    order: list[int] = []
    seen: set[int] = set()
    while ready:
        node = min(ready, key=lambda item: (_negate(keys[item]), item))
        ready.remove(node)
        order.append(node)
        seen.add(node)
        for dst in succ[node]:
            indeg[dst] -= 1
            if indeg[dst] == 0 and dst not in seen:
                ready.append(dst)
    if len(order) < len(keys):
        rest = [node for node in keys if node not in seen]
        rest.sort(key=lambda item: (_negate(keys[item]), item))
        order.extend(rest)
    return order


def _add_arc(graph: list[list[list]], src: int, dst: int, cap: float) -> list | None:
    """Add a residual arc when ``cap`` is positive.

    Args:
        graph: Residual adjacency. Each entry is ``[head, capacity, reverse index]``.
        src: Tail of the forward arc.
        dst: Head of the forward arc.
        cap: Forward capacity.

    Returns:
        list | None: The forward residual entry, or ``None`` when ``cap`` is negligible.
    """
    if cap <= ABS_TOL:
        return None
    forward = [dst, float(cap), len(graph[dst])]
    backward = [src, 0.0, len(graph[src])]
    graph[src].append(forward)
    graph[dst].append(backward)
    return forward


def _edmonds_karp(graph: list[list[list]], source: int, sink: int) -> float:
    """Maximum flow by BFS augmenting paths.

    Args:
        graph: Residual adjacency. Capacities are updated in place.
        source: Super-source index.
        sink: Super-sink index.

    Returns:
        float: Total flow pushed from ``source`` to ``sink``.
    """
    total = 0.0
    while True:
        parent = [-1] * len(graph)
        parent_edge = [-1] * len(graph)
        parent[source] = source
        queue: deque[int] = deque([source])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            for index, edge in enumerate(graph[node]):
                if edge[1] > ABS_TOL and parent[edge[0]] == -1:
                    parent[edge[0]] = node
                    parent_edge[edge[0]] = index
                    queue.append(edge[0])
        if parent[sink] == -1:
            break
        bottleneck = inf
        node = sink
        while node != source:
            prev = parent[node]
            bottleneck = min(bottleneck, graph[prev][parent_edge[node]][1])
            node = prev
        if bottleneck <= ABS_TOL:
            break
        node = sink
        while node != source:
            prev = parent[node]
            forward = graph[prev][parent_edge[node]]
            forward[1] -= bottleneck
            graph[node][forward[2]][1] += bottleneck
            node = prev
        total += bottleneck
    return total


def _distances(
    adj: dict[int, set[int]],
    keys: dict[int, _MetricKey],
) -> tuple[dict[int, int], dict[int, int]]:
    """BFS distance to the nearest sink and the nearest source, per component.

    Args:
        adj: Free-graph adjacency of the active nodes.
        keys: Metric key of each active node.

    Returns:
        tuple[dict[int, int], dict[int, int]]: Distance to a lowest-metric seed
        and distance to a highest-metric seed.
    """
    components = _components(adj)
    sinks: list[int] = []
    sources: list[int] = []
    for component in components:
        sinks.extend(_extremal(component, keys, lowest=True))
        sources.extend(_extremal(component, keys, lowest=False))
    return _bfs(adj, sinks), _bfs(adj, sources)


def _components(adj: dict[int, set[int]]) -> list[list[int]]:
    """Connected components of a free adjacency, each in discovery order.

    Args:
        adj: Undirected adjacency.

    Returns:
        list[list[int]]: One node list per component. Starts are sorted.
    """
    seen: set[int] = set()
    found: list[list[int]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adj[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        found.append(component)
    return found


def _extremal(
    component: list[int],
    keys: dict[int, _MetricKey],
    lowest: bool,
) -> list[int]:
    """Nodes of a component that share the lowest or highest metric.

    Args:
        component: Nodes in one free component.
        keys: Metric key of each active node.
        lowest: True to select sinks, false to select sources.

    Returns:
        list[int]: Every node whose key equals that extreme.
    """
    best = (
        min(keys[node] for node in component)
        if lowest
        else max(keys[node] for node in component)
    )
    return [node for node in component if keys[node] == best]


def _bfs(adj: dict[int, set[int]], seeds: list[int]) -> dict[int, int]:
    """Distance from the nearest seed. Neighbor order is sorted.

    Args:
        adj: Undirected adjacency.
        seeds: Level-zero nodes.

    Returns:
        dict[int, int]: Distance of every node reached from ``seeds``.
    """
    dist: dict[int, int] = {}
    queue: deque[int] = deque()
    for seed in sorted(seeds):
        if seed not in dist:
            dist[seed] = 0
            queue.append(seed)
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adj[node]):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def _adjacency(instance: Instance, free: set[int]) -> dict[int, set[int]]:
    """Undirected adjacency of the edges that are still free.

    Args:
        instance: Graph being oriented.
        free: Edge ids still unoriented.

    Returns:
        dict[int, set[int]]: Neighbors of each endpoint of a free edge.
    """
    adj: dict[int, set[int]] = defaultdict(set)
    for edge in instance.edges:
        if edge.id not in free:
            continue
        adj[edge.u].add(edge.v)
        adj[edge.v].add(edge.u)
    return adj


def _free_neighbors(instance: Instance, node: int, free: set[int]) -> set[int]:
    """Endpoints reached from ``node`` by an edge that is still free.

    Args:
        instance: Graph being oriented.
        node: Node whose free neighborhood is requested.
        free: Edge ids that have not been oriented.

    Returns:
        set[int]: Distinct free neighbors of ``node``.
    """
    found = set()
    for edge in instance.incident(node):
        if edge.id in free:
            found.add(edge.other(node))
    return found


def _fits(load: dict[int, float], node: int, edge, threshold: float) -> bool:
    """True when ``edge`` can be oriented toward ``node`` without passing ``T``.

    Args:
        load: Current load of every node.
        node: Endpoint that would receive ``edge``.
        edge: Edge being tested.
        threshold: Candidate makespan ``T``.

    Returns:
        bool: True when the resulting load is at most ``T``.
    """
    return nearly_leq(load[node] + edge.weight_toward(node), threshold)


def _orient(
    edge, head: int, load: dict[int, float], assign: dict[int, int], free: set[int]
) -> None:
    """Point ``edge`` at ``head`` and add its weight to that load.

    Args:
        edge: Edge being oriented.
        head: Endpoint that receives it.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented. ``edge`` is removed.
    """
    assign[edge.id] = head
    load[head] += edge.weight_toward(head)
    free.discard(edge.id)


def _complete(
    instance: Instance,
    locked: dict[int, int],
    assign: dict[int, int],
) -> dict[int, int]:
    """Copy a head for every edge, preferring the trial assignment.

    Args:
        instance: Graph whose edges must all appear.
        locked: Fallback heads for edges the trial did not write.
        assign: Heads written by the trial.

    Returns:
        dict[int, int]: Complete orientation.

    Raises:
        RuntimeError: An edge has no head.
    """
    heads: dict[int, int] = {}
    for edge in instance.edges:
        head = assign.get(edge.id, locked.get(edge.id))
        if head is None:
            raise RuntimeError(f"edge {edge.id} was left unoriented")
        heads[edge.id] = head
    return heads
