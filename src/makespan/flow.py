"""Choice-network lower bound for a candidate makespan.

For a threshold ``T`` the network asks whether every free edge can be given
to one admissible endpoint without exceeding a count capacity at any node.

* An endpoint is admissible while its incident weight still fits in the
  residual load ``T - base_load``.
* Edges with only one admissible endpoint are forced, and that can make
  another endpoint inadmissible. Forcing repeats to a fixpoint.
* ``cap(v)`` is the largest number of free edges that could still enter
  ``v`` if the lightest candidates were chosen. It is an upper bound on the
  number of free edges any feasible orientation can send to ``v``.
* Source sends one unit to each free edge, the edge offers that unit to each
  admissible endpoint, and the endpoint forwards at most ``cap(v)`` units
  to the terminal.

If the maximum flow is smaller than the number of free edges, no orientation
has makespan at most ``T``. If the flow saturates, the count relaxation is
feasible. That is necessary for a real orientation, and it is also sufficient
when every weight is 1, because then the count is the load. With unequal
weights the recovered orientation is evaluated with the real weights and kept
only when it improves the incumbent. The smallest saturating ``T`` is a
lower bound.

The search upper bound is the best greedy makespan. At that value the
network is feasible: the greedy orientation itself is a saturating flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import ABS_TOL, Instance, nearly_leq

_SCAN_LIMIT = 100_000


@dataclass
class _SearchHit:
    """Smallest proven threshold found by binary search, plus one orientation.

    Args:
        lower_bound: Proven lower bound on the saturating threshold, if any.
        heads: One saturating orientation, or ``None`` when none was recovered.
        monotone: Whether every rejected probe sat below every accepted probe.

    Returns:
        _SearchHit: Proven bound, recovered heads if any, and monotonicity flag.
    """

    lower_bound: float | None
    heads: dict[int, int] | None
    monotone: bool


class _Dinic:
    """Maximum flow with integer capacities.

    Args:
        node_count: Number of vertices in the residual network.
    """

    def __init__(self, node_count: int):
        """Create an empty graph on ``node_count`` nodes.

        Args:
            node_count: Number of vertices in the residual network.
        """
        self.graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(self, src: int, dst: int, cap: int) -> int:
        """Add an arc and return the index of its forward residual entry.

        Args:
            src: Tail of the forward residual arc.
            dst: Head of the forward residual arc.
            cap: Integer capacity of the forward residual arc.

        Returns:
            int: Index of the forward residual entry at ``src``.
        """
        forward_index = len(self.graph[src])
        self.graph[src].append([dst, cap, len(self.graph[dst])])
        self.graph[dst].append([src, 0, forward_index])
        return forward_index

    def max_flow(self, source: int, sink: int) -> int:
        """Push as much integer flow as possible from ``source`` to ``sink``.

        Args:
            source: Source vertex of the flow.
            sink: Sink vertex of the flow.

        Returns:
            int: Value of a maximum integer flow.
        """
        total = 0
        while True:
            level = self._levels(source)
            if level[sink] < 0:
                return total
            progress = [0] * len(self.graph)
            while True:
                pushed = self._push(source, sink, 10**18, level, progress)
                if pushed == 0:
                    break
                total += pushed

    def _levels(self, source: int) -> list[int]:
        """Breadth-first distances from ``source`` along residual arcs with spare capacity.

        Args:
            source: Vertex from which residual distances are measured.

        Returns:
            list[int]: Distance of each vertex, or ``-1`` if unreachable.
        """
        level = [-1] * len(self.graph)
        level[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for nxt, cap, _rev in self.graph[node]:
                if cap > 0 and level[nxt] < 0:
                    level[nxt] = level[node] + 1
                    queue.append(nxt)
        return level

    def _push(
        self,
        node: int,
        sink: int,
        limit: int,
        level: list[int],
        progress: list[int],
    ) -> int:
        """Send up to ``limit`` units along the level graph, or return 0.

        Args:
            node: Current vertex on the blocking-flow DFS.
            sink: Target of the flow.
            limit: Remaining capacity that may be pushed from ``node``.
            level: BFS distances used to stay on the level graph.
            progress: Next unused residual-arc index at each vertex.

        Returns:
            int: Units pushed from ``node`` toward ``sink``, or ``0``.
        """
        if node == sink:
            return limit
        edges = self.graph[node]
        while progress[node] < len(edges):
            index = progress[node]
            nxt, cap, rev = edges[index]
            if cap > 0 and level[nxt] == level[node] + 1:
                pushed = self._push(nxt, sink, min(limit, cap), level, progress)
                if pushed:
                    edges[index][1] -= pushed
                    self.graph[nxt][rev][1] += pushed
                    return pushed
            progress[node] += 1
        return 0


def saturating_orientation(
    instance: Instance,
    threshold: float,
    locked_heads: dict[int, int] | None = None,
) -> tuple[bool, dict[int, int] | None]:
    """Return whether ``threshold`` saturates, and one full orientation if it does.

    ``locked_heads`` are edges already oriented by an earlier step. Their
    weight is base load. The returned orientation includes those edges, the
    edges forced by ``threshold``, and the free edges assigned by the flow.

    Args:
        instance: Graph whose free edges are tested at this threshold.
        threshold: Candidate makespan. Nodes whose load would exceed it are inadmissible.
        locked_heads: Edges already oriented, mapping each such edge id to its head.

    Returns:
        tuple[bool, dict[int, int] | None]: ``(True, heads)`` when the network
        saturates, otherwise ``(False, None)``.
    """
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    locked_heads = dict(locked_heads or {})
    base = {node: 0.0 for node in instance.nodes}
    heads: dict[int, int] = {}
    for edge_id, head in locked_heads.items():
        edge = instance.edge(edge_id)
        if head != edge.u and head != edge.v:
            raise ValueError(f"edge {edge_id} cannot be oriented toward {head}")
        heads[edge_id] = head
        base[head] += edge.weight_toward(head)

    undecided = [edge for edge in instance.edges if edge.id not in heads]
    free = list(undecided)
    while True:
        if any(load > threshold + ABS_TOL for load in base.values()):
            return False, None
        residual = {node: threshold - base[node] for node in instance.nodes}
        force_now: list[tuple] = []
        free = []
        for edge in undecided:
            options = _admissible(edge, residual)
            if not options:
                return False, None
            if len(options) == 1:
                force_now.append((edge, options[0]))
            else:
                free.append(edge)
        if not force_now:
            break
        for edge, head in force_now:
            heads[edge.id] = head
            base[head] += edge.weight_toward(head)
        undecided = free

    residual = {node: threshold - base[node] for node in instance.nodes}
    if not free:
        return True, heads

    source = 0
    sink = 1
    edge_node = {edge.id: 2 + index for index, edge in enumerate(free)}
    vertex_node = {
        node: 2 + len(free) + index for index, node in enumerate(instance.nodes)
    }
    flow = _Dinic(2 + len(free) + len(instance.nodes))
    arcs: dict[int, list[tuple[int, int, int]]] = {}
    caps = _slot_capacities(free, residual)
    for edge in free:
        flow.add_edge(source, edge_node[edge.id], 1)
        for head in _admissible(edge, residual):
            index = flow.add_edge(edge_node[edge.id], vertex_node[head], 1)
            arcs.setdefault(edge.id, []).append((edge_node[edge.id], index, head))
    for node, cap in caps.items():
        if cap:
            flow.add_edge(vertex_node[node], sink, cap)

    if flow.max_flow(source, sink) != len(free):
        return False, None
    for edge in free:
        chosen = [
            head
            for node, index, head in arcs.get(edge.id, [])
            if 1 - flow.graph[node][index][1] >= 1
        ]
        if len(chosen) != 1:
            raise RuntimeError(f"edge {edge.id} did not receive a single unit of flow")
        heads[edge.id] = chosen[0]
    if len(heads) != len(instance.edges):
        raise RuntimeError("flow orientation is missing an edge")
    return True, heads


def assign_free_by_flow(
    instance: Instance,
    locked_heads: dict[int, int],
    residual_cap: dict[int, float],
) -> dict[int, int] | None:
    """Orient every free edge by one saturating flow under residual capacities.

    ``locked_heads`` are already chosen and are copied into the result. A free
    edge may enter a node only when its weight fits in ``residual_cap``. The
    node's count capacity is the most free edges that could fit if the
    lightest candidates were chosen. The function returns ``None`` when the
    flow cannot place every free edge.

    Args:
        instance: Graph whose remaining edges are assigned by the flow.
        locked_heads: Edges already oriented, mapping each such edge id to its head.
        residual_cap: Remaining load each node can still accept from free edges.

    Returns:
        dict[int, int] | None: Complete orientation, or ``None`` if the flow cannot place every free edge.
    """
    if not all(isfinite(value) for value in residual_cap.values()):
        raise ValueError("residual capacities must be finite")
    free = [edge for edge in instance.edges if edge.id not in locked_heads]
    heads = dict(locked_heads)
    if not free:
        return heads
    residual = {node: residual_cap.get(node, 0.0) for node in instance.nodes}
    source = 0
    sink = 1
    edge_node = {edge.id: 2 + index for index, edge in enumerate(free)}
    vertex_node = {
        node: 2 + len(free) + index for index, node in enumerate(instance.nodes)
    }
    flow = _Dinic(2 + len(free) + len(instance.nodes))
    arcs: dict[int, list[tuple[int, int, int]]] = {}
    caps = _slot_capacities(free, residual)
    for edge in free:
        flow.add_edge(source, edge_node[edge.id], 1)
        for head in _admissible(edge, residual):
            index = flow.add_edge(edge_node[edge.id], vertex_node[head], 1)
            arcs.setdefault(edge.id, []).append((edge_node[edge.id], index, head))
    for node, cap in caps.items():
        if cap:
            flow.add_edge(vertex_node[node], sink, cap)
    if flow.max_flow(source, sink) != len(free):
        return None
    for edge in free:
        chosen = [
            head
            for node, index, head in arcs.get(edge.id, [])
            if 1 - flow.graph[node][index][1] >= 1
        ]
        if len(chosen) != 1:
            raise RuntimeError(f"edge {edge.id} did not receive a single unit of flow")
        heads[edge.id] = chosen[0]
    return heads


class FlowSearchStep(Step):
    """Binary-search the smallest saturating threshold and keep a better orientation.

    Lightest-greedy and load-greedy run first when the context has no
    incumbent, so the search always has a feasible upper bound.

    Args:
        tolerance: Width at which a fractional threshold search stops.
    """

    def __init__(self, tolerance: float = 1e-7):
        """Stop a fractional search once the bracket is narrower than ``tolerance``.

        Args:
            tolerance: Width at which a fractional threshold search stops.
        """
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance = tolerance

    def run(self, ctx: SolveContext) -> None:
        """Search for the smallest saturating threshold and keep a better orientation.

        Lightest-greedy and load-greedy run first when the context has no
        incumbent, so the search always has a feasible upper bound. The
        threshold itself is stored as a lower bound. The orientation read off
        the flow is kept only when its true makespan improves the incumbent.

        Args:
            ctx: Solve state that supplies the greedy upper bound and stores the result.
        """

        if ctx.best_so_far_assignment is None:
            raise RuntimeError("flow search has no feasible orientation to bound it")
        locked = {
            edge_id: ctx.orientation[edge_id]
            for edge_id in ctx.locked
            if ctx.orientation[edge_id] is not None
        }
        hit = _search(
            ctx.instance,
            low=ctx.analytic_lower_bound(),
            high=ctx.best_so_far_makespan,
            locked=locked,
            tolerance=self.tolerance,
        )
        ctx.metadata["flow_monotone"] = hit.monotone
        if hit.lower_bound is not None:
            ctx.metadata["flow_lower_bound"] = hit.lower_bound
            ctx.raise_lower_bound(hit.lower_bound)
        if hit.heads is not None:
            for edge_id, head in hit.heads.items():
                if edge_id not in ctx.locked:
                    ctx.orientation[edge_id] = head
            ctx.consider(hit.heads)


def _search(
    instance: Instance,
    low: float,
    high: float,
    locked: dict[int, int],
    tolerance: float,
) -> _SearchHit:
    """Search from the analytic lower bound ``low`` up to the greedy makespan ``high``.

    Integral weights use an integer search, because every load is then an
    integer. Other weights use a real search that stops at ``tolerance``.

    Args:
        instance: Graph whose saturating threshold is searched.
        low: Analytic lower bound used as the search floor.
        high: Feasible greedy makespan used as the search ceiling.
        locked: Edges already oriented, mapping each such edge id to its head.
        tolerance: Width at which a fractional search stops.

    Returns:
        _SearchHit: Proven bound, recovered heads if any, and monotonicity flag.
    """
    # Integral weights have an integral optimum, so the search stays on integers
    # even when the analytic bound (a sum divided by n) is fractional.
    if instance.weights_are_integral() and _is_snap_int(high):
        return _search_integral(
            instance,
            low=int(ceil(low - 1e-9)),
            high=int(round(high)),
            locked=locked,
        )
    return _search_real(instance, low, high, locked, tolerance)


def _search_integral(
    instance: Instance,
    low: int,
    high: int,
    locked: dict[int, int],
) -> _SearchHit:
    """Integer binary search. A non-monotone probe falls back to a linear scan.

    Args:
        instance: Graph whose saturating threshold is searched.
        low: Integer search floor.
        high: Integer search ceiling, typically the greedy makespan.
        locked: Edges already oriented, mapping each such edge id to its head.

    Returns:
        _SearchHit: Proven integer bound, recovered heads if any, and monotonicity flag.
    """
    if low > high:
        low = high
    cache: dict[int, tuple[bool, dict[int, int] | None]] = {}

    def check(threshold: int) -> tuple[bool, dict[int, int] | None]:
        """Return the cached saturation test for this integer threshold.

        Args:
            threshold: Integer candidate makespan to test.

        Returns:
            tuple[bool, dict[int, int] | None]: Cached saturation result for ``threshold``.
        """
        if threshold not in cache:
            cache[threshold] = saturating_orientation(
                instance, float(threshold), locked
            )
        return cache[threshold]

    high_ok, high_heads = check(high)
    if not high_ok:
        raise RuntimeError("flow network did not saturate at the greedy makespan")
    if low == high or check(low)[0]:
        chosen = low if check(low)[0] else high
        return _hit_from(cache, chosen, check(chosen)[1], integral=True)

    lo = low
    hi = high
    while lo < hi:
        mid = (lo + hi) // 2
        feasible, _heads = check(mid)
        if _violates(cache):
            return _scan_integral(cache, check, low, high)
        if feasible:
            hi = mid
        else:
            lo = mid + 1
    if _violates(cache) or (hi > low and check(hi - 1)[0]):
        return _scan_integral(cache, check, low, high)
    return _hit_from(cache, hi, check(hi)[1], integral=True)


def _search_real(
    instance: Instance,
    low: float,
    high: float,
    locked: dict[int, int],
    tolerance: float,
) -> _SearchHit:
    """Real binary search down to ``tolerance``. A non-monotone probe keeps a safe bound.

    Args:
        instance: Graph whose saturating threshold is searched.
        low: Real search floor.
        high: Real search ceiling, typically the greedy makespan.
        locked: Edges already oriented, mapping each such edge id to its head.
        tolerance: Width at which the search stops.

    Returns:
        _SearchHit: Proven real bound, recovered heads if any, and monotonicity flag.
    """
    if low > high:
        low = high
    cache: dict[float, tuple[bool, dict[int, int] | None]] = {}

    def check(threshold: float) -> tuple[bool, dict[int, int] | None]:
        """Return the cached saturation test for this real threshold.

        Args:
            threshold: Real candidate makespan to test.

        Returns:
            tuple[bool, dict[int, int] | None]: Cached saturation result for ``threshold``.
        """
        if threshold not in cache:
            cache[threshold] = saturating_orientation(
                instance, float(threshold), locked
            )
        return cache[threshold]

    high_ok, high_heads = check(high)
    if not high_ok:
        raise RuntimeError("flow network did not saturate at the greedy makespan")
    if high - low <= tolerance or check(low)[0]:
        chosen = low if check(low)[0] else high
        return _hit_from(cache, chosen, check(chosen)[1], integral=False)
    lo = low
    hi = high
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        feasible, _heads = check(mid)
        if _violates(cache):
            return _SearchHit(_proven_bound(cache, integral=False), high_heads, False)
        if feasible:
            hi = mid
        else:
            lo = mid
    heads = check(hi)[1]
    return _hit_from(cache, hi, heads, integral=False)


def _scan_integral(cache, check, low: int, high: int) -> _SearchHit:
    """Walk every integer from ``low`` to ``high`` and keep the first success.

    Ranges wider than ``_SCAN_LIMIT`` are not scanned. The bound then comes
    only from thresholds already probed, which stays a valid lower bound.

    Args:
        cache: Map from probed integer thresholds to saturation results.
        check: Callable that tests one integer threshold, filling ``cache``.
        low: First integer threshold to try.
        high: Last integer threshold to try.

    Returns:
        _SearchHit: First feasible integer threshold, or a safe bound if the scan is skipped.
    """
    if high - low > _SCAN_LIMIT:
        return _SearchHit(_proven_bound(cache, integral=True), check(high)[1], False)
    threshold = low
    while threshold <= high:
        feasible, heads = check(threshold)
        if feasible:
            return _SearchHit(
                _proven_bound(cache, integral=True),
                heads,
                False,
            )
        threshold += 1
    return _SearchHit(_proven_bound(cache, integral=True), check(high)[1], False)


def _hit_from(cache, chosen, heads, integral: bool) -> _SearchHit:
    """Package ``chosen`` when the probes are monotone, otherwise a safe bound only.

    Args:
        cache: Map from probed thresholds to saturation results.
        chosen: Threshold selected by the search.
        heads: Orientation recovered at ``chosen``, if any.
        integral: Whether the search used integer thresholds.

    Returns:
        _SearchHit: Packaged bound and heads, with ``monotone`` set from ``cache``.
    """
    monotone = not _violates(cache)
    if not monotone:
        return _SearchHit(_proven_bound(cache, integral), heads, False)
    bound = _proven_bound(cache, integral)
    if bound is None:
        bound = float(chosen) if integral else None
    return _SearchHit(bound, heads, True)


def _proven_bound(cache: dict, integral: bool) -> float | None:
    """Lower bound implied by rejected thresholds, tightened when monotone.

    A rejected threshold is a proof that the makespan is larger. When every
    rejected probe sits below every accepted probe, the smallest accepted
    threshold is feasible for the relaxation and is at most the optimum.

    Args:
        cache: Map from probed thresholds to saturation results.
        integral: When true, a rejected integer threshold proves the next integer.

    Returns:
        float | None: Lower bound implied by the probes, or ``None`` if none is proven.
    """
    falses = [threshold for threshold, (ok, _heads) in cache.items() if not ok]
    trues = [threshold for threshold, (ok, _heads) in cache.items() if ok]
    pieces: list[float] = []
    if falses:
        pieces.append(float(max(falses) + (1 if integral else 0)))
    if trues and not _violates(cache):
        pieces.append(float(min(trues)))
    if not pieces:
        return None
    return max(pieces)


def _violates(cache: dict) -> bool:
    """True when some rejected threshold is larger than some accepted one.

    Args:
        cache: Map from probed thresholds to saturation results.

    Returns:
        bool: True when some rejected threshold is larger than some accepted one.
    """
    falses = [threshold for threshold, (ok, _heads) in cache.items() if not ok]
    trues = [threshold for threshold, (ok, _heads) in cache.items() if ok]
    if not falses or not trues:
        return False
    return max(falses) > min(trues)


def _admissible(edge, residual: dict[int, float]) -> list[int]:
    """Endpoints whose incident weight still fits in the residual load.

    Args:
        edge: Edge whose two endpoints are tested.
        residual: Remaining load each node can still accept.

    Returns:
        list[int]: Endpoints whose incident weight still fits.
    """
    options = []
    if nearly_leq(edge.w_u, residual[edge.u]):
        options.append(edge.u)
    if nearly_leq(edge.w_v, residual[edge.v]):
        options.append(edge.v)
    return options


def _slot_capacities(free, residual: dict[int, float]) -> dict[int, int]:
    """Most free edges each node could accept if the lightest candidates were chosen.

    Args:
        free: Edges still unoriented.
        residual: Remaining load each node can still accept.

    Returns:
        dict[int, int]: Count capacity of each node under the lightest-fit rule.
    """
    grouped: dict[int, list[float]] = {node: [] for node in residual}
    for edge in free:
        for head in _admissible(edge, residual):
            grouped[head].append(edge.weight_toward(head))
    caps = {}
    for node, weights in grouped.items():
        weights.sort()
        running = 0.0
        cap = 0
        for weight in weights:
            if nearly_leq(running + weight, residual[node]):
                running += weight
                cap += 1
            else:
                break
        caps[node] = cap
    return caps


def _is_snap_int(value: float) -> bool:
    """True when ``value`` is finite and within ``1e-6`` of an integer.

    Args:
        value: Number to test.

    Returns:
        bool: True when ``value`` is finite and within ``1e-6`` of an integer.
    """
    return isfinite(value) and abs(value - round(value)) <= 1e-6
