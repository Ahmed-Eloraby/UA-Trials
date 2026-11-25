"""Ratio-greedy threshold search.

One trial at a threshold ``T`` peels pendant bundles, orients the core, then
propagates overload. A pendant node keeps edges in increasing order of its
weight over the parent's weight, for as long as they fit in ``T``. The rest
of the bundle is sent to the parent. If either endpoint then exceeds ``T``,
the trial rejects. The core that remains is oriented toward the lighter
endpoint, or toward a random endpoint. Repair then gives each component as
many passes as it has nodes. In a pass, overloaded nodes are visited
heaviest first and hand free edges to a neighbor that this pass has not
visited yet. The neighbor may exceed ``T``. That is how overload propagates.

The outer search keeps the orientation with the smallest measured makespan
among trials that finished inside their threshold. A rejected trial is not a
proof that ``T`` is infeasible, so it does not raise the lower bound.
"""

from __future__ import annotations

import random
from fractions import Fraction

from makespan.context import SolveContext
from makespan.factory import Step
from makespan.model import Instance, is_integral_number, makespan


def _ratio(numerator: float, denominator: float) -> tuple[int, Fraction]:
    """Sort key for ``numerator / denominator``, in increasing order.

    A positive numerator over a zero denominator sorts after every finite ratio.
    Both zero sorts as zero.

    Args:
        numerator: Weight at the node being filled or unloaded.
        denominator: Weight at the other endpoint.

    Returns:
        tuple[int, Fraction]: ``(0, ratio)`` when the ratio is finite, or
        ``(1, 0)`` when it is treated as infinite.
    """
    if denominator == 0:
        if numerator > 0:
            return (1, Fraction(0))
        return (0, Fraction(0))
    return (0, Fraction(numerator) / Fraction(denominator))


def _descending_ratio(numerator: float, denominator: float) -> tuple[int, Fraction]:
    """Sort key for ``numerator / denominator``, in decreasing order.

    Args:
        numerator: Weight at the node being unloaded.
        denominator: Weight at the endpoint that would receive the edge.

    Returns:
        tuple[int, Fraction]: Key that sorts a larger ratio first.
    """
    band, ratio = _ratio(numerator, denominator)
    return (-band, -ratio)


class RatioGreedyStep(Step):
    """Bisect a threshold and keep the best orientation the ratio-greedy trial finds.

    ``start`` chooses the core orientation that repair starts from.
    ``"light"`` sends each core edge to its lighter endpoint. ``"random"``
    uses ``seed`` so the same instance gives the same orientation on every run.

    Args:
        start: ``"light"`` or ``"random"``.
        seed: Seed for a random core. Ignored when ``start`` is ``"light"``.
        tolerance: Width at which a fractional threshold search stops.
    """

    def __init__(self, start: str = "light", seed: int = 0, tolerance: float = 1e-7):
        """Store the core rule and the fractional stopping width.

        Args:
            start: ``"light"`` or ``"random"``.
            seed: Seed for a random core. Ignored when ``start`` is ``"light"``.
            tolerance: Width at which a fractional threshold search stops.
        """
        if start not in ("light", "random"):
            raise ValueError("start must be 'light' or 'random'")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.start = start
        self.seed = seed
        self.tolerance = tolerance

    def run(self, ctx: SolveContext) -> None:
        """Search thresholds and record the best orientation that respected one.

        Locked edges stay as they are. Their weight is already in the load
        when a trial starts, and repair does not move them.

        Args:
            ctx: Solve state that receives the incumbent orientation.
        """
        rng = random.Random(self.seed) if self.start == "random" else None
        heads, trials = search_heads(
            ctx.instance,
            locked=_locked_heads(ctx),
            start=self.start,
            rng=rng,
            tolerance=self.tolerance,
        )
        for edge_id, head in heads.items():
            if edge_id not in ctx.locked:
                ctx.orientation[edge_id] = head
        ctx.consider(heads)
        ctx.metadata["ratio_trials"] = trials
        ctx.metadata["ratio_start"] = self.start


def search_heads(
    instance: Instance,
    locked: dict[int, int] | None = None,
    start: str = "light",
    rng: random.Random | None = None,
    tolerance: float = 1e-7,
) -> tuple[dict[int, int], int]:
    """Return the best orientation found by bisection, and the number of trials.

    Integral weights use the integer search: the floor is the heaviest lighter
    endpoint, and at least 1, and the ceiling is the sum of the heavier
    endpoint weights. Other weights bisect until the bracket is narrower than
    ``tolerance``. Among trials that return an orientation, the one with the
    smallest measured makespan is kept.

    Args:
        instance: Graph to orient.
        locked: Edges already oriented, mapping each such edge id to its head.
        start: ``"light"`` or ``"random"``.
        rng: Generator used when ``start`` is ``"random"``.
        tolerance: Width at which a fractional search stops.

    Returns:
        tuple[dict[int, int], int]: Complete orientation, and the trial count.

    Raises:
        RuntimeError: No trial produced an orientation.
    """
    locked = dict(locked or {})
    if not instance.edges or all(edge.id in locked for edge in instance.edges):
        return _complete(instance, locked, {}), 0
    if _integral(instance, locked):
        return _search_integral(instance, locked, start, rng)
    return _search_real(instance, locked, start, rng, tolerance)


def attempt(
    instance: Instance,
    threshold: float,
    locked: dict[int, int] | None = None,
    start: str = "light",
    rng: random.Random | None = None,
) -> dict[int, int] | None:
    """Try one threshold. Return a complete orientation, or ``None`` on reject.

    Args:
        instance: Graph to orient at this threshold.
        threshold: Candidate makespan ``T``.
        locked: Edges already oriented, mapping each such edge id to its head.
        start: ``"light"`` or ``"random"`` core orientation.
        rng: Generator used when ``start`` is ``"random"``.

    Returns:
        dict[int, int] | None: Complete orientation, or ``None`` when the trial rejects.
    """
    locked = dict(locked or {})
    load = _locked_load(instance, locked)
    assign = dict(locked)
    free = {edge.id for edge in instance.edges if edge.id not in locked}
    if not _peel(instance, threshold, load, assign, free):
        return None
    _orient_core(instance, assign, free, load, start, rng)
    if not _propagate(instance, threshold, load, assign, free):
        return None
    return _complete(instance, locked, assign)


def _search_integral(
    instance: Instance,
    locked: dict[int, int],
    start: str,
    rng: random.Random | None,
) -> tuple[dict[int, int], int]:
    """Integer bisection. A success narrows the ceiling; a reject raises the floor.

    Args:
        instance: Graph to orient.
        locked: Edges already oriented, mapping each such edge id to its head.
        start: ``"light"`` or ``"random"``.
        rng: Generator used when ``start`` is ``"random"``.

    Returns:
        tuple[dict[int, int], int]: Best orientation and the number of trials.

    Raises:
        RuntimeError: No integer threshold produced an orientation.
    """
    free = [edge for edge in instance.edges if edge.id not in locked]
    light = max(min(edge.w_u, edge.w_v) for edge in free)
    ceiling = sum(max(edge.w_u, edge.w_v) for edge in free)
    ceiling += sum(_locked_load(instance, locked).values())
    lo = max(1, int(round(light)))
    hi = int(round(ceiling))
    if lo > hi:
        lo = hi
    best_heads: dict[int, int] | None = None
    best_score: float | None = None
    trials = 0
    while lo <= hi:
        threshold = (lo + hi) // 2
        trials += 1
        found = attempt(instance, float(threshold), locked, start, rng)
        if found is None:
            lo = threshold + 1
            continue
        score = makespan(instance, found)
        if best_score is None or score < best_score:
            best_score = score
            best_heads = found
        hi = threshold - 1
    if best_heads is None:
        raise RuntimeError("ratio-greedy found no orientation")
    return best_heads, trials


def _search_real(
    instance: Instance,
    locked: dict[int, int],
    start: str,
    rng: random.Random | None,
    tolerance: float,
) -> tuple[dict[int, int], int]:
    """Real bisection down to ``tolerance``, seeded by a trial at the ceiling.

    Args:
        instance: Graph to orient.
        locked: Edges already oriented, mapping each such edge id to its head.
        start: ``"light"`` or ``"random"``.
        rng: Generator used when ``start`` is ``"random"``.
        tolerance: Width at which the search stops.

    Returns:
        tuple[dict[int, int], int]: Best orientation and the number of trials.

    Raises:
        RuntimeError: The ceiling trial produced no orientation.
    """
    free = [edge for edge in instance.edges if edge.id not in locked]
    lo = max(min(edge.w_u, edge.w_v) for edge in free)
    hi = sum(max(edge.w_u, edge.w_v) for edge in free)
    hi += sum(_locked_load(instance, locked).values())
    trials = 1
    found = attempt(instance, hi, locked, start, rng)
    if found is None:
        raise RuntimeError("ratio-greedy found no orientation at the weight ceiling")
    best_heads = found
    best_score = makespan(instance, found)
    while hi - lo > tolerance:
        threshold = (lo + hi) / 2.0
        if threshold <= lo or threshold >= hi:
            break
        trials += 1
        found = attempt(instance, threshold, locked, start, rng)
        if found is None:
            lo = threshold
            continue
        score = makespan(instance, found)
        if score < best_score:
            best_score = score
            best_heads = found
        hi = threshold
    return best_heads, trials


def _peel(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
) -> bool:
    """Peel pendant nodes until none remain. Reject when a peel exceeds ``T``.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids still unoriented. Peeled ids are removed.

    Returns:
        bool: False when a peeled bundle pushes an endpoint above ``T``.
    """
    changed = True
    while changed:
        changed = False
        for node in instance.nodes:
            neighbors = _free_neighbors(instance, node, free)
            if len(neighbors) != 1:
                continue
            parent = next(iter(neighbors))
            bundle = [
                edge
                for edge in instance.edges
                if edge.id in free and node in edge.endpoints()
            ]
            bundle.sort(
                key=lambda edge: (
                    _ratio(edge.weight_toward(node), edge.weight_toward(parent)),
                    -edge.weight_toward(node),
                    -edge.weight_toward(parent),
                )
            )
            for edge in bundle:
                if load[node] + edge.weight_toward(node) <= threshold:
                    assign[edge.id] = node
                    load[node] += edge.weight_toward(node)
                else:
                    assign[edge.id] = parent
                    load[parent] += edge.weight_toward(parent)
                free.discard(edge.id)
            if load[parent] > threshold or load[node] > threshold:
                return False
            changed = True
    return True


def _orient_core(
    instance: Instance,
    assign: dict[int, int],
    free: set[int],
    load: dict[int, float],
    start: str,
    rng: random.Random | None,
) -> None:
    """Orient every still-free edge and add its weight to the chosen endpoint.

    The edge stays free, so a later pass may move it.

    Args:
        instance: Graph being oriented.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids that repair is allowed to move.
        load: Current load of every node. Updated in place.
        start: ``"light"`` or ``"random"``.
        rng: Generator used when ``start`` is ``"random"``.
    """
    for edge in instance.edges:
        if edge.id not in free:
            continue
        if start == "light":
            head = edge.lighter_head()
        else:
            if rng is None:
                raise RuntimeError("a random core requires a generator")
            head = rng.choice(edge.endpoints())
        assign[edge.id] = head
        load[head] += edge.weight_toward(head)


def _propagate(
    instance: Instance,
    threshold: float,
    load: dict[int, float],
    assign: dict[int, int],
    free: set[int],
) -> bool:
    """Move free edges off overloaded nodes for one pass per component node.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
        load: Current load of every node. Updated in place.
        assign: Heads chosen so far. Updated in place.
        free: Edge ids that repair is allowed to move.

    Returns:
        bool: True when every node load is at most ``T`` after the passes.
    """
    sizes = _component_sizes(instance)
    if not sizes:
        return True
    for turn in range(max(sizes.values())):
        if all(value <= threshold for value in load.values()):
            return True
        visited: set[int] = set()
        while True:
            waiting = [
                node
                for node in instance.nodes
                if load[node] > threshold
                and node not in visited
                and turn < sizes[node]
            ]
            if not waiting:
                break
            node = max(waiting, key=lambda item: load[item])
            visited.add(node)
            movable = [
                edge
                for edge in instance.edges
                if edge.id in free and assign.get(edge.id) == node
            ]
            movable.sort(
                key=lambda edge: (
                    _descending_ratio(
                        edge.weight_toward(node),
                        edge.weight_toward(edge.other(node)),
                    ),
                    -edge.weight_toward(node),
                )
            )
            for edge in movable:
                if load[node] <= threshold:
                    break
                other = edge.other(node)
                if other in visited:
                    continue
                assign[edge.id] = other
                load[node] -= edge.weight_toward(node)
                load[other] += edge.weight_toward(other)
    return all(value <= threshold for value in load.values())


def _component_sizes(instance: Instance) -> dict[int, int]:
    """Number of nodes in the component of each node, counting every edge.

    Args:
        instance: Graph whose components are measured.

    Returns:
        dict[int, int]: Component size of every node. Empty when there are no nodes.
    """
    parent = {node: node for node in instance.nodes}

    def find(node: int) -> int:
        """Root of ``node`` with path compression.

        Args:
            node: Node whose component root is requested.

        Returns:
            int: Representative of that component.
        """
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in instance.edges:
        parent[find(edge.u)] = find(edge.v)
    totals: dict[int, int] = {}
    for node in instance.nodes:
        root = find(node)
        totals[root] = totals.get(root, 0) + 1
    return {node: totals[find(node)] for node in instance.nodes}


def _free_neighbors(instance: Instance, node: int, free: set[int]) -> set[int]:
    """Endpoints reached from ``node`` by an edge that is still free.

    Args:
        instance: Graph being oriented.
        node: Node whose free neighborhood is requested.
        free: Edge ids that have not been peeled.

    Returns:
        set[int]: Distinct free neighbors of ``node``.
    """
    found = set()
    for edge in instance.incident(node):
        if edge.id in free:
            found.add(edge.other(node))
    return found


def _locked_heads(ctx: SolveContext) -> dict[int, int]:
    """Heads of edges a previous step has frozen.

    Args:
        ctx: Solve state whose locked edges are copied.

    Returns:
        dict[int, int]: Map from each locked edge id to its head.
    """
    heads: dict[int, int] = {}
    for edge_id in ctx.locked:
        head = ctx.orientation[edge_id]
        if head is None:
            raise RuntimeError(f"locked edge {edge_id} has no head")
        heads[edge_id] = head
    return heads


def _locked_load(instance: Instance, locked: dict[int, int]) -> dict[int, float]:
    """Load contributed by edges that are already oriented.

    Args:
        instance: Graph whose nodes receive load.
        locked: Edges already oriented, mapping each such edge id to its head.

    Returns:
        dict[int, float]: Locked load of every node, including zeros.
    """
    total = {node: 0.0 for node in instance.nodes}
    for edge_id, head in locked.items():
        edge = instance.edge(edge_id)
        total[head] += edge.weight_toward(head)
    return total


def _integral(instance: Instance, locked: dict[int, int]) -> bool:
    """True when every weight and every locked load is an integer.

    Args:
        instance: Graph whose weights are tested.
        locked: Edges already oriented, mapping each such edge id to its head.

    Returns:
        bool: True when the integer bisection applies.
    """
    if not instance.weights_are_integral():
        return False
    return all(
        is_integral_number(value)
        for value in _locked_load(instance, locked).values()
    )


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
