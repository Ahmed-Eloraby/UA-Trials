"""LP-free balance search that orients residual edges by breaking cycles.

The outer search bisects a threshold ``T``. ``proven_lower`` moves only when
a reduction, or a bridge budget check, proves that no orientation has
makespan at most ``T``. A cycle flip is a choice among edges that still fit
on both ends, so an overflow after that choice does not move the fence.
``upper`` is the best measured makespan, including an orientation finished
by the load-greedy rule.

One trial runs the same reductions as ``balance-lp``: low-load stars,
forced edges, and leaf bundles. What remains is oriented as follows.

* ``balance-cycle`` breaks the shortest intact cycle. Each pass keeps both
  directions, reverses the heavier top edge in the direction whose reverse
  changes the makespan less, then settles forced edges. Leaf pruning runs
  again once a channel on that cycle is empty.
* ``balance-bridge`` peels bridge channels first. A channel is split using
  the spare capacity of each whole side, not only the two attachment nodes.
  Cycles inside the blocks are then broken the same way.
"""

from __future__ import annotations

from functools import cmp_to_key
from math import inf

from makespan.balance import (
    _FIT,
    _leaf_sort_key,
    force_closure,
    leaf_pruning,
    low_load_assignment,
)
from makespan.context import SolveContext
from makespan.factory import Step
from makespan.graph_algorithms.components import find_components, split_at_bridges
from makespan.graph_algorithms.cycles import find_all_simple_cycles
from makespan.greedy import _lower_resulting_load
from makespan.model import Edge, Instance, makespan


# Channels longer than this are split by the leaf ratio instead of enumeration.
_BRIDGE_ENUM_LIMIT = 20
# The probe window is closed once it is thinner than this relative gap.
_PROBE_GAP = 1e-9


class CycleTrial:
    """Outcome of one threshold trial.

    Args:
        status: ``"accept"`` when the trial produced an orientation, or
            ``"reject"`` when ``T`` is proved infeasible.
        heads: Complete orientation, or ``None`` for a reject.
        certified: True when every locked edge was forced by a sound rule.
            A false value means a later overflow must not raise the lower fence.

    Returns:
        CycleTrial: Frozen trial outcome.
    """

    __slots__ = ("status", "heads", "certified")

    def __init__(
        self,
        status: str,
        heads: dict[int, int] | None,
        certified: bool,
    ):
        """Store the status, the orientation, and whether the proof is intact.

        Args:
            status: ``"accept"`` or ``"reject"``.
            heads: Complete orientation, or ``None`` for a reject.
            certified: True when no optional edge has been locked.
        """
        self.status = status
        self.heads = heads
        self.certified = certified


class CycleBalanceSearchStep(Step):
    """Fenced threshold search that breaks residual cycles instead of rounding an LP.

    ``use_bridges`` selects the bridge-aware variant. ``eps`` is the relative
    gap at which the proven lower fence and the measured upper bound are
    close enough to stop.

    Args:
        use_bridges: When true, peel bridge channels before breaking cycles.
        eps: Relative gap at which the search fences are close enough to stop.
    """

    def __init__(self, use_bridges: bool, eps: float = 0.25):
        """Store the bridge switch and the stopping gap.

        Args:
            use_bridges: When true, peel bridge channels before breaking cycles.
            eps: Relative gap at which the search fences are close enough to stop.
        """
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.use_bridges = use_bridges
        self.eps = eps

    def run(self, ctx: SolveContext) -> None:
        """Search thresholds and record the incumbent, the fences, and the trials.

        Args:
            ctx: Solve state that receives the incumbent orientation and fences.
        """
        instance = ctx.instance
        if ctx.best_so_far_assignment is None:
            ctx.consider({edge.id: edge.lighter_head() for edge in instance.edges})
        if ctx.best_so_far_assignment is None:
            raise RuntimeError("cycle balance search has no incumbent orientation")
        upper = ctx.best_so_far_makespan
        proven = ctx.analytic_lower_bound()
        probe = proven
        incumbent = dict(ctx.best_so_far_assignment.edge_node_map)
        trials = 0
        certified_search = True
        while upper > (1.0 + self.eps) * proven:
            if upper - probe <= _PROBE_GAP * max(1.0, upper):
                break
            threshold = (probe + upper) / 2.0
            if threshold <= probe or threshold >= upper:
                break
            proven_before = proven
            upper_before = upper
            trials += 1
            trial = decide_cycle(instance, threshold, self.use_bridges)
            if trial.status == "reject":
                proven = threshold
                probe = proven
            else:
                if trial.heads is None:
                    raise RuntimeError("cycle balance trial accepted without an orientation")
                measured = makespan(instance, trial.heads)
                if measured < upper:
                    incumbent = dict(trial.heads)
                    upper = measured
                if not trial.certified:
                    certified_search = False
                if (not trial.certified) and measured > threshold + _FIT:
                    probe = threshold
                else:
                    probe = proven
                if probe > upper:
                    probe = proven
            if proven == proven_before and upper == upper_before:
                break
        ctx.consider(incumbent)
        ctx.raise_lower_bound(proven)
        ctx.metadata["cycle_lower"] = proven
        ctx.metadata["cycle_upper"] = upper
        ctx.metadata["cycle_trials"] = trials
        ctx.metadata["cycle_certified"] = certified_search
        ctx.metadata["cycle_bridges"] = self.use_bridges


def decide_cycle(
    instance: Instance,
    threshold: float,
    use_bridges: bool,
) -> CycleTrial:
    """Try one threshold and accept an orientation or reject on a sound proof.

    Reductions run until they stop changing the free edges. Bridge channels
    are peeled next when ``use_bridges`` is set. The shortest intact cycle is
    then broken. A pass that frees nothing finishes by sending each leftover
    edge to the endpoint with the smaller resulting load, and that finish is
    uncertified.

    Args:
        instance: Graph being oriented at this trial.
        threshold: Candidate makespan ``T`` used as residual capacity.
        use_bridges: When true, split bridge channels before breaking cycles.

    Returns:
        CycleTrial: ``reject`` with no orientation, or ``accept`` with heads.
    """
    state = _Residual(instance, threshold)
    while state.free:
        before = set(state.free)
        if state.reduce() == "reject":
            if state.certified:
                return CycleTrial("reject", None, True)
            return CycleTrial("accept", state.greedy_heads(), False)
        if not state.free:
            break
        if use_bridges:
            peeled = state.peel_bridges()
            if peeled == "reject":
                return CycleTrial("reject", None, True)
            if peeled == "fail":
                return CycleTrial("accept", state.greedy_heads(), False)
            if not state.free:
                break
        broken = state.break_one_cycle()
        if broken == "reject":
            return CycleTrial("reject", None, True)
        if broken == "fail":
            return CycleTrial("accept", state.greedy_heads(), False)
        if not state.free:
            break
        if state.free == before:
            return CycleTrial("accept", state.greedy_heads(), False)
    return CycleTrial("accept", state.heads(), state.certified)


class _Residual:
    """Free edges, residual capacities, and the certified-proof flag of one trial.

    Args:
        instance: Graph being oriented.
        threshold: Candidate makespan ``T``.
    """

    def __init__(self, instance: Instance, threshold: float):
        """Start with every edge free and every node able to accept ``T``.

        Args:
            instance: Graph being oriented.
            threshold: Candidate makespan ``T``.
        """
        self.instance = instance
        self.threshold = threshold
        self.cap = {node: float(threshold) for node in instance.nodes}
        self.orientation: dict[int, int | None] = {
            edge.id: None for edge in instance.edges
        }
        self.free = {edge.id for edge in instance.edges}
        self.certified = True

    def reduce(self) -> str:
        """Run low-load, forced edges, and leaf bundles until a pass fixes nothing.

        Returns:
            str: ``"reject"`` when a reduction proves ``T`` infeasible, else ``"accept"``.
        """
        while True:
            before = set(self.free)
            low_load_assignment(self.instance, self.free, self.cap, self.orientation)
            if force_closure(self.instance, self.free, self.cap, self.orientation) == "reject":
                return "reject"
            status, _closed = leaf_pruning(
                self.instance, self.free, self.cap, self.orientation, self.threshold
            )
            if status == "reject":
                return "reject"
            if self.free == before:
                return "accept"

    def peel_bridges(self) -> str:
        """Split every leaf bridge channel, then reduce what that split frees.

        Returns:
            str: ``"none"`` when no bridge remains, ``"progress"`` when a channel
            was split, ``"reject"`` on a sound budget failure, or ``"fail"``
            when the trial must finish uncertified.
        """
        moved = False
        while self.free:
            found = self._next_leaf_bridge()
            if found is None:
                return "progress" if moved else "none"
            attachment, other, side_a, side_b = found
            status = self._assign_bridge(attachment, other, side_a, side_b)
            if status != "ok":
                return status
            moved = True
            if self.reduce() == "reject":
                return "reject" if self.certified else "fail"
        return "progress" if moved else "none"

    def break_one_cycle(self) -> str:
        """Break the shortest intact cycle, trying the next cycle if one cannot move.

        Returns:
            str: ``"progress"``, ``"stuck"``, ``"reject"``, or ``"fail"``.
        """
        for cycle in self._intact_cycles():
            status = self._break_cycle(cycle)
            if status != "stuck":
                return status
        return "stuck"

    def greedy_heads(self) -> dict[int, int]:
        """Orient every still-unset edge toward the smaller resulting load.

        Returns:
            dict[int, int]: Complete orientation.
        """
        load = self._loads()
        pending = [
            edge for edge in self.instance.edges if self.orientation[edge.id] is None
        ]
        pending.sort(
            key=lambda edge: (
                -max(edge.w_u, edge.w_v),
                -abs(edge.w_u - edge.w_v),
                edge.id,
            )
        )
        for edge in pending:
            head = _lower_resulting_load(edge, load)
            self.orientation[edge.id] = head
            load[head] += edge.weight_toward(head)
        self.free.clear()
        return self.heads()

    def heads(self) -> dict[int, int]:
        """Copy the trial orientation.

        Returns:
            dict[int, int]: Complete orientation of every edge.

        Raises:
            RuntimeError: An edge is still unoriented.
        """
        assigned: dict[int, int] = {}
        for edge in self.instance.edges:
            head = self.orientation[edge.id]
            if head is None:
                raise RuntimeError(f"edge {edge.id} was left unoriented")
            assigned[edge.id] = head
        return assigned

    def _settle(self) -> str:
        """Repeat low-load stars and forced edges until a pass fixes nothing.

        Leaf bundles are left for the outer reduction. A reject here is sound
        only while ``certified`` is still true.

        Returns:
            str: ``"reject"`` or ``"accept"``.
        """
        while True:
            before = set(self.free)
            low_load_assignment(self.instance, self.free, self.cap, self.orientation)
            if force_closure(self.instance, self.free, self.cap, self.orientation) == "reject":
                return "reject"
            if self.free == before:
                return "accept"

    def _break_cycle(self, cycle: list[int]) -> str:
        """Reverse tops until one channel of ``cycle`` has no free edge.

        Args:
            cycle: Vertex order of one simple cycle, without a repeated close.

        Returns:
            str: ``"progress"``, ``"stuck"``, ``"reject"``, or ``"fail"``.
        """
        if not self._channels_alive(cycle):
            return "stuck"
        ordered = self._ordered_directions(cycle)
        committed = False
        while self._channels_alive(cycle):
            proposal = self._proposal(cycle, ordered)
            if proposal is None:
                return "progress" if committed else "stuck"
            edge, head = proposal
            optional = self._fits(edge, edge.u) and self._fits(edge, edge.v)
            self._commit(edge, head)
            committed = True
            if optional:
                self.certified = False
            if self._settle() == "reject":
                return "reject" if self.certified else "fail"
        return "progress"

    def _proposal(
        self,
        cycle: list[int],
        ordered: dict[tuple[int, int], list[Edge]],
    ) -> tuple[Edge, int] | None:
        """Choose the reverse whose committed makespan is smaller.

        Each direction proposes its heaviest remaining top. The reverse sends
        that edge to the opposite endpoint. A reverse that does not fit is
        skipped. When neither reverse fits, the caller leaves the cycle.

        Args:
            cycle: Vertex order of the cycle being broken.
            ordered: Per-direction channel lists, heaviest top first.

        Returns:
            tuple[Edge, int] | None: Edge and head to commit, or ``None``.
        """
        proposals: list[tuple[float, int, int, Edge, int]] = []
        for direction in (1, -1):
            top = self._heaviest_top(cycle, ordered, direction)
            if top is None:
                continue
            edge, node = top
            head = edge.other(node)
            if not self._fits(edge, head):
                continue
            score = self._makespan_if([(head, edge.weight_toward(head))])
            proposals.append((score, edge.id, direction, edge, head))
        if not proposals:
            return None
        proposals.sort()
        _score, _edge_id, _direction, edge, head = proposals[0]
        return edge, head

    def _heaviest_top(
        self,
        cycle: list[int],
        ordered: dict[tuple[int, int], list[Edge]],
        direction: int,
    ) -> tuple[Edge, int] | None:
        """Heaviest still-free top among the nodes, in this direction.

        Args:
            cycle: Vertex order of the cycle.
            ordered: Per-direction channel lists, heaviest top first.
            direction: ``1`` toward the successor, ``-1`` toward the predecessor.

        Returns:
            tuple[Edge, int] | None: The top edge and the node that lists it.
        """
        best: tuple[Edge, int] | None = None
        for index, node in enumerate(cycle):
            for edge in ordered[(direction, index)]:
                if edge.id not in self.free:
                    continue
                pair = (edge, node)
                if best is None or _heavier_first(pair, best) < 0:
                    best = pair
                break
        return best

    def _ordered_directions(
        self, cycle: list[int]
    ) -> dict[tuple[int, int], list[Edge]]:
        """Sort each node's channel once in each direction.

        The key depends only on weights, so later passes drop edges that are
        no longer free and keep this order.

        Args:
            cycle: Vertex order of the cycle.

        Returns:
            dict[tuple[int, int], list[Edge]]: Lists keyed by direction and index.
        """
        ordered: dict[tuple[int, int], list[Edge]] = {}
        count = len(cycle)
        for direction in (1, -1):
            for index, node in enumerate(cycle):
                neighbor = cycle[(index + direction) % count]
                ordered[(direction, index)] = _sort_toward(
                    self._channel(node, neighbor), node
                )
        return ordered

    def _intact_cycles(self) -> list[list[int]]:
        """Shortest intact simple cycles in the free graph, shortest first.

        Returns:
            list[list[int]]: Cycles whose every channel still has a free edge.
        """
        adjacency = self._simple_adj()
        found: list[list[int]] = []
        for component in find_components(adjacency):
            if len(component) < 3:
                continue
            sub = {
                node: {neighbor for neighbor in adjacency[node] if neighbor in component}
                for node in component
            }
            for cycle in find_all_simple_cycles(sub):
                if self._channels_alive(cycle):
                    found.append(cycle)
        found.sort(key=lambda cycle: (len(cycle), tuple(cycle)))
        return found

    def _channels_alive(self, cycle: list[int]) -> bool:
        """True when every consecutive pair on ``cycle`` still has a free edge.

        Args:
            cycle: Vertex order of one simple cycle.

        Returns:
            bool: True when no channel on the cycle is empty.
        """
        count = len(cycle)
        for index in range(count):
            if not self._channel(cycle[index], cycle[(index + 1) % count]):
                return False
        return True

    def _next_leaf_bridge(
        self,
    ) -> tuple[int, int, set[int], set[int]] | None:
        """First leaf bridge in the free simple graph.

        The leaf attachment is the endpoint that lies in the block with only
        one incident bridge. The other side is the rest of that component.

        Returns:
            tuple[int, int, set[int], set[int]] | None: Attachment, other end,
            leaf block, and the complementary side. ``None`` when no bridge remains.
        """
        adjacency = self._simple_adj()
        for component in find_components(adjacency):
            if len(component) < 2:
                continue
            sub = {
                node: {neighbor for neighbor in adjacency[node] if neighbor in component}
                for node in component
            }
            bridges, blocks = split_at_bridges(sub)
            if not bridges:
                continue
            node_block: dict[int, int] = {}
            for index, block in enumerate(blocks):
                for node in block:
                    node_block[node] = index
            incident: list[list[tuple[int, int]]] = [[] for _block in blocks]
            for left, right in bridges:
                incident[node_block[left]].append((left, right))
                incident[node_block[right]].append((left, right))
            choices = []
            for index, links in enumerate(incident):
                if len(links) != 1:
                    continue
                left, right = links[0]
                if left in blocks[index]:
                    attachment, other = left, right
                else:
                    attachment, other = right, left
                if attachment == other or not self._channel(attachment, other):
                    continue
                leaf = set(blocks[index])
                rest = set(component) - leaf
                choices.append(
                    (
                        min(attachment, other),
                        max(attachment, other),
                        attachment,
                        other,
                        leaf,
                        rest,
                    )
                )
            if not choices:
                continue
            choices.sort(key=lambda item: item[:4])
            _lo, _hi, attachment, other, leaf, rest = choices[0]
            return attachment, other, leaf, rest
        return None

    def _assign_bridge(
        self,
        attachment: int,
        other: int,
        side_a: set[int],
        side_b: set[int],
    ) -> str:
        """Split the free channel between ``attachment`` and ``other``.

        The weight placed on an attachment cannot exceed that node's residual
        capacity or the spare capacity of its side. Spare capacity is the
        total residual room minus the lighter weight of every other free edge
        inside the side. No such split is a sound reject while the trial is
        certified. Several splits are optional, so the trial stops being a proof.

        Args:
            attachment: Endpoint in the leaf block.
            other: Endpoint on the complementary side.
            side_a: Nodes of the leaf block.
            side_b: Nodes of the complementary side.

        Returns:
            str: ``"ok"``, ``"reject"``, or ``"fail"``.
        """
        channel = self._channel(attachment, other)
        channel.sort(key=lambda edge: edge.id)
        if not channel:
            return "ok"
        cut = {edge.id for edge in channel}
        budget_a = self._budget(side_a, cut)
        budget_b = self._budget(side_b, cut)
        if budget_a < -_FIT or budget_b < -_FIT:
            return "reject" if self.certified else "fail"
        allow_a = min(self.cap[attachment], budget_a)
        allow_b = min(self.cap[other], budget_b)
        if len(channel) > _BRIDGE_ENUM_LIMIT:
            return self._assign_long_bridge(channel, attachment, other, allow_a, allow_b)
        mask = self._best_mask(
            channel, attachment, other, allow_a, allow_b, budget_a, budget_b
        )
        if mask is None:
            return "reject" if self.certified else "fail"
        for index, edge in enumerate(channel):
            head = attachment if (mask >> index) & 1 else other
            self._commit(edge, head)
        return "ok"

    def _assign_long_bridge(
        self,
        channel: list[Edge],
        attachment: int,
        other: int,
        allow_a: float,
        allow_b: float,
    ) -> str:
        """Split a long channel by the leaf-bundle ratio.

        The prefix that fits on the leaf attachment is kept there. When the
        tail does not fit on the other side, the caller finishes uncertified
        instead of rejecting.

        Args:
            channel: Free edges between the two attachments, in edge-id order.
            attachment: Leaf-block endpoint, used as the leaf in the ratio.
            other: Endpoint that receives the tail.
            allow_a: Most weight the leaf attachment may take.
            allow_b: Most weight the other attachment may take.

        Returns:
            str: ``"ok"`` when the prefix split fits, else ``"fail"``.
        """
        ordered = sorted(channel, key=lambda edge: _leaf_sort_key(edge, attachment, other))
        prefix: list[Edge] = []
        prefix_weight = 0.0
        for edge in ordered:
            weight = edge.weight_toward(attachment)
            if prefix_weight + weight <= allow_a + _FIT:
                prefix.append(edge)
                prefix_weight += weight
            else:
                break
        prefix_ids = {edge.id for edge in prefix}
        tail = [edge for edge in ordered if edge.id not in prefix_ids]
        tail_weight = sum(edge.weight_toward(other) for edge in tail)
        if tail_weight > allow_b + _FIT:
            return "fail"
        self.certified = False
        for edge in prefix:
            self._commit(edge, attachment)
        for edge in tail:
            self._commit(edge, other)
        return "ok"

    def _best_mask(
        self,
        channel: list[Edge],
        attachment: int,
        other: int,
        allow_a: float,
        allow_b: float,
        budget_a: float,
        budget_b: float,
    ) -> int | None:
        """Mask of edges sent to ``attachment`` in the best feasible split.

        Bit ``i`` set means channel edge ``i`` is oriented toward the leaf
        attachment. Ties prefer the side with more spare budget.

        Args:
            channel: Free bridge edges, in edge-id order.
            attachment: Leaf-block endpoint.
            other: Complementary endpoint.
            allow_a: Most weight the leaf attachment may take.
            allow_b: Most weight the other attachment may take.
            budget_a: Spare capacity of the leaf side.
            budget_b: Spare capacity of the complementary side.

        Returns:
            int | None: Winning mask, or ``None`` when no split fits.
        """
        rich_on_attachment = budget_a >= budget_b
        best_key: tuple[float, float, int] | None = None
        best_mask: int | None = None
        feasible = 0
        span = 1 << len(channel)
        for mask in range(span):
            sum_a = 0.0
            sum_b = 0.0
            fits = True
            for index, edge in enumerate(channel):
                if (mask >> index) & 1:
                    sum_a += edge.weight_toward(attachment)
                else:
                    sum_b += edge.weight_toward(other)
                if sum_a > allow_a + _FIT or sum_b > allow_b + _FIT:
                    fits = False
                    break
            if not fits:
                continue
            feasible += 1
            score = self._makespan_if([(attachment, sum_a), (other, sum_b)])
            on_rich = sum_a if rich_on_attachment else sum_b
            key = (score, -on_rich, mask)
            if best_key is None or key < best_key:
                best_key = key
                best_mask = mask
        if feasible == 0:
            return None
        if feasible > 1:
            self.certified = False
        return best_mask

    def _budget(self, side: set[int], cut: set[int]) -> float:
        """Residual room of ``side`` after paying the lighter internal edges.

        Args:
            side: Nodes whose total residual capacity is pooled.
            cut: Edge ids of the bridge channel, which are not internal.

        Returns:
            float: Spare capacity of the side. Negative means the side cannot
            hold its own free edges.
        """
        room = sum(self.cap[node] for node in side)
        payable = 0.0
        for edge in self.instance.edges:
            if edge.id not in self.free or edge.id in cut:
                continue
            if edge.u in side and edge.v in side:
                payable += min(edge.weight_toward(edge.u), edge.weight_toward(edge.v))
        return room - payable

    def _commit(self, edge: Edge, head: int) -> None:
        """Lock ``edge`` toward ``head`` and shrink that residual capacity.

        Args:
            edge: Free edge to orient.
            head: Endpoint that receives ``edge``.
        """
        self.orientation[edge.id] = head
        self.cap[head] -= edge.weight_toward(head)
        self.free.remove(edge.id)

    def _fits(self, edge: Edge, node: int) -> bool:
        """Whether ``edge`` still fits on ``node``.

        Args:
            edge: Edge whose weight toward ``node`` is tested.
            node: Candidate head.

        Returns:
            bool: True when the weight is within the residual capacity.
        """
        return edge.weight_toward(node) <= self.cap[node] + _FIT

    def _loads(self) -> dict[int, float]:
        """Load already committed by oriented edges.

        Returns:
            dict[int, float]: Committed load of every node.
        """
        total = {node: 0.0 for node in self.instance.nodes}
        for edge in self.instance.edges:
            head = self.orientation[edge.id]
            if head is not None:
                total[head] += edge.weight_toward(head)
        return total

    def _makespan_if(self, additions: list[tuple[int, float]]) -> float:
        """Makespan after adding weight on top of the committed loads.

        Args:
            additions: Pairs of node and extra weight.

        Returns:
            float: Maximum load after those additions.
        """
        total = self._loads()
        for node, weight in additions:
            total[node] += weight
        if not total:
            return 0.0
        return max(total.values())

    def _channel(self, left: int, right: int) -> list[Edge]:
        """Free edges with endpoints ``left`` and ``right``.

        Args:
            left: One endpoint.
            right: The other endpoint.

        Returns:
            list[Edge]: Matching free edges, in input order.
        """
        pair = {left, right}
        return [
            edge
            for edge in self.instance.edges
            if edge.id in self.free and {edge.u, edge.v} == pair
        ]

    def _simple_adj(self) -> dict[int, set[int]]:
        """Undirected simple graph of the edges that are still free.

        Returns:
            dict[int, set[int]]: Neighbors of every node, ignoring parallel edges.
        """
        adjacency = {node: set() for node in self.instance.nodes}
        for edge in self.instance.edges:
            if edge.id not in self.free:
                continue
            adjacency[edge.u].add(edge.v)
            adjacency[edge.v].add(edge.u)
        return adjacency


def _direction_ratio(edge: Edge, node: int) -> float:
    """Weight toward ``node`` divided by the weight toward the other end.

    A zero weight on the other end sorts ahead of every finite ratio.

    Args:
        edge: Edge being ranked at ``node``.
        node: Endpoint whose direction is under consideration.

    Returns:
        float: Ratio used as the tie break, or infinity when the other weight is zero.
    """
    other = edge.weight_toward(edge.other(node))
    if other <= _FIT:
        return inf
    return edge.weight_toward(node) / other


def _heavier_first(left: tuple[Edge, int], right: tuple[Edge, int]) -> int:
    """Order two directed tops so the heavier one comes first.

    Weights within ``_FIT`` count as equal and fall through to the ratio.
    The edge id breaks whatever remains, so the order is stable.

    Args:
        left: Edge and the node that lists it.
        right: Edge and the node that lists it.

    Returns:
        int: Negative when ``left`` is the heavier top, positive when ``right`` is.
    """
    left_edge, left_node = left
    right_edge, right_node = right
    left_weight = left_edge.weight_toward(left_node)
    right_weight = right_edge.weight_toward(right_node)
    if abs(left_weight - right_weight) > _FIT:
        if left_weight > right_weight:
            return -1
        return 1
    left_ratio = _direction_ratio(left_edge, left_node)
    right_ratio = _direction_ratio(right_edge, right_node)
    if left_ratio > right_ratio:
        return -1
    if right_ratio > left_ratio:
        return 1
    return left_edge.id - right_edge.id


def _sort_toward(edges: list[Edge], node: int) -> list[Edge]:
    """Sort ``edges`` so the heaviest top at ``node`` is first.

    Args:
        edges: Channel edges incident to ``node``.
        node: Endpoint whose direction is under consideration.

    Returns:
        list[Edge]: ``edges`` ordered by :func:`_heavier_first`.
    """
    return sorted(
        edges,
        key=cmp_to_key(
            lambda left, right: _heavier_first((left, node), (right, node))
        ),
    )
