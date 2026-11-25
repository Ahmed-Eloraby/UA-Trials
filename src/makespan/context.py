"""Mutable state shared by the steps of one solve.

A later algorithm can lock some edges, then run an LP, a rounding rule, or a
flow search on whatever remains. Locked edges contribute base load and are
left unchanged. The incumbent is the best complete orientation seen so far.
"""

from __future__ import annotations

from math import inf

from makespan.model import (
    ABS_TOL,
    Assignment,
    Instance,
    Solution,
    loads,
    makespan,
    trivial_lower_bound,
)


class SolveContext:
    """Partial orientation plus the best complete orientation found so far.

    Args:
        instance: Graph being solved. Edges start unlocked and unoriented.
    """

    def __init__(self, instance: Instance):
        """Start from an unoriented instance and the analytic lower bound.

        Args:
            instance: Graph being solved. Edges start unlocked and unoriented.
        """
        self.instance = instance
        self.orientation: dict[int, int | None] = {
            edge.id: None for edge in instance.edges
        }
        self.locked: set[int] = set()
        self.fractional: dict[int, float] = {}
        self.best_so_far_assignment: Assignment | None = None
        self.best_so_far_makespan: float = inf
        self.lower_bound: float = trivial_lower_bound(instance)
        self.upper_bound: float = inf
        self.metadata: dict = {}

    def edge_is_locked(self, edge_id: int) -> bool:
        """True when a previous step has frozen this edge.

        Args:
            edge_id: Identifier of the edge to test.

        Returns:
            bool: True when ``edge_id`` is in the locked set.
        """
        return edge_id in self.locked

    def free_edges(self):
        """Edges the current step is allowed to reorient.

        Returns:
            list: Unlocked :class:`~makespan.model.Edge` objects, in input order.
        """
        return [edge for edge in self.instance.edges if edge.id not in self.locked]

    def lock(self, edge_id: int, head: int) -> None:
        """Fix ``edge_id`` toward ``head`` so later steps leave it alone.

        Args:
            edge_id: Identifier of the edge to freeze.
            head: Endpoint that receives this edge. Must be ``u`` or ``v``.
        """
        edge = self.instance.edge(edge_id)
        if head != edge.u and head != edge.v:
            raise ValueError(f"edge {edge_id} cannot be oriented toward {head}")
        self.orientation[edge_id] = head
        self.locked.add(edge_id)

    def base_load(self) -> dict[int, float]:
        """Load already committed by locked edges.

        Returns:
            dict[int, float]: Locked load of every node, including zeros.
        """
        total = {node: 0.0 for node in self.instance.nodes}
        for edge in self.instance.edges:
            if edge.id not in self.locked:
                continue
            head = self.orientation[edge.id]
            if head is None:
                raise RuntimeError(f"locked edge {edge.id} has no head")
            total[head] += edge.weight_toward(head)
        return total

    def assignment_from(self, free_heads: dict[int, int]) -> dict[int, int]:
        """Combine locked heads with a proposal for every unlocked edge.

        Args:
            free_heads: Map from each unlocked edge id to the chosen endpoint.

        Returns:
            dict[int, int]: Complete orientation of every edge.
        """
        heads: dict[int, int] = {}
        for edge in self.instance.edges:
            if edge.id in self.locked:
                head = self.orientation[edge.id]
                if head is None:
                    raise RuntimeError(f"locked edge {edge.id} has no head")
                heads[edge.id] = head
            else:
                try:
                    heads[edge.id] = free_heads[edge.id]
                except KeyError as exc:
                    raise RuntimeError(f"edge {edge.id} was not oriented") from exc
        return heads

    def analytic_lower_bound(self) -> float:
        """Lower bound from lighter weights and load already locked in.

        Returns:
            float: The larger of the stored lower bound and the heaviest locked load.
        """
        bound = self.lower_bound
        base = self.base_load()
        if base:
            bound = max(bound, max(base.values()))
        return bound

    def consider(self, heads: dict[int, int]) -> bool:
        """Keep ``heads`` when its evaluated makespan improves the incumbent.

        The upper bound tightens to the same value. A tie keeps the earlier
        orientation, so the result stays deterministic.

        Args:
            heads: Complete orientation from each edge id to its head.

        Returns:
            bool: True when ``heads`` replaced the incumbent.
        """
        score = makespan(self.instance, heads)
        if (
            self.best_so_far_assignment is not None
            and score >= self.best_so_far_makespan - ABS_TOL
        ):
            return False
        self.best_so_far_assignment = Assignment(heads)
        self.best_so_far_makespan = score
        if score < self.upper_bound:
            self.upper_bound = score
        return True

    def raise_lower_bound(self, value: float) -> None:
        """Raise the lower bound, never past a complete incumbent.

        Args:
            value: Candidate lower bound. Ignored when it is not an improvement.
        """
        if (
            self.best_so_far_assignment is not None
            and value > self.best_so_far_makespan
        ):
            value = self.best_so_far_makespan
        if value > self.lower_bound:
            self.lower_bound = value

    def to_solution(self, algorithm: str) -> Solution:
        """Build a solution from the incumbent orientation.

        The makespan is recomputed from the heads, so the reported value is
        the load of that orientation rather than a solver's internal objective.

        Args:
            algorithm: Name stored on the returned :class:`Solution`.

        Returns:
            Solution: Incumbent orientation with evaluated makespan and bounds.
        """
        if self.best_so_far_assignment is None:
            raise RuntimeError(f"{algorithm} did not produce an orientation")
        heads = dict(self.best_so_far_assignment.edge_node_map)
        node_loads = loads(self.instance, heads)
        score = max(node_loads.values()) if node_loads else 0.0
        upper = self.upper_bound if self.upper_bound < inf else score
        return Solution(
            algorithm=algorithm,
            instance_name=self.instance.name,
            makespan=score,
            heads=heads,
            loads=node_loads,
            lower_bound=self.lower_bound,
            upper_bound=upper,
            metadata=dict(self.metadata),
        )
