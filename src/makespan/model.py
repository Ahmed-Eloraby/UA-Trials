"""Orientation instances and the makespan of an assignment.

An edge carries two weights, one for each endpoint. Orienting the edge
toward an endpoint adds that weight to the endpoint's load. The makespan
is the maximum load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

ABS_TOL = 1e-9


def nearly_leq(left: float, right: float, tol: float = ABS_TOL) -> bool:
    """True when ``left`` does not exceed ``right`` beyond ``tol``.

    Args:
        left: Candidate left-hand value.
        right: Candidate right-hand value.
        tol: Additive slack allowed above ``right``.

    Returns:
        bool: True when ``left <= right + tol``.
    """
    return left <= right + tol


def is_integral_number(value: float, tol: float = 1e-8) -> bool:
    """True when ``value`` is finite and within ``tol`` of an integer.

    Args:
        value: Number to test.
        tol: Maximum distance from the nearest integer.

    Returns:
        bool: True when ``value`` is finite and within ``tol`` of an integer.
    """
    return isfinite(value) and abs(value - round(value)) <= tol


def as_number(value: float) -> int | float:
    """Return an ``int`` when ``value`` is integral, otherwise the float.

    Args:
        value: Number to render as an integer when it is integral.

    Returns:
        int | float: An ``int`` when ``value`` is integral, otherwise the float.
    """
    if is_integral_number(value):
        return int(round(value))
    return float(value)


@dataclass(frozen=True)
class Edge:
    """One undirected edge, possibly parallel to another edge.

    ``w_u`` is the load added to ``u`` when the edge is oriented toward ``u``.
    ``w_v`` is the load added to ``v`` in the other direction. Ties in the
    lightest-endpoint rule go to ``u``, the first endpoint in the input.

    Args:
        id: Unique identifier of this edge in the instance.
        u: First endpoint.
        v: Second endpoint.
        w_u: Load added to ``u`` when the edge is oriented toward ``u``.
        w_v: Load added to ``v`` when the edge is oriented toward ``v``.

    Returns:
        Edge: Frozen undirected edge with these endpoints and weights.
    """

    id: int
    u: int
    v: int
    w_u: float
    w_v: float

    def weight_toward(self, node: int) -> float:
        """Weight added to ``node`` when this edge is oriented toward it.

        Args:
            node: Endpoint that would receive this edge.

        Returns:
            float: ``w_u`` or ``w_v``, depending on which endpoint ``node`` is.
        """
        if node == self.u:
            return self.w_u
        if node == self.v:
            return self.w_v
        raise ValueError(f"node {node} is not an endpoint of edge {self.id}")

    def other(self, node: int) -> int:
        """The endpoint that is not ``node``.

        Args:
            node: One endpoint of this edge.

        Returns:
            int: The opposite endpoint.
        """
        if node == self.u:
            return self.v
        if node == self.v:
            return self.u
        raise ValueError(f"node {node} is not an endpoint of edge {self.id}")

    def lighter_head(self) -> int:
        """Endpoint with the smaller weight. Equal weights select ``u``.

        Returns:
            int: ``u`` or ``v``, whichever has the smaller incident weight.
        """
        if self.w_u <= self.w_v:
            return self.u
        return self.v

    def endpoints(self) -> tuple[int, int]:
        """``(u, v)`` in the order stored on the edge.

        Returns:
            tuple[int, int]: The two endpoints in stored order.
        """
        return self.u, self.v


@dataclass(frozen=True)
class Instance:
    """One orientation instance.

    Node ids are the ids from the builder export. They are not renumbered.

    Args:
        name: Label of this instance.
        tags: Optional tags copied from the builder export.
        nodes: Node ids in file order.
        edges: Edges of the instance, numbered ``1..m`` in file order.

    Returns:
        Instance: Frozen orientation instance with those nodes and edges.
    """

    name: str
    tags: tuple[str, ...]
    nodes: tuple[int, ...]
    edges: tuple[Edge, ...]
    _by_id: dict[int, Edge] = field(init=False, repr=False, compare=False)
    _incident: dict[int, tuple[Edge, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Index edges by id and by endpoint, and reject loops or bad weights."""
        node_set = set(self.nodes)
        if len(node_set) != len(self.nodes):
            raise ValueError(f"instance {self.name!r} has duplicate node ids")
        by_id: dict[int, Edge] = {}
        incident: dict[int, list[Edge]] = {node: [] for node in self.nodes}
        for edge in self.edges:
            if edge.id in by_id:
                raise ValueError(
                    f"instance {self.name!r} has duplicate edge id {edge.id}"
                )
            if edge.u == edge.v:
                raise ValueError(
                    f"instance {self.name!r} edge {edge.id} has equal endpoints"
                )
            if edge.u not in node_set or edge.v not in node_set:
                raise ValueError(
                    f"instance {self.name!r} edge {edge.id} references an unknown node"
                )
            if (
                not isfinite(edge.w_u)
                or not isfinite(edge.w_v)
                or edge.w_u < 0
                or edge.w_v < 0
            ):
                raise ValueError(
                    f"instance {self.name!r} edge {edge.id} has a negative weight"
                )
            by_id[edge.id] = edge
            incident[edge.u].append(edge)
            incident[edge.v].append(edge)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(
            self,
            "_incident",
            {node: tuple(edges) for node, edges in incident.items()},
        )

    def edge(self, edge_id: int) -> Edge:
        """Return the edge with this id.

        Args:
            edge_id: Identifier of the requested edge.

        Returns:
            Edge: The edge stored under ``edge_id``.
        """
        try:
            return self._by_id[edge_id]
        except KeyError as exc:
            raise KeyError(f"unknown edge id {edge_id}") from exc

    def incident(self, node: int) -> tuple[Edge, ...]:
        """Edges that have ``node`` as an endpoint, in input order.

        Args:
            node: Node whose incident edges are requested.

        Returns:
            tuple[Edge, ...]: Incident edges in input order.
        """
        try:
            return self._incident[node]
        except KeyError as exc:
            raise KeyError(f"unknown node {node}") from exc

    def weights_are_integral(self) -> bool:
        """True when every edge weight is an integer, up to a small tolerance.

        Returns:
            bool: True when every ``w_u`` and ``w_v`` is an integer.
        """
        return all(
            is_integral_number(edge.w_u) and is_integral_number(edge.w_v)
            for edge in self.edges
        )


@dataclass
class Assignment:
    """Integral orientation: each edge id maps to the endpoint that receives it.

    Args:
        edge_node_map: Map from each edge id to the endpoint that receives that edge.

    Returns:
        Assignment: Copy of this orientation.
    """

    edge_node_map: dict[int, int]

    def __post_init__(self) -> None:
        """Copy ``edge_node_map`` so later mutation of the caller's dict is ignored."""
        self.edge_node_map = dict(self.edge_node_map)


@dataclass
class Solution:
    """Result of one algorithm on one instance.

    ``makespan`` is the evaluated maximum load of ``heads``, so it is always
    achieved by that orientation. ``lower_bound`` and ``upper_bound`` bracket
    the optimum of the instance the algorithm was given.

    Args:
        algorithm: Name of the algorithm that produced this result.
        instance_name: Label of the instance that was solved.
        makespan: Maximum node load of ``heads``.
        heads: Map from each edge id to the endpoint that receives that edge.
        loads: Map from each node id to its load under ``heads``.
        lower_bound: Proven lower bound on the optimum of this instance.
        upper_bound: Upper bound on the optimum, typically the incumbent makespan.
        metadata: Extra solver data such as LP objectives or trial counts.

    Returns:
        Solution: Evaluated orientation together with its bounds and metadata.
    """

    algorithm: str
    instance_name: str
    makespan: float
    heads: dict[int, int]
    loads: dict[int, float]
    lower_bound: float
    upper_bound: float
    metadata: dict


def loads(instance: Instance, heads: Mapping[int, int]) -> dict[int, float]:
    """Load of every node under a complete orientation.

    Args:
        instance: Graph whose nodes receive load.
        heads: Map from each edge id to the endpoint that receives it.

    Returns:
        dict[int, float]: Load of every node under ``heads``.
    """
    if len(heads) != len(instance.edges):
        raise ValueError(
            f"assignment orients {len(heads)} edges, instance has {len(instance.edges)}"
        )
    total = {node: 0.0 for node in instance.nodes}
    for edge in instance.edges:
        if edge.id not in heads:
            raise ValueError(f"edge {edge.id} is unoriented")
        head = heads[edge.id]
        if head != edge.u and head != edge.v:
            raise ValueError(
                f"edge {edge.id} is oriented toward {head}, not an endpoint"
            )
        total[head] += edge.weight_toward(head)
    return total


def makespan(instance: Instance, heads: Mapping[int, int]) -> float:
    """Maximum node load. An instance with no nodes has makespan 0.

    Args:
        instance: Graph whose maximum load is evaluated.
        heads: Map from each edge id to the endpoint that receives it.

    Returns:
        float: Maximum node load, or ``0.0`` when the instance has no nodes.
    """
    if not instance.nodes:
        return 0.0
    return max(loads(instance, heads).values())


def trivial_lower_bound(instance: Instance) -> float:
    """Analytic lower bound that does not depend on a solver.

    Every edge contributes at least its lighter weight to some node, so the
    optimum is at least the heaviest of those lighter weights. The sum of the
    lighter weights is a lower bound on the total load, so the optimum is also
    at least that sum divided by the number of nodes.

    Args:
        instance: Graph whose analytic bound is computed.

    Returns:
        float: Analytic lower bound from lighter weights and average load.
    """
    if not instance.edges:
        return 0.0
    light = [min(edge.w_u, edge.w_v) for edge in instance.edges]
    bound = max(light)
    if instance.nodes:
        bound = max(bound, sum(light) / len(instance.nodes))
    return bound
