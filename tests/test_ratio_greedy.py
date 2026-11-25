"""Checks for the ratio-greedy threshold search."""

from __future__ import annotations

import random
import sys
import unittest
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from makespan import SolveContext, parse, solve
from makespan.algorithms import public_algorithms
from makespan.model import Edge, Instance, makespan
from makespan.ratio_greedy import RatioGreedyStep


ONE_EDGE = """\
# graph-instances v1 count=1 index-base=0
# instance: one
2 1
0 1 3 5
"""

FRACTIONAL = """\
# graph-instances v1 count=1 index-base=0
# instance: fractional
2 1
0 1 0.5 1.5
"""


def _build(node_count: int, rows: list[tuple[int, int, int, int]]) -> Instance:
    """Build a 0-based instance whose edge ``i + 1`` is ``rows[i]``.

    Args:
        node_count: Number of nodes, numbered ``0 .. node_count - 1``.
        rows: One ``(u, v, w_u, w_v)`` tuple per edge, in the reference order.

    Returns:
        Instance: Graph with those edges numbered from 1.
    """
    edges = tuple(
        Edge(index, u, v, float(w_u), float(w_v))
        for index, (u, v, w_u, w_v) in enumerate(rows, start=1)
    )
    return Instance("ratio", (), tuple(range(node_count)), edges)


def _reference(node_count: int, edges: list[tuple[int, int, int, int]]) -> list[int]:
    """Run the original integer search and return one node per edge.

    This is the design that ``ratio-greedy`` is supposed to reproduce on
    positive integral weights: ratio leaf peeling, a light core, then one
    propagation pass per node of the largest component.

    Args:
        node_count: Number of nodes, numbered ``0 .. node_count - 1``.
        edges: ``(u, v, w_u, w_v)`` in edge-index order.

    Returns:
        list[int]: Node chosen for each edge.

    Raises:
        AssertionError: The reference search did not find an orientation.
    """
    found = _reference_search(node_count, edges)
    if found is None:
        raise AssertionError("reference search found no orientation")
    return found


def _reference_search(node_count: int, edges: list[tuple[int, int, int, int]]):
    """Integer bisection used by the original script.

    Args:
        node_count: Number of nodes.
        edges: ``(u, v, w_u, w_v)`` in edge-index order.

    Returns:
        list[int] | None: Best orientation, or ``None`` when every trial rejected.
    """
    lo = max(1, max(min(edge[2], edge[3]) for edge in edges))
    hi = sum(max(edge[2], edge[3]) for edge in edges)
    best = None
    best_score = None
    while lo <= hi:
        threshold = (lo + hi) // 2
        assignment = _reference_attempt(node_count, edges, threshold)
        if assignment is None:
            lo = threshold + 1
            continue
        score = _reference_makespan(node_count, edges, assignment)
        if best_score is None or score < best_score:
            best_score = score
            best = assignment
        hi = threshold - 1
    return best


def _reference_attempt(node_count: int, edges: list[tuple[int, int, int, int]], threshold: int):
    """One original trial at an integer threshold.

    Args:
        node_count: Number of nodes.
        edges: ``(u, v, w_u, w_v)`` in edge-index order.
        threshold: Candidate makespan.

    Returns:
        list[int] | None: Orientation, or ``None`` when the trial rejects.
    """
    count = len(edges)
    load = [0] * node_count
    assign = [None] * count
    free = set(range(count))

    def neighbors(node: int) -> set[int]:
        """Free neighbors of ``node``.

        Args:
            node: Node whose free neighborhood is requested.

        Returns:
            set[int]: Distinct other endpoints of free edges at ``node``.
        """
        found = set()
        for edge_id in free:
            left, right, _weight_u, _weight_v = edges[edge_id]
            if node == left:
                found.add(right)
            elif node == right:
                found.add(left)
        return found

    def weight(edge_id: int, node: int) -> int:
        """Weight of edge ``edge_id`` at ``node``.

        Args:
            edge_id: Index into ``edges``.
            node: Endpoint that would receive the edge.

        Returns:
            int: Weight added to ``node``.
        """
        left, right, weight_u, weight_v = edges[edge_id]
        return weight_u if node == left else weight_v

    def other(edge_id: int, node: int) -> int:
        """Endpoint of edge ``edge_id`` that is not ``node``.

        Args:
            edge_id: Index into ``edges``.
            node: One endpoint.

        Returns:
            int: The other endpoint.
        """
        left, right, _weight_u, _weight_v = edges[edge_id]
        return right if node == left else left

    changed = True
    while changed:
        changed = False
        for node in range(node_count):
            neighborhood = neighbors(node)
            if len(neighborhood) != 1:
                continue
            parent = next(iter(neighborhood))
            bundle = [edge_id for edge_id in free if node in edges[edge_id][:2]]
            bundle.sort(
                key=lambda edge_id: (
                    Fraction(weight(edge_id, node), weight(edge_id, parent)),
                    -weight(edge_id, node),
                    -weight(edge_id, parent),
                )
            )
            for edge_id in bundle:
                if load[node] + weight(edge_id, node) <= threshold:
                    assign[edge_id] = node
                    load[node] += weight(edge_id, node)
                else:
                    assign[edge_id] = parent
                    load[parent] += weight(edge_id, parent)
                free.discard(edge_id)
            if load[parent] > threshold or load[node] > threshold:
                return None
            changed = True
    for edge_id in free:
        left, right, weight_u, weight_v = edges[edge_id]
        head = left if weight_u <= weight_v else right
        assign[edge_id] = head
        load[head] += weight(edge_id, head)
    parent = list(range(node_count))

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

    for left, right, _weight_u, _weight_v in edges:
        parent[find(left)] = find(right)
    size: dict[int, int] = defaultdict(int)
    for node in range(node_count):
        size[find(node)] += 1
    for turn in range(max(size.values())):
        if all(value <= threshold for value in load):
            return assign
        visited: set[int] = set()
        while True:
            waiting = [
                node
                for node in range(node_count)
                if load[node] > threshold and node not in visited and turn < size[find(node)]
            ]
            if not waiting:
                break
            node = max(waiting, key=lambda item: load[item])
            visited.add(node)
            movable = [edge_id for edge_id in free if assign[edge_id] == node]
            movable.sort(
                key=lambda edge_id: (
                    -Fraction(weight(edge_id, node), weight(edge_id, other(edge_id, node))),
                    -weight(edge_id, node),
                )
            )
            for edge_id in movable:
                if load[node] <= threshold:
                    break
                neighbor = other(edge_id, node)
                if neighbor in visited:
                    continue
                assign[edge_id] = neighbor
                load[node] -= weight(edge_id, node)
                load[neighbor] += weight(edge_id, neighbor)
    if all(value <= threshold for value in load):
        return assign
    return None


def _reference_makespan(node_count: int, edges, assign: list[int]) -> int:
    """Maximum load of a reference orientation.

    Args:
        node_count: Number of nodes.
        edges: ``(u, v, w_u, w_v)`` in edge-index order.
        assign: Node chosen for each edge.

    Returns:
        int: Maximum node load.
    """
    load = [0] * node_count
    for edge_id, node in enumerate(assign):
        left, _right, weight_u, weight_v = edges[edge_id]
        load[node] += weight_u if node == left else weight_v
    return max(load)


class RatioGreedyTests(unittest.TestCase):
    """Checks that ratio-greedy matches its reference search and stays feasible."""

    def test_registered_after_the_other_greeds(self) -> None:
        """``ratio-greedy`` is public and listed with the constructive algorithms."""
        names = public_algorithms()
        self.assertIn("ratio-greedy", names)
        self.assertLess(names.index("greedy-load"), names.index("ratio-greedy"))
        self.assertLess(names.index("ratio-greedy"), names.index("brute"))

    def test_matches_the_reference_search(self) -> None:
        """Reproduce the original integer search on bundles, cycles, and random graphs."""
        families = [
            (2, [(0, 1, 6, 9), (0, 1, 5, 7), (0, 1, 5, 7)]),
            (5, [(0, 1, 19, 15), (1, 2, 12, 13), (2, 3, 11, 13), (3, 4, 17, 6), (4, 0, 17, 14)]),
            (3, [(0, 1, 4, 20), (1, 2, 7, 11), (1, 2, 3, 6)]),
        ]
        rng = random.Random(3)
        for _draw in range(40):
            node_count = rng.randint(3, 5)
            edge_count = rng.randint(node_count - 1, 8)
            rows = [(i - 1, i, rng.randint(1, 20), rng.randint(1, 20)) for i in range(1, node_count)]
            while len(rows) < edge_count:
                left, right = rng.sample(range(node_count), 2)
                rows.append((left, right, rng.randint(1, 20), rng.randint(1, 20)))
            families.append((node_count, rows))
        for node_count, rows in families:
            instance = _build(node_count, rows)
            solution = solve(instance, "ratio-greedy")
            reference = _reference(node_count, rows)
            with self.subTest(edges=rows):
                self.assertEqual(
                    [solution.heads[index] for index in range(1, len(rows) + 1)],
                    reference,
                )
                self.assertEqual(solution.makespan, _reference_makespan(node_count, rows, reference))
                optimum = solve(instance, "brute").makespan
                self.assertGreaterEqual(solution.makespan, optimum)
                self.assertLessEqual(solution.lower_bound, optimum)

    def test_fractional_edge_uses_the_lighter_endpoint(self) -> None:
        """A single fractional edge is sent to the endpoint of weight 0.5."""
        instance = parse(FRACTIONAL)[0]
        solution = solve(instance, "ratio-greedy")
        self.assertEqual(solution.heads[1], 0)
        self.assertAlmostEqual(solution.makespan, 0.5)

    def test_zero_parent_weight_does_not_crash(self) -> None:
        """A zero weight at the parent is a finite trial, not a division by zero."""
        instance = _build(2, [(0, 1, 4, 0), (0, 1, 3, 5)])
        solution = solve(instance, "ratio-greedy")
        self.assertEqual(set(solution.heads), {1, 2})
        for edge in instance.edges:
            self.assertIn(solution.heads[edge.id], edge.endpoints())
        self.assertGreaterEqual(solution.makespan, solve(instance, "brute").makespan)

    def test_locked_edge_is_kept(self) -> None:
        """An edge frozen by an earlier step stays on the head it was given."""
        instance = parse(ONE_EDGE)[0]
        ctx = SolveContext(instance)
        ctx.lock(1, 1)
        RatioGreedyStep().run(ctx)
        solution = ctx.to_solution("ratio-greedy")
        self.assertEqual(solution.heads[1], 1)
        self.assertAlmostEqual(solution.makespan, 5)

    def test_random_core_repeats_for_the_same_seed(self) -> None:
        """The same seed orients a cycle the same way twice."""
        rows = [(0, 1, 4, 5), (1, 2, 6, 1), (2, 3, 3, 8), (3, 0, 7, 2)]
        instance = _build(4, rows)
        first = solve(instance, "ratio-greedy", start="random", seed=4)
        second = solve(instance, "ratio-greedy", start="random", seed=4)
        self.assertEqual(first.heads, second.heads)
        self.assertGreaterEqual(first.makespan, solve(instance, "brute").makespan)
        self.assertIn(first.metadata["ratio_start"], ("random",))

    def test_isolated_node_has_zero_load(self) -> None:
        """A node with no edge stays at load 0."""
        instance = _build(3, [(0, 1, 2, 9)])
        solution = solve(instance, "ratio-greedy")
        self.assertEqual(solution.loads[2], 0)
        self.assertEqual(makespan(instance, solution.heads), solution.makespan)


if __name__ == "__main__":
    unittest.main()
