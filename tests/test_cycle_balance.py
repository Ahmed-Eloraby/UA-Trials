"""Checks for the LP-free cycle and bridge balance searches."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from makespan import solve
from makespan.algorithms import public_algorithms
from makespan.cycle_balance import decide_cycle
from makespan.model import Edge, Instance, makespan


def _build(name: str, node_count: int, rows: list[tuple[int, int, float, float]]) -> Instance:
    """Build an instance on ``0 .. node_count - 1`` from ``(u, v, w_u, w_v)`` rows.

    Args:
        name: Instance label.
        node_count: Number of nodes.
        rows: One ``(u, v, w_u, w_v)`` tuple per edge.

    Returns:
        Instance: Graph with edges numbered from 1.
    """
    edges = tuple(
        Edge(index, u, v, float(w_u), float(w_v))
        for index, (u, v, w_u, w_v) in enumerate(rows, start=1)
    )
    return Instance(name, (), tuple(range(node_count)), edges)


def _opt(instance: Instance) -> float:
    """Brute-force optimum makespan.

    Args:
        instance: Graph whose optimum is computed.

    Returns:
        float: Optimum makespan.
    """
    return solve(instance, "brute").makespan


def _random_graph(node_count: int, edge_count: int, seed: int, unit: bool) -> Instance:
    """Random multigraph with unit weights or weights in ``1..6``.

    Args:
        node_count: Number of nodes.
        edge_count: Number of edges.
        seed: RNG seed.
        unit: When true every weight is 1.

    Returns:
        Instance: Random graph.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(edge_count):
        u, v = rng.sample(range(node_count), 2)
        if unit:
            rows.append((u, v, 1, 1))
        else:
            rows.append((u, v, rng.randint(1, 6), rng.randint(1, 6)))
    return _build(f"{'unit' if unit else 'mixed'}-{seed}", node_count, rows)


def _triangle() -> Instance:
    """Triangle whose cheaper reverse is the unique best first flip at ``T = 10``.

    Edge 1 joins nodes 0 and 1 with weights 9 and 6. Listed from node 0 it is the heaviest
    top in that direction, and reversing it onto node 1 costs 6. The other
    direction's heaviest reverse costs 7.

    Returns:
        Instance: Three-edge triangle.
    """
    return _build(
        "triangle-flip",
        3,
        [
            (0, 1, 9, 6),
            (1, 2, 6, 4),
            (2, 0, 7, 8),
        ],
    )


def _budget_bridge() -> Instance:
    """Two triangles joined by a bridge that the component budget sends sideways.

    Side ``{0, 1, 2}`` is filled by three edges of weight 10, so its spare
    budget is 0 even though node 0 still has room for the bridge weight 1.
    The locally cheaper head is node 0. The only split that respects the
    budgets orients the bridge toward node 3.

    Returns:
        Instance: Bridge between a full component and a component with room.
    """
    rows = [
        (0, 1, 10, 10),
        (1, 2, 10, 10),
        (2, 0, 10, 10),
        (3, 4, 6, 6),
        (4, 5, 6, 6),
        (5, 3, 6, 6),
        (0, 3, 1, 4),
    ]
    return _build("budget-bridge", 6, rows)


def _linked_triangles(bridge_rows: list[tuple[float, float]]) -> Instance:
    """Two triangles joined by one parallel edge for each pair in ``bridge_rows``.

    Args:
        bridge_rows: ``(weight toward 0, weight toward 3)`` for each bridge edge.

    Returns:
        Instance: Nodes ``0..5`` with the bridge channel between 0 and 3.
    """
    rows = [
        (0, 1, 6, 6),
        (1, 2, 6, 6),
        (2, 0, 6, 6),
        (3, 4, 6, 6),
        (4, 5, 6, 6),
        (5, 3, 6, 6),
    ]
    rows.extend((0, 3, toward_0, toward_3) for toward_0, toward_3 in bridge_rows)
    return _build("long-bridge", 6, rows)


class TrialTests(unittest.TestCase):
    """Checks for one threshold trial."""

    def test_triangle_takes_the_cheaper_reverse(self) -> None:
        """The first flip orients edge 1 toward node 1, then the path is settled."""
        instance = _triangle()
        for use_bridges in (False, True):
            trial = decide_cycle(instance, 10.0, use_bridges)
            with self.subTest(use_bridges=use_bridges):
                self.assertEqual(trial.status, "accept")
                self.assertIsNotNone(trial.heads)
                self.assertEqual(trial.heads[1], 1)
                self.assertEqual(trial.heads[2], 2)
                self.assertEqual(trial.heads[3], 0)

    def test_bridge_budget_overrides_the_cheaper_endpoint(self) -> None:
        """The bridge goes to node 3 even though node 0 has room and a smaller weight."""
        instance = _budget_bridge()
        trial = decide_cycle(instance, 10.0, use_bridges=True)
        self.assertEqual(trial.status, "accept")
        self.assertIsNotNone(trial.heads)
        self.assertEqual(trial.heads[7], 3)

    def test_feasible_bridge_is_not_rejected(self) -> None:
        """A feasible threshold is not proved infeasible, and the fence stays below OPT."""
        instance = _budget_bridge()
        optimum = _opt(instance)
        self.assertLessEqual(optimum, 10.0 + 1e-6)
        for use_bridges in (False, True):
            trial = decide_cycle(instance, optimum, use_bridges)
            with self.subTest(use_bridges=use_bridges):
                self.assertNotEqual(trial.status, "reject")
        for name in ("balance-cycle", "balance-bridge"):
            solution = solve(instance, name)
            with self.subTest(algorithm=name):
                self.assertLessEqual(solution.lower_bound, optimum + 1e-6)
                self.assertGreaterEqual(solution.makespan, optimum - 1e-6)

    def test_edge_that_fits_nowhere_rejects(self) -> None:
        """An edge heavier than ``T`` at both ends rejects while the proof is intact."""
        instance = _build("too-heavy", 2, [(0, 1, 5, 4)])
        for use_bridges in (False, True):
            trial = decide_cycle(instance, 3.0, use_bridges)
            with self.subTest(use_bridges=use_bridges):
                self.assertEqual(trial.status, "reject")
                self.assertIsNone(trial.heads)
                self.assertTrue(trial.certified)

    def test_long_bridge_uses_the_ratio_prefix(self) -> None:
        """A channel past the enumeration limit keeps the prefix that fits on the leaf side."""
        instance = _linked_triangles([(0.4, 0.4)] * 21)
        trial = decide_cycle(instance, 10.0, use_bridges=True)
        self.assertEqual(trial.status, "accept")
        self.assertIsNotNone(trial.heads)
        self.assertFalse(trial.certified)
        bridge_ids = [
            edge.id for edge in instance.edges if {edge.u, edge.v} == {0, 3}
        ]
        self.assertEqual(len(bridge_ids), 21)
        for edge_id in bridge_ids:
            self.assertEqual(trial.heads[edge_id], 0)

    def test_long_bridge_that_does_not_fit_stays_uncertified(self) -> None:
        """A ratio split that overflows the far side finishes without rejecting ``T``."""
        instance = _linked_triangles([(1.0, 1.0)] * 21)
        trial = decide_cycle(instance, 10.0, use_bridges=True)
        self.assertEqual(trial.status, "accept")
        self.assertIsNotNone(trial.heads)
        self.assertFalse(trial.certified)
        self.assertEqual(set(trial.heads), {edge.id for edge in instance.edges})


class SearchTests(unittest.TestCase):
    """Checks for the full pipelines on small graphs."""

    def test_random_graphs_are_complete_and_do_not_pass_the_optimum(self) -> None:
        """Every edge is oriented, the makespan is at least OPT, and the fence is not above it."""
        for seed in range(6):
            for unit in (True, False):
                instance = _random_graph(5, 8, seed, unit)
                optimum = _opt(instance)
                for name in ("balance-cycle", "balance-bridge"):
                    solution = solve(instance, name)
                    with self.subTest(instance=instance.name, algorithm=name):
                        self._assert_feasible(instance, solution, optimum)

    def test_registered_on_the_command_line(self) -> None:
        """Both searches are public algorithm names, in catalog order."""
        names = public_algorithms()
        self.assertIn("balance-cycle", names)
        self.assertIn("balance-bridge", names)
        self.assertLess(names.index("balance-lp"), names.index("balance-cycle"))
        self.assertLess(names.index("balance-cycle"), names.index("balance-bridge"))

    def _assert_feasible(self, instance: Instance, solution, optimum: float) -> None:
        """Check the orientation, the makespan, and the proven fence.

        Args:
            instance: Graph that was solved.
            solution: Result of one cycle-balance algorithm.
            optimum: Brute-force optimum.
        """
        self.assertEqual(set(solution.heads), {edge.id for edge in instance.edges})
        for edge in instance.edges:
            self.assertIn(solution.heads[edge.id], edge.endpoints())
        self.assertAlmostEqual(solution.makespan, makespan(instance, solution.heads))
        self.assertGreaterEqual(solution.makespan, optimum - 1e-6)
        self.assertLessEqual(solution.lower_bound, optimum + 1e-6)


if __name__ == "__main__":
    unittest.main()
