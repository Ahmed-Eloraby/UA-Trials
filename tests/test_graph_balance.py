"""Checks for the LP-free graph balance search (``balance-graph``)."""

from __future__ import annotations

import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from makespan import parse, solve
from makespan.algorithms import public_algorithms
from makespan.graph_balance import certified_push, decide_graph, polish
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


def _random_forest(seed: int) -> Instance:
    """Random tree whose edges are bundles of one to three parallel edges.

    Args:
        seed: RNG seed.

    Returns:
        Instance: Tree of bundles with at most 15 edges.
    """
    rng = random.Random(seed)
    node_count = rng.randint(3, 6)
    rows = []
    for child in range(1, node_count):
        parent = rng.randrange(child)
        for _ in range(rng.randint(1, 3)):
            rows.append((parent, child, rng.randint(1, 6), rng.randint(1, 6)))
    return _build(f"forest-{seed}", node_count, rows[:15])


def _cycle_with_pendants(seed: int) -> Instance:
    """Plain cycle with one or two pendant nodes hanging on bundles.

    Args:
        seed: RNG seed.

    Returns:
        Instance: Cycle plus pendant bundles.
    """
    rng = random.Random(seed)
    length = rng.randint(3, 5)
    rows = [
        (node, (node + 1) % length, rng.randint(1, 6), rng.randint(1, 6))
        for node in range(length)
    ]
    node_count = length
    for _ in range(rng.randint(1, 2)):
        anchor = rng.randrange(length)
        for _ in range(rng.randint(1, 2)):
            rows.append((anchor, node_count, rng.randint(1, 6), rng.randint(1, 6)))
        node_count += 1
    return _build(f"cycle-pendants-{seed}", node_count, rows)


class TrialTests(unittest.TestCase):
    """Checks that one trial is sound and stays within ``2T``."""

    def test_trials_are_sound_on_random_graphs(self) -> None:
        """A reject proves OPT > T, an accept stays within 2T, and OPT <= T is never rejected."""
        for seed in range(8):
            for unit in (True, False):
                instance = _random_graph(4 + seed % 2, 7, seed, unit)
                optimum = _opt(instance)
                top = int(2 * sum(max(edge.w_u, edge.w_v) for edge in instance.edges)) + 1
                for half_steps in range(1, top):
                    threshold = half_steps / 2.0
                    trial = decide_graph(instance, threshold)
                    with self.subTest(instance=instance.name, threshold=threshold):
                        self.assertNotEqual(trial.status, "unknown")
                        if trial.status == "reject":
                            self.assertGreater(optimum, threshold - 1e-9)
                        else:
                            self.assertIsNotNone(trial.heads)
                            self.assertLessEqual(
                                makespan(instance, trial.heads), 2 * threshold + 1e-6
                            )
                        if optimum <= threshold + 1e-9:
                            self.assertNotEqual(trial.status, "reject")


class PushTests(unittest.TestCase):
    """Checks for the price-guided push and its witness."""

    def test_two_machine_bundle_gives_a_witness(self) -> None:
        """Five unit edges on two machines cannot stay within 2 * 1.2."""
        edges = [Edge(index, 0, 1, 1.0, 1.0) for index in range(1, 6)]
        result = certified_push(edges, {0: 0.0, 1: 0.0}, 1.2)
        self.assertEqual(result.status, "certificate")
        self.assertIsNotNone(result.certificate)
        exact_base = {0: Fraction(0), 1: Fraction(0)}
        self.assertTrue(result.certificate.holds(edges, exact_base, 1.2))

    def test_push_balances_within_two_t(self) -> None:
        """The same bundle at T = 2.5 is balanced with every load at most 2T."""
        edges = [Edge(index, 0, 1, 1.0, 1.0) for index in range(1, 6)]
        result = certified_push(edges, {0: 0.0, 1: 0.0}, 2.5)
        self.assertEqual(result.status, "balanced")
        load = {0: 0.0, 1: 0.0}
        for edge in edges:
            head = result.heads[edge.id]
            load[head] += edge.weight_toward(head)
        self.assertLessEqual(max(load.values()), 5.0 + 1e-9)

    def test_prices_move_edges_to_their_expensive_end(self) -> None:
        """Edges cheaper on node 0 are moved to node 1 once node 0 is overloaded."""
        edges = [Edge(index, 0, 1, 1.0, 3.0) for index in range(1, 5)]
        result = certified_push(edges, {0: 0.0, 1: 0.0}, 1.5)
        self.assertEqual(result.status, "balanced")
        load = {0: 0.0, 1: 0.0}
        for edge in edges:
            head = result.heads[edge.id]
            load[head] += edge.weight_toward(head)
        self.assertLessEqual(max(load.values()), 3.0 + 1e-9)
        self.assertGreater(load[1], 0.0)


class SearchTests(unittest.TestCase):
    """Checks for the full ``balance-graph`` pipeline."""

    def test_forests_of_bundles_are_exact(self) -> None:
        """With eps = 0 a tree of bundles is solved to optimality, with a matching bound."""
        for seed in range(12):
            instance = _random_forest(seed)
            optimum = _opt(instance)
            solution = solve(instance, "balance-graph", eps=0.0)
            with self.subTest(instance=instance.name):
                self.assertAlmostEqual(solution.makespan, optimum)
                self.assertAlmostEqual(solution.lower_bound, optimum)

    def test_cycles_with_pendants_are_exact(self) -> None:
        """Pendant bundles are peeled and the plain cycle takes its better rotation."""
        for seed in range(12):
            instance = _cycle_with_pendants(seed)
            optimum = _opt(instance)
            solution = solve(instance, "balance-graph", eps=0.0)
            with self.subTest(instance=instance.name):
                self.assertAlmostEqual(solution.makespan, optimum)
                self.assertAlmostEqual(solution.lower_bound, optimum)

    def test_heavy_edges_prove_a_bound_above_the_lp(self) -> None:
        """K4 with every weight 3: the LP bound is 4.5, the heavy-edge rule proves 6."""
        rows = [(u, v, 3, 3) for u in range(4) for v in range(u + 1, 4)]
        instance = _build("k4-weight-3", 4, rows)
        solution = solve(instance, "balance-graph", eps=0.0)
        self.assertEqual(solution.lower_bound, 6)
        self.assertEqual(solution.makespan, 6)
        self.assertIn("heavy-edges-overfull", solution.metadata["graph_reject_reasons"])

    def test_random_graphs_stay_within_twice_the_lower_bound(self) -> None:
        """Feasible, bounded by brute force, and within 2(1 + eps) of the lower fence."""
        for seed in range(6):
            for unit in (True, False):
                instance = _random_graph(5, 8, seed, unit)
                optimum = _opt(instance)
                solution = solve(instance, "balance-graph")
                with self.subTest(instance=instance.name):
                    self._assert_bracketed(instance, solution, optimum)

    def test_instances_file(self) -> None:
        """Every instance in instances.txt is bracketed and certified."""
        for instance in parse((ROOT / "instances.txt").read_text(encoding="utf-8")):
            optimum = _opt(instance)
            solution = solve(instance, "balance-graph")
            with self.subTest(instance=instance.name):
                self._assert_bracketed(instance, solution, optimum)

    def test_polish_never_raises_the_makespan(self) -> None:
        """Polishing a random orientation never increases its makespan."""
        for seed in range(10):
            instance = _random_graph(5, 9, seed, unit=seed % 2 == 0)
            rng = random.Random(seed)
            heads = {edge.id: rng.choice(edge.endpoints()) for edge in instance.edges}
            polished = polish(instance, heads)
            with self.subTest(instance=instance.name):
                self.assertEqual(set(polished), set(heads))
                self.assertLessEqual(makespan(instance, polished), makespan(instance, heads) + 1e-9)

    def test_registered_on_the_command_line(self) -> None:
        """``balance-graph`` is a public algorithm name."""
        self.assertIn("balance-graph", public_algorithms())

    def _assert_bracketed(self, instance: Instance, solution, optimum: float) -> None:
        """Check feasibility, the bounds, and the 2(1 + eps) guarantee.

        Args:
            instance: Graph that was solved.
            solution: Result of ``balance-graph`` with the default eps of 0.25.
            optimum: Brute-force optimum.
        """
        self.assertEqual(set(solution.heads), {edge.id for edge in instance.edges})
        for edge in instance.edges:
            self.assertIn(solution.heads[edge.id], edge.endpoints())
        self.assertTrue(solution.metadata["graph_certified"])
        self.assertLessEqual(solution.lower_bound, optimum + 1e-6)
        self.assertGreaterEqual(solution.makespan, optimum - 1e-6)
        self.assertLessEqual(solution.makespan, 2 * 1.25 * solution.lower_bound + 1e-6)


if __name__ == "__main__":
    unittest.main()
