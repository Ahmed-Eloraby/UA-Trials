"""Checks for the score-family threshold search."""

from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from makespan import solve
from makespan.algorithms import public_algorithms
from makespan.model import Edge, Instance, loads, makespan, trivial_lower_bound
from makespan.score_family import (
    ScoreStep,
    _assign_inflow,
    _measure_inflow,
    _scores,
    _topo_order,
    attempt,
    family_names,
    search_heads,
)
from makespan.score_through import through_names


def _build(node_count: int, rows: list[tuple[int, int, float, float]]) -> Instance:
    """Build a 0-based instance whose edge ``i + 1`` is ``rows[i]``.

    Args:
        node_count: Number of nodes, numbered ``0 .. node_count - 1``.
        rows: One ``(u, v, w_u, w_v)`` tuple per edge, in edge-id order.

    Returns:
        Instance: Graph with those edges numbered from 1.
    """
    edges = tuple(
        Edge(index, u, v, float(w_u), float(w_v))
        for index, (u, v, w_u, w_v) in enumerate(rows, start=1)
    )
    return Instance("score", (), tuple(range(node_count)), edges)


def _assert_complete(test: unittest.TestCase, instance: Instance, heads: dict[int, int]) -> None:
    """Every edge points at one of its endpoints, and the makespan matches the loads.

    Args:
        test: Test case that records the assertions.
        instance: Graph that was oriented.
        heads: Map from each edge id to its head.
    """
    test.assertEqual(set(heads), {edge.id for edge in instance.edges})
    for edge in instance.edges:
        test.assertIn(heads[edge.id], edge.endpoints())
    node_loads = loads(instance, heads)
    test.assertAlmostEqual(max(node_loads.values()), makespan(instance, heads))


class ScoreFamilyTests(unittest.TestCase):
    """The forty-eight level names share one trial and stay inside the threshold they accept."""

    def test_names_are_public_and_ordered_after_ratio_greedy(self) -> None:
        """Level names follow ratio-greedy. Each through name follows its group."""
        names = public_algorithms()
        scores = [name for name in names if name.startswith("score-")]
        level = [name for name in scores if "-through" not in name]
        self.assertEqual(level, family_names())
        self.assertEqual(len(family_names()), 48)
        self.assertEqual(
            [name for name in scores if "-through" in name],
            through_names(),
        )
        self.assertNotIn("score-ratio-peel-early-through", names)
        for metric in ("ratio", "diff"):
            for prune in ("peel", "raw"):
                previous = f"score-{metric}-{prune}-early-inflow"
                block = [
                    name
                    for name in through_names()
                    if name.startswith(f"score-{metric}-{prune}-through")
                ]
                start = names.index(previous) + 1
                self.assertEqual(names[start : start + len(block)], block)
        self.assertLess(names.index("ratio-greedy"), names.index(scores[0]))
        self.assertLess(names.index(scores[-1]), names.index("brute"))

    def test_every_name_orients_integral_and_fractional_graphs(self) -> None:
        """Each registered name returns a complete orientation and keeps its settings."""
        integral = _build(3, [(0, 1, 1, 4), (1, 2, 2, 3)])
        fractional = _build(2, [(0, 1, 0.5, 1.5)])
        for instance in (integral, fractional):
            for name in family_names():
                parts = name.split("-")
                metric, prune = parts[1], parts[2]
                if parts[3] == "early":
                    same_first = True
                    mode = parts[4]
                else:
                    same_first = False
                    mode = parts[3]
                solution = solve(instance, name)
                _assert_complete(self, instance, solution.heads)
                self.assertAlmostEqual(solution.makespan, makespan(instance, solution.heads))
                self.assertAlmostEqual(
                    solution.lower_bound, trivial_lower_bound(instance)
                )
                self.assertEqual(solution.metadata["score_metric"], metric)
                self.assertEqual(solution.metadata["score_prune"], prune == "peel")
                self.assertEqual(solution.metadata["score_mode"], mode)
                self.assertEqual(solution.metadata["score_same_first"], same_first)
                self.assertGreater(solution.metadata["score_trials"], 0)

    def test_attempt_rejects_a_threshold_below_every_endpoint(self) -> None:
        """An edge that fits on neither end rejects the trial."""
        instance = _build(2, [(0, 1, 3, 5)])
        self.assertIsNone(
            attempt(instance, 2, metric="ratio", prune=False, mode="sink")
        )
        self.assertIsNone(
            attempt(instance, 2, metric="diff", prune=True, mode="topo")
        )

    def test_full_star_is_absorbed(self) -> None:
        """A node that can take every free incident edge receives all of them."""
        instance = _build(3, [(0, 1, 1, 10), (0, 2, 1, 10)])
        found = attempt(instance, 2, metric="ratio", prune=False, mode="flow")
        self.assertEqual(found, {1: 0, 2: 0})

    def test_peel_and_raw_differ_on_a_pendant(self) -> None:
        """Leaf pruning keeps the cheap pendant edge; scoring sends that edge inward."""
        instance = _build(3, [(0, 1, 3, 4), (0, 1, 3, 4), (1, 2, 1, 4)])
        for metric in ("ratio", "diff"):
            peeled = attempt(instance, 5, metric=metric, prune=True, mode="sink")
            raw = attempt(instance, 5, metric=metric, prune=False, mode="sink")
            self.assertEqual(peeled, {1: 0, 2: 1, 3: 2})
            self.assertEqual(raw, {1: 1, 2: 0, 3: 2})

    def test_sink_and_source_are_feasible(self) -> None:
        """Both seed rules return a complete orientation on a short path."""
        instance = _build(3, [(0, 1, 2, 5), (1, 2, 4, 1)])
        for mode in ("sink", "source"):
            heads, trials = search_heads(
                instance, metric="ratio", prune=True, mode=mode
            )
            self.assertGreater(trials, 0)
            _assert_complete(self, instance, heads)

    def test_topo_emits_a_higher_score_before_a_lower_score(self) -> None:
        """Ready nodes are conceded from the larger metric toward the smaller one."""
        keys = {
            0: Fraction(5),
            1: Fraction(1),
            2: Fraction(-3),
        }
        self.assertEqual(_topo_order(keys, [(0, 1), (0, 2), (1, 2)]), [0, 1, 2])
        tied = {0: Fraction(4), 1: Fraction(4)}
        self.assertEqual(_topo_order(tied, []), [0, 1])

    def test_same_level_edges_can_run_before_the_cuts(self) -> None:
        """Orienting the level-1 edge first leaves room the later cut can use."""
        instance = _build(
            4,
            [
                (0, 1, 2, 4),
                (0, 2, 2, 4),
                (1, 2, 4, 4),
                (0, 3, 2, 4),
                (0, 3, 2, 4),
            ],
        )
        for metric in ("ratio", "diff"):
            early = attempt(
                instance, 5, metric=metric, prune=False, mode="sink", same_first=True
            )
            late = attempt(
                instance, 5, metric=metric, prune=False, mode="sink", same_first=False
            )
            self.assertEqual(early, {1: 1, 2: 0, 3: 2, 4: 3, 5: 0})
            self.assertIsNone(late)
        heads, trials = search_heads(
            instance, metric="ratio", prune=False, mode="sink", same_first=True
        )
        self.assertGreater(trials, 0)
        _assert_complete(self, instance, heads)

    def test_flow_returns_a_complete_orientation(self) -> None:
        """The max-flow order still orients every edge."""
        instance = _build(3, [(0, 1, 2, 5), (1, 2, 4, 1)])
        heads, trials = search_heads(
            instance, metric="diff", prune=False, mode="flow"
        )
        self.assertGreater(trials, 0)
        _assert_complete(self, instance, heads)

    def test_equal_scores_open_both_channel_directions(self) -> None:
        """An equal-score bundle is an arc each way, and a closed pair carries no flow."""
        instance = _build(2, [(0, 1, 4, 4), (0, 1, 2, 2)])
        free = {1, 2}
        keys, out_load = _scores(instance, free, "diff")
        inflow, channel, caps = _measure_inflow(
            instance, 10, {0: 0.0, 1: 0.0}, free, keys, out_load
        )
        self.assertEqual(caps[(0, 1)], 6)
        self.assertEqual(caps[(1, 0)], 6)
        self.assertEqual(channel[(0, 1)], 0)
        self.assertEqual(channel[(1, 0)], 0)
        self.assertEqual(inflow[0], 0)
        self.assertEqual(inflow[1], 0)

    def test_inflow_keeps_the_light_edge_inside_the_channel_allowance(self) -> None:
        """Share times capacity keeps the light edge and sends the overflow across."""
        instance = _build(2, [(0, 1, 10, 1), (0, 1, 1, 4)])
        load = {0: 0.0, 1: 0.0}
        assign: dict[int, int] = {}
        free = {1, 2}
        keys = {0: Fraction(0), 1: Fraction(0)}
        kept = _assign_inflow(
            instance,
            20,
            load,
            assign,
            free,
            "diff",
            keys,
            {0: 0.0, 1: 4.0},
            {(0, 1): 2.0},
            {(0, 1): 5.0},
        )
        self.assertTrue(kept)
        self.assertEqual(assign, {1: 1, 2: 0})
        self.assertEqual(free, set())

    def test_through_is_not_a_score_family_mode(self) -> None:
        """The through search is a separate algorithm, so this step rejects that mode."""
        with self.assertRaises(ValueError):
            ScoreStep(mode="through")
        instance = _build(2, [(0, 1, 3, 5)])
        with self.assertRaises(ValueError):
            attempt(instance, 2, metric="ratio", prune=False, mode="through")


if __name__ == "__main__":
    unittest.main()
