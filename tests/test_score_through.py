"""Checks for the through-flow threshold search."""

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
from makespan.score_through import (
    ORDERS,
    _Rank,
    _direct_bundles,
    _rank_graph,
    _scores,
    _topo_turns,
    _turn_order,
    attempt,
    search_heads,
    through_names,
)


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
    return Instance("through", (), tuple(range(node_count)), edges)


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


def _settings(name: str) -> tuple[str, str, str]:
    """Read the metric, the pruning choice, and the turn order from a through name.

    Args:
        name: Command-line name such as ``score-ratio-peel-through-flow-asc``.

    Returns:
        tuple[str, str, str]: Metric, ``"peel"`` or ``"raw"``, and the turn order.
    """
    metric, prune, rest = name.split("-", 3)[1:]
    order = "flow-desc" if rest == "through" else rest.removeprefix("through-")
    return metric, prune, order


class ScoreThroughTests(unittest.TestCase):
    """The twenty-four through names share one trial and differ in whose turn comes first."""

    def test_names_are_public(self) -> None:
        """Each through name is registered, and none of them is an early name."""
        names = public_algorithms()
        self.assertEqual(
            [name for name in names if "-through" in name],
            through_names(),
        )
        self.assertEqual(len(through_names()), 24)
        self.assertEqual(
            [name for name in through_names() if name.endswith("-through")],
            [
                "score-ratio-peel-through",
                "score-ratio-raw-through",
                "score-diff-peel-through",
                "score-diff-raw-through",
            ],
        )
        self.assertNotIn("score-ratio-peel-early-through", names)

    def test_every_name_orients_integral_and_fractional_graphs(self) -> None:
        """Each registered name returns a complete orientation and keeps its settings."""
        integral = _build(3, [(0, 1, 1, 4), (1, 2, 2, 3)])
        fractional = _build(2, [(0, 1, 0.5, 1.5)])
        for instance in (integral, fractional):
            for name in through_names():
                metric, prune, order = _settings(name)
                solution = solve(instance, name)
                _assert_complete(self, instance, solution.heads)
                self.assertAlmostEqual(solution.makespan, makespan(instance, solution.heads))
                self.assertAlmostEqual(
                    solution.lower_bound, trivial_lower_bound(instance)
                )
                self.assertEqual(solution.metadata["score_metric"], metric)
                self.assertEqual(solution.metadata["score_prune"], prune == "peel")
                self.assertEqual(solution.metadata["score_mode"], "through")
                self.assertEqual(solution.metadata["score_order"], order)
                self.assertFalse(solution.metadata["score_same_first"])
                self.assertGreater(solution.metadata["score_trials"], 0)

    def test_turn_orders_rank_flow_and_flow_over_room(self) -> None:
        """Largest flow, smallest flow, and both flow-over-room directions are distinct."""
        score = {0: Fraction(1), 1: Fraction(1), 2: Fraction(1)}
        through = {0: 2.0, 1: 6.0, 2: 3.0}
        room = {0: 2.0, 1: 3.0, 2: 6.0}
        rank = _Rank({0: 0, 1: 1, 2: 2}, {})
        self.assertEqual(_turn_order(score, through, room, "flow-desc", rank), [1, 2, 0])
        self.assertEqual(_turn_order(score, through, room, "flow-asc", rank), [0, 2, 1])
        self.assertEqual(
            _turn_order(score, through, room, "flow-room-asc", rank), [2, 0, 1]
        )
        self.assertEqual(
            _turn_order(score, through, room, "flow-room-desc", rank), [1, 0, 2]
        )
        tight = {0: Fraction(0), 1: Fraction(0)}
        tight_rank = _Rank({0: 0, 1: 1}, {})
        self.assertEqual(
            _turn_order(
                tight, {0: 5.0, 1: 1.0}, {0: 0.0, 1: 1.0}, "flow-room-asc", tight_rank
            ),
            [1, 0],
        )
        self.assertEqual(
            _turn_order(
                tight, {0: 5.0, 1: 1.0}, {0: 0.0, 1: 1.0}, "flow-room-desc", tight_rank
            ),
            [0, 1],
        )
        self.assertEqual(
            set(ORDERS),
            {"flow-desc", "flow-asc", "flow-room-asc", "flow-room-desc", "quota", "topo"},
        )

    def test_quota_sends_the_light_edge_and_leaves_the_heavy_edge(self) -> None:
        """Flow that covers only the lighter end points that edge there.

        The heavier edge does not fit in what remains, so the higher-score
        neighbour keeps it.
        """
        instance = _build(2, [(0, 1, 2, 6), (0, 1, 3, 2)])
        found = attempt(instance, 2, metric="diff", prune=False, order="quota")
        self.assertEqual(found, {1: 0, 2: 1})

    def test_topo_emits_a_node_before_the_neighbour_it_points_to(self) -> None:
        """Ready nodes leave before the neighbours their bundles point at."""
        score = {0: Fraction(1), 1: Fraction(5), 2: Fraction(3)}
        bundles = [(1, 0, 1.0, []), (2, 0, 1.0, []), (2, 1, 1.0, [])]
        turns = _topo_turns(score, bundles, _Rank({0: 0, 1: 1, 2: 2}, {}))
        place = {node: index for index, node in enumerate(turns)}
        self.assertLess(place[2], place[1])
        self.assertLess(place[2], place[0])
        self.assertLess(place[1], place[0])
        instance = _build(3, [(0, 1, 2, 5), (1, 2, 4, 1)])
        heads, trials = search_heads(instance, metric="diff", prune=False, order="topo")
        self.assertGreater(trials, 0)
        _assert_complete(self, instance, heads)

    def test_through_orients_a_path(self) -> None:
        """The through-flow order returns a complete orientation and then stops."""
        instance = _build(3, [(0, 1, 2, 5), (1, 2, 4, 1)])
        heads, trials = search_heads(instance, metric="diff", prune=False)
        self.assertGreater(trials, 0)
        _assert_complete(self, instance, heads)

    def test_through_sends_an_equal_score_bundle_from_the_earlier_place(self) -> None:
        """A flat equal-score component still points from the earlier canonical place.

        Both ends are a highest score and a lowest score, so the progress
        fraction does not separate them. The earlier place keeps an edge while
        it fits.
        """
        instance = _build(2, [(0, 1, 3, 3), (0, 1, 3, 3)])
        found = attempt(instance, 4, metric="diff", prune=False)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(sorted(found.values()), [0, 1])
        self.assertEqual(len(set(found.values())), 2)

    def test_equal_score_bundle_points_along_the_progress_fraction(self) -> None:
        """A tied bundle leaves the end closer to its component's high score.

        The later canonical place is the upstream end, so the fraction, rather
        than the place, chooses the tail. A second component uses its own
        highest and lowest scores. A flat component falls back to the place.
        """
        instance = _build(
            11,
            [
                (3, 2, 4, 1),
                (2, 1, 4, 1),
                (1, 0, 4, 1),
                (4, 5, 2, 1),
                (5, 6, 2, 1),
                (6, 7, 2, 1),
                (8, 9, 1, 1),
                (9, 10, 1, 1),
            ],
        )
        free = {edge.id for edge in instance.edges}
        score, _out_load, _in_load = _scores(instance, free, "diff")
        self.assertEqual(score[1], score[2])
        self.assertGreater(score[3], score[2])
        self.assertGreater(score[2], score[0])
        self.assertEqual(score[5], score[6])
        self.assertGreater(score[4], score[5])
        self.assertGreater(score[5], score[7])
        self.assertGreater(score[3], score[4])
        rank = _Rank(
            {0: 7, 1: 0, 2: 1, 3: 6, 4: 5, 5: 4, 6: 2, 7: 3, 8: 8, 9: 9, 10: 3},
            {edge.id: (edge.id,) for edge in instance.edges},
        )
        bundles = {
            (node, neighbour)
            for node, neighbour, _weight, _edges in _direct_bundles(
                instance, free, score, rank
            )
        }
        self.assertIn((2, 1), bundles)
        self.assertIn((3, 2), bundles)
        self.assertIn((1, 0), bundles)
        self.assertIn((5, 6), bundles)
        self.assertIn((4, 5), bundles)
        self.assertIn((6, 7), bundles)
        self.assertIn((10, 9), bundles)

    def test_relabeling_preserves_the_orientation(self) -> None:
        """A new numbering of the same weighted graph produces the same heads.

        Node ids, the order of the node list, and the order of the edges all change.
        Each edge still points at the image of the endpoint it pointed at before.
        """
        rows = [(0, 1, 1, 4), (1, 2, 2, 5), (2, 3, 3, 1), (0, 3, 4, 2)]
        original = _build(4, rows)
        perm = {0: 7, 1: 2, 2: 9, 3: 4}
        inverse = {image: node for node, image in perm.items()}
        shuffled = list(reversed(original.edges))
        renamed = Instance(
            "renamed",
            (),
            tuple(sorted(perm.values(), reverse=True)),
            tuple(
                Edge(index, perm[edge.u], perm[edge.v], edge.w_u, edge.w_v)
                for index, edge in enumerate(shuffled, start=1)
            ),
        )
        original_rank = _rank_graph(original)
        renamed_rank = _rank_graph(renamed)
        for node, image in perm.items():
            self.assertEqual(original_rank.node[node], renamed_rank.node[image])
        for order in ("flow-desc", "flow-asc", "quota", "topo"):
            for prune in (False, True):
                left = attempt(original, 8, metric="ratio", prune=prune, order=order)
                right = attempt(renamed, 8, metric="ratio", prune=prune, order=order)
                self.assertIsNotNone(left)
                self.assertIsNotNone(right)
                assert left is not None and right is not None
                left_heads = {
                    (edge.u, edge.v, edge.w_u, edge.w_v): left[edge.id]
                    for edge in original.edges
                }
                right_heads = {
                    (
                        inverse[edge.u],
                        inverse[edge.v],
                        edge.w_u,
                        edge.w_v,
                    ): inverse[right[edge.id]]
                    for edge in renamed.edges
                }
                self.assertEqual(left_heads, right_heads)

    def test_peel_promotes_a_parent_that_drops_to_one_neighbour(self) -> None:
        """A degree-2 parent is peeled once its leaf is gone, before absorb.

        The chain is 2—8—6, with 6 also on the cycle 0—4—1—6. After 2 is
        peeled, 8 has one neighbour left. That edge must point at 8. The same
        holds for 3—7—0 and for the leaf 5.
        """
        instance = _build(
            9,
            [
                (1, 4, 1, 1),
                (8, 6, 1, 1),
                (7, 0, 1, 1),
                (1, 6, 1, 1),
                (5, 0, 1, 1),
                (6, 0, 1, 1),
                (3, 7, 1, 1),
                (0, 4, 1, 1),
                (2, 8, 1, 1),
            ],
        )
        found = attempt(instance, 2, metric="diff", prune=True)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found[9], 2)
        self.assertEqual(found[2], 8)
        self.assertEqual(found[7], 3)
        self.assertEqual(found[3], 7)
        self.assertEqual(found[5], 5)
        _assert_complete(self, instance, found)
        self.assertLessEqual(makespan(instance, found), 2)
        heads, _trials = search_heads(instance, metric="diff", prune=True)
        self.assertEqual(heads[2], 8)
        self.assertEqual(heads[9], 2)
        self.assertLessEqual(makespan(instance, heads), 2)

    def test_through_rejects_without_a_leftover_pass(self) -> None:
        """An edge that fits on neither end rejects, and nothing repairs it."""
        instance = _build(2, [(0, 1, 3, 5)])
        self.assertIsNone(attempt(instance, 2, metric="ratio", prune=False))


if __name__ == "__main__":
    unittest.main()
