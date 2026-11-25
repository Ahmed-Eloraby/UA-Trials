"""Checks for the minimum-makespan orientation package."""

from __future__ import annotations

import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from makespan import SolveContext, parse, solve, solver_available
from makespan.brute import BruteForceStep
from makespan.factory import AlgorithmFactory, AlgorithmLimitError
from makespan.flow import saturating_orientation
from makespan.greedy import GreedyLightestStep
from makespan.io import build_instance, format_assignment_file
from makespan.model import Edge, Instance

ONE_EDGE = """\
# graph-instances v1 count=1 index-base=0
# instance: one
# tags: demo
2 1
0 1 3 5
"""

PARALLEL = """\
# graph-instances v1 count=1 index-base=0
# instance: parallel
# tags: parallel-edges
2 2
0 1 4 1
0 1 4 1
"""

CYCLE = """\
# graph-instances v1 count=1 index-base=0
# instance: cycle
# tags: cycle, unit
4 4
0 1 1 1
1 2 1 1
2 3 1 1
3 0 1 1
"""

STAR = """\
# graph-instances v1 count=1 index-base=0
# instance: star
# tags: star
4 3
0 1 1 1
0 2 1 1
0 3 1 1
"""

MIXED = """\
# graph-instances v1 count=1 index-base=0
# instance: mixed
# tags: parallel-edges
2 3
0 1 5 1
0 1 5 1
0 1 5 4
"""

ISOLATED = """\
# graph-instances v1 count=1 index-base=1
# instance: isolated
# tags:
3 1
1 2 2 2
"""

MULTI = """\
# graph-instances v1 count=2 index-base=0
# instance: one
# tags: a, b
2 1
0 1 3 5

# instance: hole
# tags: isolated
4 1
1 2 4 7
"""


def _opt(instance: Instance) -> float:
    """Return the brute-force optimum makespan of ``instance``.

    Args:
        instance: Graph whose exact makespan is computed.

    Returns:
        float: Brute-force optimum makespan.
    """
    return solve(instance, "brute").makespan


class ParserTests(unittest.TestCase):
    """Checks that builder exports keep node ids, tags, and edge weights."""

    def test_text_preserves_zero_based_ids_and_metadata(self) -> None:
        """Parse a 0-based text instance and keep its name, tags, and weights."""
        instance = parse(ONE_EDGE)[0]
        self.assertEqual(instance.name, "one")
        self.assertEqual(instance.tags, ("demo",))
        self.assertEqual(instance.nodes, (0, 1))
        self.assertEqual(instance.edges[0].id, 1)
        self.assertEqual((instance.edges[0].u, instance.edges[0].v), (0, 1))
        self.assertEqual((instance.edges[0].w_u, instance.edges[0].w_v), (3.0, 5.0))

    def test_multi_instance_text_keeps_an_isolated_node(self) -> None:
        """Parse two text blocks and keep an unused node in the second instance."""
        instances = parse(MULTI)
        self.assertEqual([item.name for item in instances], ["one", "hole"])
        self.assertEqual(instances[0].tags, ("a", "b"))
        self.assertEqual(instances[1].nodes, (0, 1, 2, 3))
        self.assertEqual(instances[1].edges[0].endpoints(), (1, 2))

    def test_one_based_header(self) -> None:
        """Honor ``index-base=1`` when numbering nodes."""
        instance = parse(ISOLATED)[0]
        self.assertEqual(instance.nodes, (1, 2, 3))
        self.assertEqual(instance.edges[0].endpoints(), (1, 2))

    def test_plain_block_detects_base(self) -> None:
        """Infer 0-based or 1-based numbering from endpoint ids."""
        zero_based = parse("3 1\n0 2 1 4\n")[0]
        one_based = parse("3 1\n1 2 1 4\n")[0]
        self.assertEqual(zero_based.nodes, (0, 1, 2))
        self.assertEqual(one_based.nodes, (1, 2, 3))

    def test_csv_and_json_match_the_text_instance(self) -> None:
        """Parse CSV and JSON backups into the same node and weight data."""
        csv_text = "instance,tags,n,m,u,v,wu,wv\n" "one,demo,2,1,0,1,3,5\n"
        from_csv = parse(csv_text)[0]
        self.assertEqual(from_csv.name, "one")
        self.assertEqual(from_csv.tags, ("demo",))
        self.assertEqual(from_csv.nodes, (0, 1))
        self.assertEqual(from_csv.edges[0].w_u, 3.0)

        payload = {
            "format": "graph-instance-builder",
            "version": 1,
            "settings": {"base": 0},
            "instances": [
                {
                    "name": "j",
                    "tags": ["demo"],
                    "nodes": [
                        {"id": "a", "x": 0.1, "y": 0.2},
                        {"id": "b", "x": 0.3, "y": 0.4},
                        {"id": "c", "x": 0.5, "y": 0.6},
                    ],
                    "edges": [{"id": "e", "a": "a", "b": "b", "wa": 2, "wb": 9}],
                }
            ],
        }
        from_json = parse(json.dumps(payload))[0]
        self.assertEqual(from_json.nodes, (0, 1, 2))
        self.assertEqual(from_json.edges[0].endpoints(), (0, 1))
        self.assertEqual((from_json.edges[0].w_u, from_json.edges[0].w_v), (2.0, 9.0))

    def test_rejects_bad_edges(self) -> None:
        """Reject negative weights, loops, and malformed JSON backups."""
        with self.assertRaises(ValueError):
            parse("2 1\n0 1 -1 2\n")
        with self.assertRaises(ValueError):
            parse("2 1\n0 0 1 1\n")
        with self.assertRaises(ValueError):
            parse('{"format": "nope"}')

    def test_assignment_file_records_direction_tags_and_loads(self) -> None:
        """Save each edge's head and weight, plus every node's load."""
        isolated = parse(ISOLATED)[0]
        solved = solve(isolated, "greedy-light")
        text = format_assignment_file([(isolated, solved)])
        self.assertTrue(text.startswith("# graph-assignments v1 count=1 index-base=1\n"))
        self.assertIn("# instance: isolated", text)
        self.assertIn("# tags:", text)
        self.assertIn("# algorithm: greedy-light", text)
        self.assertIn("1 2 1 2", text)
        self.assertIn("# loads", text)
        self.assertIn("1 2", text)
        self.assertIn("3 0", text)

        instances = parse(MULTI)
        pair = [(item, solve(item, "greedy-light")) for item in instances]
        combined = format_assignment_file(pair)
        self.assertIn("# graph-assignments v1 count=2 index-base=0", combined)
        self.assertIn("# instance: one", combined)
        self.assertIn("# tags: a, b", combined)
        self.assertIn("0 1 0 3", combined)
        self.assertIn("0 3", combined)
        self.assertIn("1 0", combined)
        self.assertIn("# instance: hole", combined)
        self.assertIn("# tags: isolated", combined)
        self.assertIn("1 2 1 4", combined)
        self.assertIn("3 0", combined)
        with self.assertRaises(ValueError):
            build_instance("loop", (), 1, 0, [(0, 0, 1.0, 1.0)])


class AlgorithmTests(unittest.TestCase):
    """Checks that registered algorithms stay feasible and respect known optima."""

    def test_hand_checked_optima(self) -> None:
        """Match brute force to hand-checked makespans on the fixture graphs."""
        expected = {
            ONE_EDGE: 3,
            PARALLEL: 2,
            CYCLE: 1,
            STAR: 1,
            MIXED: 5,
            ISOLATED: 2,
        }
        for text, optimum in expected.items():
            instance = parse(text)[0]
            with self.subTest(instance=instance.name):
                self.assertEqual(_opt(instance), optimum)
                self._assert_feasible(instance, solve(instance, "brute"))

    def test_greeds_are_feasible_upper_bounds(self) -> None:
        """Check that both greedy algorithms stay at least as large as the optimum."""
        for text in (ONE_EDGE, PARALLEL, CYCLE, STAR, MIXED, ISOLATED):
            instance = parse(text)[0]
            optimum = _opt(instance)
            for name in ("greedy-light", "greedy-load"):
                with self.subTest(instance=instance.name, algorithm=name):
                    solution = solve(instance, name)
                    self._assert_feasible(instance, solution)
                    self.assertGreaterEqual(solution.makespan, optimum - 1e-9)
                    self.assertLessEqual(solution.lower_bound, optimum + 1e-9)

    def test_load_greedy_beats_lightest_on_mixed(self) -> None:
        """Check that load-aware greedy improves on lightest-endpoint for MIXED."""
        instance = parse(MIXED)[0]
        light = solve(instance, "greedy-light")
        load = solve(instance, "greedy-load")
        self.assertEqual(light.makespan, 6)
        self.assertEqual(load.makespan, 5)

    def test_unit_flow_matches_brute_force(self) -> None:
        """Check that unit-weight flow search recovers the brute-force optimum."""
        cycle = parse(CYCLE)[0]
        saturated, heads = saturating_orientation(cycle, 1)
        self.assertTrue(saturated)
        self.assertIsNotNone(heads)
        rejected, _none = saturating_orientation(cycle, 0)
        self.assertFalse(rejected)
        for text in (CYCLE, STAR, PARALLEL):
            instance = parse(text)[0]
            optimum = _opt(instance)
            solution = solve(instance, "flow-search")
            with self.subTest(instance=instance.name):
                self._assert_feasible(instance, solution)
                self.assertAlmostEqual(solution.lower_bound, optimum)
                self.assertAlmostEqual(solution.makespan, optimum)

    def test_mixed_flow_brackets_the_optimum(self) -> None:
        """Check that mixed-weight flow search brackets the optimum and greedy."""
        instance = parse(MIXED)[0]
        optimum = _opt(instance)
        solution = solve(instance, "flow-search")
        best_greedy = min(
            solve(instance, "greedy-light").makespan,
            solve(instance, "greedy-load").makespan,
        )
        self._assert_feasible(instance, solution)
        self.assertLessEqual(solution.lower_bound, optimum + 1e-9)
        self.assertGreaterEqual(solution.makespan, optimum - 1e-9)
        self.assertLessEqual(solution.makespan, best_greedy + 1e-9)

    def test_flow_and_greeds_on_random_graphs(self) -> None:
        """Check flow search against brute force and greedy on random graphs."""
        for seed in range(6):
            unit = _random_instance(5, 8, seed, unit=True)
            mixed = _random_instance(4, 7, seed, unit=False)
            for instance in (unit, mixed):
                optimum = _opt(instance)
                flow = solve(instance, "flow-search")
                light = solve(instance, "greedy-light").makespan
                load = solve(instance, "greedy-load").makespan
                with self.subTest(seed=seed, name=instance.name):
                    self._assert_feasible(instance, flow)
                    self.assertLessEqual(flow.lower_bound, optimum + 1e-7)
                    self.assertGreaterEqual(flow.makespan, optimum - 1e-7)
                    self.assertLessEqual(flow.makespan, min(light, load) + 1e-7)
                    if instance.weights_are_integral() and _all_unit(instance):
                        self.assertAlmostEqual(flow.makespan, optimum)
                        self.assertAlmostEqual(flow.lower_bound, optimum)

    def test_locked_edge_limits_brute_force(self) -> None:
        """Keep a locked head unchanged when brute force orients the rest."""
        instance = parse(PARALLEL)[0]
        ctx = SolveContext(instance)
        ctx.lock(1, 0)
        BruteForceStep().run(ctx)
        self.assertIsNotNone(ctx.best_so_far_assignment)
        self.assertEqual(ctx.best_so_far_assignment.edge_node_map[1], 0)
        self.assertEqual(ctx.best_so_far_makespan, 4)

    def test_brute_force_refuses_too_many_edges(self) -> None:
        """Raise when brute force is asked to enumerate too many free edges."""
        instance = parse(PARALLEL)[0]
        with self.assertRaises(AlgorithmLimitError):
            solve(instance, "brute")

    def test_custom_pipeline_reuses_a_step(self) -> None:
        """Build a one-off pipeline from an existing greedy step."""
        instance = parse(ONE_EDGE)[0]
        solution = AlgorithmFactory.create(
            "custom",
            steps=[GreedyLightestStep()],
            name="only-light",
        ).solve(instance)
        self.assertEqual(solution.algorithm, "only-light")
        self.assertEqual(solution.makespan, 3)
        self.assertEqual(solution.heads[1], 0)

    def test_compare_against_brute_force(self) -> None:
        """Measure greedy algorithms against a brute-force reference."""
        from makespan.__main__ import main
        from makespan.compare import compare_instance

        instance = parse(STAR)[0]
        report = compare_instance(
            instance,
            ["greedy-light", "greedy-load"],
            "brute",
        )
        by_name = {row.name: row for row in report.rows}
        self.assertEqual(report.baseline_makespan, 1)
        self.assertEqual(by_name["greedy-light"].makespan, 3)
        self.assertEqual(by_name["greedy-load"].makespan, 1)
        self.assertEqual(by_name["brute"].makespan, 1)
        self.assertFalse(report.reference_failed)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "star.txt"
            path.write_text(STAR, encoding="utf-8")
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                status = main(
                    [str(path), "-C", "greedy-light", "greedy-load", "-R", "brute"]
                )
            self.assertEqual(status, 0)
            rendered = out.getvalue()
            self.assertIn("# reference: brute", rendered)
            self.assertIn("# optimum: 1", rendered)
            self.assertIn("greedy-light", rendered)
            self.assertIn("reference", rendered)
            self.assertEqual(err.getvalue(), "")

            json_out = io.StringIO()
            with redirect_stdout(json_out):
                status = main(
                    [
                        str(path),
                        "--compare",
                        "greedy-light",
                        "flow-search",
                        "--reference",
                        "brute",
                        "--format",
                        "json",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(json_out.getvalue())
            self.assertEqual(payload[0]["baseline_makespan"], 1)
            gaps = {row["algorithm"]: row["gap"] for row in payload[0]["rows"]}
            self.assertEqual(gaps["greedy-light"], 2)
            self.assertEqual(gaps["brute"], 0)

    def test_compare_without_reference_uses_the_best_row(self) -> None:
        """Use the best displayed algorithm as the baseline when no reference is set."""
        from makespan.compare import compare_instance, format_comparison_text

        instance = parse(STAR)[0]
        report = compare_instance(instance, ["greedy-light", "greedy-load"], None)
        self.assertIsNone(report.reference)
        self.assertEqual(report.baseline_name, "greedy-load")
        self.assertEqual(report.baseline_makespan, 1)
        text = format_comparison_text(report)
        self.assertIn("# best: greedy-load", text)
        self.assertIn("gap and ratio are measured against the best", text)

    def test_missing_reference_solver_is_an_error(self) -> None:
        """Return status 1 when the ILP reference cannot run."""
        from makespan.__main__ import main

        if solver_available():
            self.skipTest("gurobipy is installed, so the ILP reference can run")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "star.txt"
            path.write_text(STAR, encoding="utf-8")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                status = main([str(path), "-C", "greedy-light", "-R", "ilp"])
            self.assertEqual(status, 1)
            self.assertIn("reference ilp", err.getvalue())

    def test_unknown_compare_algorithm(self) -> None:
        """Return status 2 for an unknown ``--compare`` name."""
        from makespan.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "star.txt"
            path.write_text(STAR, encoding="utf-8")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                status = main([str(path), "-C", "not-an-algorithm"])
            self.assertEqual(status, 2)
            self.assertIn("unknown algorithm", err.getvalue())

    def test_balance_flow_matches_small_optima(self) -> None:
        """Check that balance-flow stays feasible and bounds the small optima."""
        expected = {
            ONE_EDGE: 3,
            PARALLEL: 2,
            CYCLE: 1,
            STAR: 1,
            MIXED: 5,
            ISOLATED: 2,
        }
        for text, optimum in expected.items():
            instance = parse(text)[0]
            solution = solve(instance, "balance-flow")
            with self.subTest(instance=instance.name):
                self._assert_feasible(instance, solution)
                self.assertGreaterEqual(solution.makespan, optimum - 1e-6)
                self.assertLessEqual(solution.lower_bound, optimum + 1e-6)
                self.assertAlmostEqual(solution.upper_bound, solution.makespan)

    def test_balance_lp_when_gurobi_is_present(self) -> None:
        """Check balance-lp against brute force when Gurobi is installed."""
        if not solver_available():
            self.skipTest("gurobipy is not installed")
        for text in (ONE_EDGE, PARALLEL, CYCLE, STAR, MIXED, ISOLATED):
            instance = parse(text)[0]
            optimum = _opt(instance)
            solution = solve(instance, "balance-lp")
            with self.subTest(instance=instance.name):
                self._assert_feasible(instance, solution)
                self.assertGreaterEqual(solution.makespan, optimum - 1e-6)
                self.assertLessEqual(solution.lower_bound, optimum + 1e-6)

    def test_ilp_and_lp_rounding_when_gurobi_is_present(self) -> None:
        """Check ILP exactness and LP-rounding's 2-approx when Gurobi is installed."""
        if not solver_available():
            self.skipTest("gurobipy is not installed")
        for seed in range(4):
            instance = _random_instance(4, 6, seed, unit=False)
            optimum = _opt(instance)
            exact = solve(instance, "ilp")
            rounded = solve(instance, "lp-round")
            with self.subTest(seed=seed):
                self._assert_feasible(instance, exact)
                self._assert_feasible(instance, rounded)
                self.assertAlmostEqual(exact.makespan, optimum, places=5)
                self.assertAlmostEqual(exact.lower_bound, optimum, places=5)
                self.assertGreaterEqual(rounded.makespan, optimum - 1e-6)
                self.assertLessEqual(rounded.makespan, 2 * optimum + 1e-5)
                lp_value = rounded.metadata["lp_objective"]
                self.assertLessEqual(rounded.makespan, 2 * lp_value + 1e-5)
                self.assertLessEqual(lp_value, optimum + 1e-5)

    def test_cli_text_and_json(self) -> None:
        """Write text and JSON reports from the command line, and reject unknown algos."""
        import csv

        from makespan.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "in.txt"
            output_dir = Path(directory) / "Outputs"
            path.write_text(MULTI, encoding="utf-8")
            text_out = io.StringIO()
            text_err = io.StringIO()
            with redirect_stdout(text_out), redirect_stderr(text_err):
                status = main(
                    [
                        str(path),
                        "--algo",
                        "greedy-light",
                        "--format",
                        "text",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(status, 0)
            rendered = text_out.getvalue()
            self.assertIn("# instance: one", rendered)
            self.assertIn("# instance: hole", rendered)
            self.assertIn("# makespan: 3", rendered)
            self.assertIn("wrote ", text_err.getvalue())
            runs = [item for item in output_dir.iterdir() if item.is_dir()]
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertRegex(run.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
            assignment = (run / "assignments" / "greedy-light.txt").read_text(encoding="utf-8")
            self.assertIn("# instance: one", assignment)
            self.assertIn("# tags: a, b", assignment)
            self.assertIn("# instance: hole", assignment)
            self.assertIn("# tags: isolated", assignment)
            self.assertIn("0 1 0 3", assignment)
            self.assertIn("1 2 1 4", assignment)
            self.assertIn("# loads", assignment)
            self.assertTrue((run / "solutions.csv").is_file())
            self.assertTrue((run / "loads.csv").is_file())
            self.assertTrue((run / "report.txt").is_file())
            with (run / "summary.csv").open(encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual([row["algorithm"] for row in summary], ["greedy-light"])
            self.assertEqual(summary[0]["solved"], "2")
            self.assertEqual(summary[0]["skipped"], "0")
            self.assertEqual(summary[0]["makespan_min"], "3")
            self.assertEqual(summary[0]["makespan_avg"], "3.5")
            self.assertEqual(summary[0]["makespan_max"], "4")
            self.assertEqual(summary[0]["gap_avg"], "")
            self.assertFalse((run / "Assignment Viewer.html").is_file())

            json_out = io.StringIO()
            with redirect_stdout(json_out), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        str(path),
                        "--algo",
                        "flow-search",
                        "--format",
                        "json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(status, 0)
            records = json.loads(json_out.getvalue())
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["algorithm"], "flow-search")
            self.assertEqual(records[0]["tags"], ["a", "b"])
            self.assertIn("edges", records[0])
            self.assertEqual(records[0]["loads"][0]["node"], 0)
            saved = list(output_dir.glob("*/solutions.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8"))[0]["algorithm"], "flow-search")

            err = io.StringIO()
            with redirect_stderr(err):
                status = main(
                    [str(path), "--algo", "missing", "--output-dir", str(output_dir)]
                )
            self.assertEqual(status, 2)
            self.assertIn("unknown algorithm", err.getvalue())

    def test_summary_csv_uses_min_average_and_max(self) -> None:
        """Aggregate finite metrics and keep an infinite ratio out of the average."""
        import csv as csv_module
        from math import inf

        from makespan.io import format_summary_csv

        text = format_summary_csv(
            {
                "greedy-light": [
                    {
                        "makespan": 3,
                        "lower_bound": 0,
                        "upper_bound": 3,
                        "gap": 0,
                        "ratio": 1,
                    },
                    {
                        "makespan": 6,
                        "lower_bound": 2,
                        "upper_bound": 6,
                        "gap": 2,
                        "ratio": inf,
                    },
                    {},
                ]
            }
        )
        rows = list(csv_module.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["algorithm"], "greedy-light")
        self.assertEqual(row["solved"], "2")
        self.assertEqual(row["skipped"], "1")
        self.assertEqual(row["makespan_min"], "3")
        self.assertEqual(row["makespan_avg"], "4.5")
        self.assertEqual(row["makespan_max"], "6")
        self.assertEqual(row["lower_bound_avg"], "1")
        self.assertEqual(row["gap_min"], "0")
        self.assertEqual(row["gap_avg"], "1")
        self.assertEqual(row["gap_max"], "2")
        self.assertEqual(row["ratio_min"], "1")
        self.assertEqual(row["ratio_avg"], "1")
        self.assertEqual(row["ratio_max"], "inf")

    def test_csv_export_for_solutions_and_comparisons(self) -> None:
        """Write per-edge solve CSV and per-algorithm comparison CSV."""
        import csv as csv_module

        from makespan.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "in.txt"
            output_dir = Path(directory) / "Outputs"
            path.write_text(ONE_EDGE, encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        str(path),
                        "--algo",
                        "greedy-light",
                        "--format",
                        "csv",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(status, 0)
            rows = list(csv_module.DictReader(io.StringIO(out.getvalue())))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["instance"], "one")
            self.assertEqual(rows[0]["tags"], "demo")
            self.assertEqual(rows[0]["algorithm"], "greedy-light")
            self.assertEqual(rows[0]["makespan"], "3")
            self.assertEqual(rows[0]["edge_id"], "1")
            self.assertEqual(rows[0]["u"], "0")
            self.assertEqual(rows[0]["v"], "1")
            self.assertEqual(rows[0]["head"], "0")
            self.assertEqual(rows[0]["weight"], "3")
            self.assertEqual(rows[0]["skipped"], "")
            run = next(item for item in output_dir.iterdir() if item.is_dir())
            with (run / "loads.csv").open(encoding="utf-8") as handle:
                loads = list(csv_module.DictReader(handle))
            self.assertEqual([(row["node"], row["load"]) for row in loads], [("0", "3"), ("1", "0")])

            compared = io.StringIO()
            with redirect_stdout(compared), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        str(path),
                        "-C",
                        "greedy-light",
                        "greedy-load",
                        "-R",
                        "brute",
                        "--format",
                        "csv",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(status, 0)
            table = list(csv_module.DictReader(io.StringIO(compared.getvalue())))
            self.assertEqual(
                [row["algorithm"] for row in table],
                ["greedy-light", "greedy-load", "brute"],
            )
            self.assertEqual(table[0]["gap"], "0")
            self.assertEqual(table[0]["ratio"], "1")
            self.assertEqual(table[-1]["role"], "reference")
            self.assertEqual(table[-1]["baseline_makespan"], "3")
            compared_run = next(
                item for item in output_dir.iterdir() if (item / "comparison.csv").is_file()
            )
            names = sorted(path.name for path in (compared_run / "assignments").iterdir())
            self.assertEqual(names, ["brute.txt", "greedy-light.txt", "greedy-load.txt"])
            self.assertTrue((compared_run / "comparison.csv").is_file())
            self.assertTrue((compared_run / "comparison.txt").is_file())
            with (compared_run / "summary.csv").open(encoding="utf-8") as handle:
                summary = list(csv_module.DictReader(handle))
            by_algorithm = {row["algorithm"]: row for row in summary}
            self.assertEqual(
                [row["algorithm"] for row in summary],
                ["greedy-light", "greedy-load", "brute"],
            )
            self.assertEqual(by_algorithm["greedy-light"]["solved"], "1")
            self.assertEqual(by_algorithm["greedy-light"]["makespan_avg"], "3")
            self.assertEqual(by_algorithm["greedy-light"]["gap_min"], "0")
            self.assertEqual(by_algorithm["greedy-light"]["gap_avg"], "0")
            self.assertEqual(by_algorithm["greedy-light"]["gap_max"], "0")
            self.assertEqual(by_algorithm["brute"]["ratio_avg"], "1")
            self.assertIn(
                "# tags: demo",
                (compared_run / "assignments" / "brute.txt").read_text(encoding="utf-8"),
            )

    def _assert_feasible(self, instance: Instance, solution) -> None:
        """Check that ``solution`` orients every edge to an endpoint of that edge.

        Args:
            instance: Graph the algorithm was given.
            solution: Orientation and metrics to validate.
        """
        self.assertEqual(set(solution.heads), {edge.id for edge in instance.edges})
        for edge in instance.edges:
            self.assertIn(solution.heads[edge.id], (edge.u, edge.v))
        if instance.nodes:
            self.assertAlmostEqual(solution.makespan, max(solution.loads.values()))
        self.assertLessEqual(solution.lower_bound, solution.makespan + 1e-6)
        self.assertGreaterEqual(solution.upper_bound, solution.makespan - 1e-6)


def _all_unit(instance: Instance) -> bool:
    """True when every edge has both direction weights equal to 1.

    Args:
        instance: Graph whose weights are inspected.

    Returns:
        bool: True when every edge has both direction weights equal to 1.
    """
    return all(edge.w_u == 1 and edge.w_v == 1 for edge in instance.edges)


def _random_instance(
    node_count: int, edge_count: int, seed: int, unit: bool
) -> Instance:
    """Build a small random instance on ``0 .. node_count - 1``.

    Args:
        node_count: Number of nodes.
        edge_count: Number of edges, allowing parallel edges.
        seed: RNG seed so the instance is reproducible.
        unit: When true, every weight is 1; otherwise weights are in ``1..6``.

    Returns:
        Instance: Random graph on ``0 .. node_count - 1``.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(edge_count):
        left = rng.randrange(node_count)
        right = rng.randrange(node_count - 1)
        if right >= left:
            right += 1
        if unit:
            weights = (1.0, 1.0)
        else:
            weights = (float(rng.randint(1, 6)), float(rng.randint(1, 6)))
        rows.append((left, right, weights[0], weights[1]))
    edges = [
        Edge(index, left, right, weight_u, weight_v)
        for index, (left, right, weight_u, weight_v) in enumerate(rows, start=1)
    ]
    kind = "unit" if unit else "mixed"
    return Instance(
        f"{kind}-{seed}",
        (),
        tuple(range(node_count)),
        tuple(edges),
    )


if __name__ == "__main__":
    unittest.main()
