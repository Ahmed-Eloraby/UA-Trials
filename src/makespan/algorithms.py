"""Registered pipelines.

This script is the catalog of algorithms a user can name from the command
line. Each function below is a factory: ``AlgorithmFactory.create(name)``
calls it and returns a fresh :class:`Pipeline`. To add an algorithm, write
a :class:`Step` (or reuse one) and register another function here.

``custom`` is the escape hatch for a one-off list of steps. It is omitted
from :func:`public_algorithms` because those steps have to be supplied in
Python rather than on the command line.
"""

from __future__ import annotations

from makespan.balance import BalanceSearchStep, FlowResidualRounder, LpResidualRounder
from makespan.brute import BruteForceStep
from makespan.cycle_balance import CycleBalanceSearchStep
from makespan.factory import AlgorithmFactory, Pipeline, Step
from makespan.flow import FlowSearchStep
from makespan.graph_balance import GraphBalanceSearchStep, PolishStep
from makespan.greedy import GreedyLightestStep, GreedyLoadStep
from makespan.ratio_greedy import RatioGreedyStep
from makespan.relaxation import RelaxationStep
from makespan.rounding import ThresholdRoundStep
from makespan.score_family import (
    METRICS,
    MODES,
    PRUNE_NAMES,
    ScoreStep,
    family_names,
)
from makespan.score_through import ORDERS, ThroughStep, through_names


def _score_catalog() -> list[str]:
    """Score-family names with each through group after its metric and pruning choice.

    ``family_names`` lists twelve level modes per metric and pruning choice.
    The through orders for that choice follow the group. ``flow-desc`` is the
    plain ``through`` name. ``quota`` and ``topo`` are the last two.

    Returns:
        list[str]: Forty-eight level names and twenty-four through names, in display order.
    """
    family = family_names()
    through = through_names()
    width = 2 * len(MODES)
    groups = len(METRICS) * len(PRUNE_NAMES)
    per_group = len(through) // groups
    names: list[str] = []
    for index in range(groups):
        start = index * width
        names.extend(family[start : start + width])
        names.extend(through[index * per_group : (index + 1) * per_group])
    return names


# Display order used by ``--algo all`` and by ``--compare all``.
# Score names sit after ``ratio-greedy``. Each metric and pruning choice lists
# its twelve level modes, then the through orders for that same choice.
ALGORITHM_ORDER = (
    "greedy-light",
    "greedy-load",
    "ratio-greedy",
    *_score_catalog(),
    "brute",
    "flow-search",
    "balance-flow",
    "balance-graph",
    "lp-round",
    "balance-lp",
    "balance-cycle",
    "balance-bridge",
    "ilp",
)


def public_algorithms() -> list[str]:
    """Algorithms a user can name on the command line, in display order.

    ``custom`` is left out. Algorithms registered after this module was
    imported are appended after :data:`ALGORITHM_ORDER`.

    Returns:
        list[str]: Public algorithm names in display order.
    """
    known = set(AlgorithmFactory.available())
    names = [name for name in ALGORITHM_ORDER if name in known]
    names.extend(
        name
        for name in AlgorithmFactory.available()
        if name not in names and name != "custom"
    )
    return names


@AlgorithmFactory.register("greedy-light")
def _greedy_light() -> Pipeline:
    """Orient every unlocked edge toward its lighter endpoint.

    Returns:
        Pipeline: One-step lightest-endpoint algorithm.
    """
    return Pipeline("greedy-light", [GreedyLightestStep()])


@AlgorithmFactory.register("greedy-load")
def _greedy_load() -> Pipeline:
    """Orient each unlocked edge toward the endpoint with the smaller resulting load.

    Returns:
        Pipeline: One-step load-aware greedy algorithm.
    """
    return Pipeline("greedy-load", [GreedyLoadStep()])


@AlgorithmFactory.register("ratio-greedy")
def _ratio_greedy(
    start: str = "light", seed: int = 0, tolerance: float = 1e-7
) -> Pipeline:
    """Peel leaves by weight ratio, orient the core, then propagate overload.

    The core starts at the lighter endpoint unless ``start`` is ``"random"``.
    ``seed`` fixes that random core. Integral weights are bisected on integers.
    Other weights stop once the bracket is narrower than ``tolerance``.

    Args:
        start: ``"light"`` or ``"random"``.
        seed: Seed for a random core. Ignored when ``start`` is ``"light"``.
        tolerance: Width at which a fractional threshold search stops.

    Returns:
        Pipeline: One-step ratio-greedy threshold search.
    """
    return Pipeline(
        "ratio-greedy",
        [RatioGreedyStep(start=start, seed=seed, tolerance=tolerance)],
    )


@AlgorithmFactory.register("brute")
def _brute() -> Pipeline:
    """Exact branch and bound. ``max_edges`` limits how many free edges are enumerated.

    Args:
        max_edges: Maximum number of unlocked edges brute force will enumerate.

    Returns:
        Pipeline: Lightest-greedy seed followed by exact branch and bound.
    """
    return Pipeline(
        "brute",
        [GreedyLightestStep(), BruteForceStep()],
    )


@AlgorithmFactory.register("ilp")
def _ilp(time_limit: float | None = None) -> Pipeline:
    """Exact Gurobi integer program. ``time_limit`` is optional, in seconds.

    Args:
        time_limit: Gurobi time limit in seconds, or ``None`` for the solver default.

    Returns:
        Pipeline: Exact integer program over unlocked edges.
    """
    return Pipeline("ilp", [RelaxationStep(integral=True, time_limit=time_limit)])


@AlgorithmFactory.register("lp-round")
def _lp_round(threshold: float = 0.5, time_limit: float | None = None) -> Pipeline:
    """Solve the LP, then round each edge toward an endpoint with fraction at least ``threshold``.

    Args:
        threshold: Minimum LP fraction that makes an endpoint eligible when rounding.
        time_limit: Gurobi time limit in seconds, or ``None`` for the solver default.

    Returns:
        Pipeline: LP relaxation followed by threshold rounding.
    """
    return Pipeline(
        "lp-round",
        [
            RelaxationStep(integral=False, time_limit=time_limit),
            ThresholdRoundStep(threshold),
        ],
    )


@AlgorithmFactory.register("balance-flow")
def _balance_flow(eps: float = 0.25) -> Pipeline:
    """Fenced threshold search whose residual edges are rounded by max flow.

    Args:
        eps: Relative gap at which the search fences are close enough to stop.

    Returns:
        Pipeline: Fenced search whose residual edges are rounded by max flow.
    """
    return Pipeline(
        "balance-flow",
        [BalanceSearchStep(FlowResidualRounder(), eps=eps)],
    )


@AlgorithmFactory.register("balance-graph")
def _balance_graph(eps: float = 0.25) -> Pipeline:
    """LP-free threshold search: exact graph reductions, then a certified push, then polishing.

    Args:
        eps: Relative gap between the accepted threshold and the lower fence at which to stop.

    Returns:
        Pipeline: Graph balance search followed by flip-chain polishing.
    """
    return Pipeline(
        "balance-graph",
        [GraphBalanceSearchStep(eps=eps), PolishStep()],
    )


@AlgorithmFactory.register("balance-lp")
def _balance_lp(eps: float = 0.25) -> Pipeline:
    """Fenced threshold search whose residual edges are rounded from the capacitated LP.

    Args:
        eps: Relative gap at which the search fences are close enough to stop.

    Returns:
        Pipeline: Fenced search whose residual edges are rounded from the capacitated LP.
    """
    return Pipeline(
        "balance-lp",
        [BalanceSearchStep(LpResidualRounder(), eps=eps)],
    )


@AlgorithmFactory.register("balance-cycle")
def _balance_cycle(eps: float = 0.25) -> Pipeline:
    """Fenced threshold search whose residual cycles are broken in both directions.

    Args:
        eps: Relative gap at which the search fences are close enough to stop.

    Returns:
        Pipeline: Fenced search that breaks residual cycles without an LP.
    """
    return Pipeline(
        "balance-cycle",
        [CycleBalanceSearchStep(use_bridges=False, eps=eps)],
    )


@AlgorithmFactory.register("balance-bridge")
def _balance_bridge(eps: float = 0.25) -> Pipeline:
    """Fenced threshold search that splits bridge bundles, then breaks cycles.

    A bridge channel is split using the spare capacity of each side of the
    cut. The cycles inside those sides are then broken as in ``balance-cycle``.

    Args:
        eps: Relative gap at which the search fences are close enough to stop.

    Returns:
        Pipeline: Fenced search that peels bridges before breaking cycles.
    """
    return Pipeline(
        "balance-bridge",
        [CycleBalanceSearchStep(use_bridges=True, eps=eps)],
    )


@AlgorithmFactory.register("flow-search")
def _flow_search(tolerance: float = 1e-7) -> Pipeline:
    """Greedy upper bound, then binary search on the choice-network threshold.

    Args:
        tolerance: Width at which a fractional threshold search stops.

    Returns:
        Pipeline: Two greedy seeds followed by choice-network binary search.
    """
    return Pipeline(
        "flow-search",
        [GreedyLightestStep(), GreedyLoadStep(), FlowSearchStep(tolerance)],
    )


@AlgorithmFactory.register("custom")
def _custom(steps: list[Step], name: str = "custom") -> Pipeline:
    """Build a one-off pipeline without adding a permanent command-line name.

    Args:
        steps: Transformations applied in order on a fresh context.
        name: Label written on the returned :class:`Pipeline`.

    Returns:
        Pipeline: One-off pipeline built from ``steps``.
    """
    return Pipeline(name, list(steps))


def _register_score_family() -> None:
    """Register the forty-eight score-family command-line names.

    Each mode is ``score-{ratio|diff}-{peel|raw}-{mode}`` plus an ``early``
    name that orients same-level edges first. The builder closes over its
    own metric, pruning flag, mode, and timing through default arguments.
    """

    def register(
        metric: str,
        prune: bool,
        mode: str,
        same_first: bool,
        name: str,
    ) -> None:
        """Register one score-family pipeline under ``name``.

        Args:
            metric: ``"ratio"`` or ``"diff"``.
            prune: True when pendant bundles are peeled before scoring.
            mode: Stage-3 order.
            same_first: True when same-level edges are oriented before the cuts.
            name: Command-line name of the pipeline.
        """

        def builder(
            metric: str = metric,
            prune: bool = prune,
            mode: str = mode,
            name: str = name,
            same_first: bool = same_first,
            tolerance: float = 1e-7,
        ) -> Pipeline:
            """Build one score-family pipeline.

            Args:
                metric: ``"ratio"`` or ``"diff"``.
                prune: True when pendant bundles are peeled before scoring.
                mode: Stage-3 order.
                name: Label written on the returned pipeline.
                same_first: True when same-level edges are oriented before the cuts.
                tolerance: Width at which a fractional threshold search stops.

            Returns:
                Pipeline: One-step score-family threshold search.
            """
            return Pipeline(
                name,
                [
                    ScoreStep(
                        metric=metric,
                        prune=prune,
                        mode=mode,
                        same_first=same_first,
                        tolerance=tolerance,
                    )
                ],
            )

        AlgorithmFactory.register(name)(builder)

    for metric in METRICS:
        for prune_name, prune in PRUNE_NAMES:
            for mode in MODES:
                for same_first, tag in ((False, ""), (True, "-early")):
                    register(
                        metric,
                        prune,
                        mode,
                        same_first,
                        f"score-{metric}-{prune_name}{tag}-{mode}",
                    )


def _register_score_through() -> None:
    """Register the twenty-four through command-line names.

    Each plain name is ``score-{ratio|diff}-{peel|raw}-through`` and walks
    nodes from the largest flow. The other orders add ``-flow-asc``,
    ``-flow-room-asc``, ``-flow-room-desc``, ``-quota``, or ``-topo``.
    There is no ``early`` name.
    """

    def register(metric: str, prune: bool, order: str, name: str) -> None:
        """Register one through pipeline under ``name``.

        Args:
            metric: ``"ratio"`` or ``"diff"``.
            prune: True when pendant bundles are peeled before scoring.
            order: Whose turn comes first.
            name: Command-line name of the pipeline.
        """

        def builder(
            metric: str = metric,
            prune: bool = prune,
            order: str = order,
            name: str = name,
            tolerance: float = 1e-7,
        ) -> Pipeline:
            """Build one through pipeline.

            Args:
                metric: ``"ratio"`` or ``"diff"``.
                prune: True when pendant bundles are peeled before scoring.
                order: Whose turn comes first.
                name: Label written on the returned pipeline.
                tolerance: Width at which a fractional threshold search stops.

            Returns:
                Pipeline: One-step through-flow threshold search.
            """
            return Pipeline(
                name,
                [
                    ThroughStep(
                        metric=metric,
                        prune=prune,
                        order=order,
                        tolerance=tolerance,
                    )
                ],
            )

        AlgorithmFactory.register(name)(builder)

    for metric in METRICS:
        for prune_name, prune in PRUNE_NAMES:
            for order in ORDERS:
                suffix = "" if order == "flow-desc" else f"-{order}"
                register(
                    metric,
                    prune,
                    order,
                    f"score-{metric}-{prune_name}-through{suffix}",
                )


_register_score_family()
_register_score_through()
