"""Compare several orientation algorithms on the same instances.

Each selected algorithm is solved independently. ``--reference brute`` or
``--reference ilp`` also solves an exact method. Gap and ratio are then
measured against that optimum. Without a reference, they are measured
against the best makespan among the algorithms that produced an orientation.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from math import inf, isfinite

from makespan.algorithms import public_algorithms
from makespan.factory import AlgorithmFactory, AlgorithmLimitError
from makespan.model import Instance, Solution, as_number
from makespan.relaxation import SolverUnavailable

# Exact methods the comparison can use as an optimum.
EXACT_ALGORITHMS = ("brute", "ilp")


@dataclass
class AlgorithmOutcome:
    """One algorithm's result on one instance, or the reason it was skipped.

    Args:
        name: Registered algorithm name.
        solution: Completed solve, or ``None`` when the algorithm was skipped.
        skipped: Human-readable skip reason, or ``None`` when it ran.

    Returns:
        AlgorithmOutcome: One algorithm's result or skip record.
    """

    name: str
    solution: Solution | None
    skipped: str | None

    @property
    def makespan(self) -> float | None:
        """Evaluated makespan, or ``None`` when the algorithm did not run.

        Returns:
            float | None: ``solution.makespan`` when the algorithm ran, otherwise ``None``.
        """
        if self.solution is None:
            return None
        return self.solution.makespan


@dataclass
class InstanceComparison:
    """Side-by-side results for every requested algorithm on one instance.

    ``baseline_name`` is the reference algorithm when one was requested and
    it produced a solution. Otherwise it is the algorithm with the smallest
    makespan in this table. Gap and ratio use that baseline.

    Args:
        instance: Graph that every algorithm in this table was given.
        reference: Exact solver name, or ``None`` when no reference was requested.
        rows: Outcome of each algorithm, in the order they were solved.
        baseline_name: Algorithm used as the gap and ratio baseline.
        baseline_makespan: Makespan of that baseline, or ``None`` if none exists.
        reference_failed: True when a requested reference solver could not run.

    Returns:
        InstanceComparison: Side-by-side outcomes and the baseline used for gap and ratio.
    """

    instance: Instance
    reference: str | None
    rows: list[AlgorithmOutcome]
    baseline_name: str | None
    baseline_makespan: float | None
    reference_failed: bool


def resolve_algorithm_names(requested: list[str]) -> list[str]:
    """Turn CLI names into registered algorithms.

    The token ``all`` expands to every public algorithm, in display order.
    Names are returned once, in the order they were requested. ``custom`` is
    rejected because it needs a list of steps supplied in Python.

    Args:
        requested: CLI algorithm names, including the token ``all``.

    Returns:
        list[str]: Registered names once each, in request order.
    """
    known = set(public_algorithms())
    resolved: list[str] = []
    seen: set[str] = set()
    for name in requested:
        choices = public_algorithms() if name == "all" else [name]
        for choice in choices:
            if choice not in known:
                known_text = ", ".join(public_algorithms())
                raise KeyError(
                    f"unknown algorithm {choice!r}; known: {known_text}, all"
                )
            if choice not in seen:
                seen.add(choice)
                resolved.append(choice)
    if not resolved:
        raise KeyError("no algorithms selected")
    return resolved


def compare_instance(
    instance: Instance,
    algorithms: list[str],
    reference: str | None,
) -> InstanceComparison:
    """Solve ``algorithms`` on ``instance`` and attach an optional exact baseline.

    The reference is solved even when it is absent from ``algorithms``, and it
    is solved only once when it is already in the list. A reference that
    cannot run (brute force over the edge limit, or an ILP without Gurobi)
    is recorded as skipped and ``reference_failed`` is set.

    Args:
        instance: Graph every selected algorithm is given.
        algorithms: Registered names to solve and display.
        reference: Exact solver name, or ``None`` to use the best displayed row.

    Returns:
        InstanceComparison: Outcomes for ``algorithms`` plus the optional reference.
    """
    if reference is not None and reference not in EXACT_ALGORITHMS:
        raise ValueError(f"reference must be one of {', '.join(EXACT_ALGORITHMS)}")
    order = list(algorithms)
    if reference is not None and reference not in order:
        order.append(reference)
    outcomes = {name: run_algorithm(instance, name) for name in order}
    rows = [outcomes[name] for name in order]
    reference_failed = False
    baseline_name: str | None = None
    baseline_makespan: float | None = None
    if reference is not None:
        reference_row = outcomes[reference]
        if reference_row.solution is None:
            reference_failed = True
        else:
            baseline_name = reference
            baseline_makespan = reference_row.solution.makespan
    else:
        successful = [row for row in rows if row.makespan is not None]
        if successful:
            best = min(successful, key=lambda row: row.makespan)
            baseline_name = best.name
            baseline_makespan = best.makespan
    return InstanceComparison(
        instance=instance,
        reference=reference,
        rows=rows,
        baseline_name=baseline_name,
        baseline_makespan=baseline_makespan,
        reference_failed=reference_failed,
    )


def run_algorithm(instance: Instance, name: str) -> AlgorithmOutcome:
    """Solve one named algorithm, converting expected limits into a skip reason.

    Args:
        instance: Graph to solve.
        name: Registered algorithm name.

    Returns:
        AlgorithmOutcome: Completed solution, or a skip reason for expected limits.
    """
    print("=" * 80)
    print(f"running algorithm {name}")
    # print("=" * 80)
    try:
        solution = AlgorithmFactory.create(name).solve(instance)
    except SolverUnavailable as exc:
        return AlgorithmOutcome(name, None, str(exc))
    except AlgorithmLimitError as exc:
        return AlgorithmOutcome(name, None, str(exc))
    return AlgorithmOutcome(name, solution, None)


def gap_and_ratio(makespan: float, baseline: float) -> tuple[float, float]:
    """Return ``makespan - baseline`` and ``makespan / baseline``.

    A zero baseline has ratio 1 when the makespan is also zero, and infinity
    when the algorithm still placed a positive load.

    Args:
        makespan: Evaluated maximum load of the compared algorithm.
        baseline: Makespan used as the reference or best-in-table value.

    Returns:
        tuple[float, float]: ``(makespan - baseline, makespan / baseline)``.
    """
    gap = makespan - baseline
    if abs(baseline) <= 1e-12:
        ratio = 1.0 if abs(makespan) <= 1e-12 else inf
    else:
        ratio = makespan / baseline
    return gap, ratio


def format_comparison_text(report: InstanceComparison) -> str:
    """Render one instance comparison as a comment header plus an aligned table.

    Args:
        report: Side-by-side results for one instance.

    Returns:
        str: Comment header plus an aligned table.
    """
    lines = [f"# instance: {report.instance.name}"]
    if report.reference is not None:
        lines.append(f"# reference: {report.reference}")
        if report.baseline_makespan is not None:
            lines.append(f"# optimum: {_fmt(report.baseline_makespan)}")
            lines.append("# gap and ratio are measured against the reference optimum")
        else:
            skipped = _row_by_name(report, report.reference)
            reason = skipped.skipped if skipped is not None else "reference did not run"
            lines.append(f"# optimum: unavailable ({reason})")
    elif report.baseline_name is not None:
        lines.append(f"# best: {report.baseline_name}")
        lines.append(f"# best_makespan: {_fmt(report.baseline_makespan)}")
        lines.append(
            "# gap and ratio are measured against the best algorithm in this table"
        )
    header = [
        "algorithm",
        "role",
        "makespan",
        "gap",
        "ratio",
        "lower_bound",
        "upper_bound",
        "note",
    ]
    table = [header]
    for row in report.rows:
        table.append(_text_row(report, row))
    lines.append(_align(table))
    return "\n".join(lines)


def format_comparison_json(report: InstanceComparison) -> dict:
    """Render one instance comparison as a JSON object.

    Args:
        report: Side-by-side results for one instance.

    Returns:
        dict: JSON object with instance metadata and one record per algorithm.
    """
    rows = []
    for row in report.rows:
        gap, ratio = _metrics(report, row)
        record = {
            "algorithm": row.name,
            "role": _role(report, row.name),
            "makespan": _json_number(row.makespan),
            "gap": _json_number(gap),
            "ratio": _json_number(ratio),
            "lower_bound": None,
            "upper_bound": None,
            "skipped": row.skipped,
        }
        if row.solution is not None:
            record["lower_bound"] = _json_number(row.solution.lower_bound)
            record["upper_bound"] = _json_number(row.solution.upper_bound)
        rows.append(record)
    return {
        "instance": report.instance.name,
        "reference": report.reference,
        "baseline": report.baseline_name,
        "baseline_makespan": _json_number(report.baseline_makespan),
        "reference_failed": report.reference_failed,
        "rows": rows,
    }


_COMPARISON_CSV_COLUMNS = (
    "instance",
    "reference",
    "baseline",
    "baseline_makespan",
    "reference_failed",
    "algorithm",
    "role",
    "makespan",
    "gap",
    "ratio",
    "lower_bound",
    "upper_bound",
    "skipped",
)


def format_comparison_csv(records: list[dict]) -> str:
    """Render comparison records as CSV, one row per algorithm per instance.

    Gap and ratio stay empty when the baseline is missing. A skipped
    algorithm keeps its reason in ``skipped``.

    Args:
        records: JSON comparison objects, one per instance.

    Returns:
        str: CSV text, one row per algorithm per instance.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=_COMPARISON_CSV_COLUMNS,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for record in records:
        for row in record.get("rows") or []:
            writer.writerow(
                {
                    "instance": record.get("instance") or "",
                    "reference": record.get("reference") or "",
                    "baseline": record.get("baseline") or "",
                    "baseline_makespan": _csv_cell(record.get("baseline_makespan")),
                    "reference_failed": record.get("reference_failed", False),
                    "algorithm": row.get("algorithm") or "",
                    "role": row.get("role") or "",
                    "makespan": _csv_cell(row.get("makespan")),
                    "gap": _csv_cell(row.get("gap")),
                    "ratio": _csv_cell(row.get("ratio")),
                    "lower_bound": _csv_cell(row.get("lower_bound")),
                    "upper_bound": _csv_cell(row.get("upper_bound")),
                    "skipped": row.get("skipped") or "",
                }
            )
    return buffer.getvalue()


def _csv_cell(value: object) -> str:
    """Render a CSV cell. Missing values stay empty.

    Args:
        value: Cell contents. ``None`` becomes an empty string.

    Returns:
        str: String form of ``value``, or ``""`` when it is ``None``.
    """
    if value is None:
        return ""
    return str(value)


def _text_row(report: InstanceComparison, row: AlgorithmOutcome) -> list[str]:
    """One table row: metrics when the algorithm ran, dashes when it was skipped.

    Args:
        report: Comparison that supplies the baseline for gap and ratio.
        row: Algorithm outcome to render.

    Returns:
        list[str]: Table cells for this algorithm.
    """
    gap, ratio = _metrics(report, row)
    if row.solution is None:
        return [
            row.name,
            _role(report, row.name),
            "-",
            "-",
            "-",
            "-",
            "-",
            row.skipped or "",
        ]
    return [
        row.name,
        _role(report, row.name),
        _fmt(row.makespan),
        _fmt(gap),
        _fmt(ratio),
        _fmt(row.solution.lower_bound),
        _fmt(row.solution.upper_bound),
        "",
    ]


def _metrics(
    report: InstanceComparison, row: AlgorithmOutcome
) -> tuple[float | None, float | None]:
    """Gap and ratio for one row, or a pair of ``None`` when either side is missing.

    Args:
        report: Comparison that supplies the baseline makespan.
        row: Algorithm outcome whose metrics are computed.

    Returns:
        tuple[float | None, float | None]: Gap and ratio, or ``(None, None)`` when either side is missing.
    """
    if row.makespan is None or report.baseline_makespan is None:
        return None, None
    if report.reference_failed:
        return None, None
    return gap_and_ratio(row.makespan, report.baseline_makespan)


def _role(report: InstanceComparison, name: str) -> str:
    """Mark the exact solver as ``reference`` and every other row as ``compared``.

    Args:
        report: Comparison that may name a reference algorithm.
        name: Algorithm name of the current row.

    Returns:
        str: ``"reference"`` for the exact solver, otherwise ``"compared"``.
    """
    if report.reference is not None and name == report.reference:
        return "reference"
    return "compared"


def _row_by_name(report: InstanceComparison, name: str) -> AlgorithmOutcome | None:
    """Find the outcome recorded for ``name``.

    Args:
        report: Comparison whose rows are searched.
        name: Algorithm name to find.

    Returns:
        AlgorithmOutcome | None: Matching row, or ``None`` if ``name`` is absent.
    """
    for row in report.rows:
        if row.name == name:
            return row
    return None


def _align(rows: list[list[str]]) -> str:
    """Left-align a rectangular table and drop trailing padding on each line.

    Args:
        rows: Table cells, including the header as the first row.

    Returns:
        str: Left-aligned table with trailing padding stripped.
    """
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    """Format a metric the same way single-algorithm text output formats numbers.

    Args:
        value: Metric to render. ``None`` becomes ``-``.

    Returns:
        str: Compact number, ``"inf"``, or ``"-"``.
    """
    if value is None:
        return "-"
    if not isfinite(value):
        return "inf"
    number = as_number(value)
    if isinstance(number, int):
        return str(number)
    return format(number, ".6g")


def _json_number(value: float | None):
    """JSON value for a metric. Non-finite ratios are the string ``inf``.

    Args:
        value: Metric to encode. ``None`` stays JSON null.

    Returns:
        JSON-compatible number, the string ``"inf"``, or ``None``.
    """
    if value is None:
        return None
    if not isfinite(value):
        return "inf"
    return as_number(value)
