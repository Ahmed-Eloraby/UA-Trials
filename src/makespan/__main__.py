"""Command line for the orientation algorithms.

From the project root::

    python solve.py instances.txt --algo lp-round
    python solve.py instances.txt --algo all --format json
    python solve.py instances.txt -C greedy-light flow-search -R brute --format csv

``--compare`` / ``-C`` solves several algorithms and prints one table per
instance. ``--reference`` / ``-R`` adds an exact solver, ``brute`` or
``ilp``, and measures every row against that optimum.

Every successful run also writes ``Outputs/<timestamp>/``. That folder holds
one assignment text file per algorithm, ``solutions.csv``, ``loads.csv``,
``summary.csv``, and the comparison report when several algorithms were run.
``summary.csv`` is one row per algorithm: how many instances were solved or
skipped, then the minimum, average, and maximum of makespan and bounds.
A comparison also fills gap and ratio the same way. The per-instance report
is still printed to the terminal. ``--output-dir`` changes the parent folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from makespan.algorithms import public_algorithms
from makespan.compare import (
    compare_instance,
    format_comparison_csv,
    format_comparison_json,
    format_comparison_text,
    resolve_algorithm_names,
)
from makespan.factory import AlgorithmFactory, AlgorithmLimitError
from makespan.io import (
    assignment_filename,
    format_assignment_file,
    format_json_record,
    format_loads_csv,
    format_skip,
    format_solution_csv,
    format_summary_csv,
    format_text,
    parse,
)
from makespan.model import Instance, Solution
from makespan.relaxation import SolverUnavailable


@dataclass
class Attempt:
    """One algorithm's try on one instance.

    Args:
        instance: Graph that was given to the algorithm.
        algorithm: Registered name that was requested.
        solution: Completed orientation, or ``None`` when the algorithm was skipped.
        skipped: Human-readable skip reason, or ``None`` when it ran.

    Returns:
        Attempt: One solve or one skip.
    """

    instance: Instance
    algorithm: str
    solution: Solution | None = None
    skipped: str | None = None


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, solve each instance, and write the report.

    Returns 0 on success, 2 when the invocation or the input file is invalid,
    and 1 when a requested algorithm or reference solver fails.

    Args:
        argv: Command-line arguments without the program name. ``None`` uses ``sys.argv``.

    Returns:
        int: ``0`` on success, ``2`` for invalid input, ``1`` when a solver fails.
    """
    parser = argparse.ArgumentParser(
        description="Orient edges to minimize the maximum node load."
    )
    parser.add_argument("path", help="Graph Instance Builder export (.txt, .csv, or .json)")
    parser.add_argument(
        "--algo",
        default="lp-round",
        help="algorithm name, or 'all' (default: lp-round). Ignored when --compare is set.",
    )
    parser.add_argument(
        "-C",
        "--compare",
        nargs="+",
        metavar="ALGO",
        help="compare these algorithms (names, or 'all') on each instance",
    )
    parser.add_argument(
        "-R",
        "--reference",
        choices=("brute", "ilp"),
        help="exact solver to measure the other algorithms against: brute or ilp",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="output format: text, json, or csv (default: text)",
    )
    parser.add_argument("--out", help="also write the terminal report to this file")
    parser.add_argument(
        "--output-dir",
        default="Outputs",
        help="parent folder for timestamped run directories (default: Outputs)",
    )

    args = parser.parse_args(argv)

    try:
        instances = parse(args.path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.compare is not None or args.reference is not None:
        return _compare(args, instances)

    try:
        names = _algorithm_names(args.algo)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records: list[dict] = []
    blocks: list[str] = []
    attempts: list[Attempt] = []
    for instance in instances:
        for name in names:
            attempt, record, block, status = _run_one(instance, name, args.algo)
            if status != 0:
                return status
            if attempt is not None:
                attempts.append(attempt)
            if record is not None:
                records.append(record)
            if block is not None:
                blocks.append(block)
    return _emit(
        args.format,
        args.out,
        args.output_dir,
        records,
        blocks,
        attempts,
        comparison=False,
    )


def _compare(args, instances: list[Instance]) -> int:
    """Run the comparison report. A failed reference solver returns exit code 1.

    Args:
        args: Parsed command-line namespace with compare, reference, format, and out.
        instances: Graphs loaded from the input file.

    Returns:
        int: ``0`` on success, ``2`` for unknown algorithms, ``1`` when a reference fails.
    """
    requested = args.compare if args.compare is not None else [args.algo]
    try:
        names = resolve_algorithm_names(requested)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records: list[dict] = []
    blocks: list[str] = []
    attempts: list[Attempt] = []
    failed = False
    for instance in instances:
        print("="*100)
        print(f"solving instance {instance.name}")
        try:
            report = compare_instance(instance, names, args.reference)
        except (RuntimeError, ValueError) as exc:
            print(f"error: {instance.name}: {exc}", file=sys.stderr)
            return 1
        if report.reference_failed:
            failed = True
            reason = _reference_reason(report)
            print(
                f"error: {instance.name}: reference {args.reference}: {reason}",
                file=sys.stderr,
            )
        for row in report.rows:
            attempts.append(
                Attempt(instance, row.name, row.solution, row.skipped)
            )
        records.append(format_comparison_json(report))
        blocks.append(format_comparison_text(report))
    status = _emit(
        args.format,
        args.out,
        args.output_dir,
        records,
        blocks,
        attempts,
        comparison=True,
    )
    if status != 0:
        return status
    return 1 if failed else 0


def _reference_reason(report) -> str:
    """Skip reason recorded on the reference row.

    Args:
        report: Comparison whose reference row may contain a skip reason.

    Returns:
        str: Skip reason from the reference row, or a fallback message.
    """
    for row in report.rows:
        if row.name == report.reference and row.skipped:
            return row.skipped
    return "reference did not produce a solution"


def _emit(
    fmt: str,
    out: str | None,
    output_dir: str,
    records: list[dict],
    blocks: list[str],
    attempts: list[Attempt],
    comparison: bool,
) -> int:
    """Print the report and archive every output in a timestamped run folder.

    The terminal report is unchanged: JSON, per-edge CSV, or text. ``--out``
    writes that same report to an extra path. The run folder always receives
    assignment text files, ``solutions.csv``, ``loads.csv``, and ``summary.csv``.
    A comparison also receives ``comparison.csv`` and ``comparison.txt``.

    Args:
        fmt: Output format, one of ``text``, ``json``, or ``csv``.
        out: Extra path for the terminal report, or ``None``.
        output_dir: Parent directory. A ``YYYY-MM-DD_HH-MM-SS`` folder is created inside it.
        records: JSON objects for json and csv output.
        blocks: Text blocks for the text format.
        attempts: Every solve and skip, used to write assignment files and loads.
        comparison: When true, the terminal CSV is the comparison table.

    Returns:
        int: ``0`` when the folder was written, ``1`` when saving it failed.
    """
    rendered = _render(fmt, records, blocks, comparison)
    try:
        run_dir = _write_run(Path(output_dir), fmt, rendered, records, blocks, attempts, comparison)
    except OSError as exc:
        print(f"error: could not write {output_dir}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {run_dir.resolve()}", file=sys.stderr)
    if out:
        try:
            with open(out, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except OSError as exc:
            print(f"error: could not write {out}: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(rendered)
    return 0


def _render(fmt: str, records: list[dict], blocks: list[str], comparison: bool) -> str:
    """Render the terminal report.

    Args:
        fmt: Output format, one of ``text``, ``json``, or ``csv``.
        records: JSON objects for json and csv output.
        blocks: Text blocks for the text format.
        comparison: When true, render comparison CSV instead of per-edge solve CSV.

    Returns:
        str: The report text, ending in a newline when it is not empty.
    """
    if fmt == "json":
        return json.dumps(records, indent=2) + "\n"
    if fmt == "csv":
        if comparison:
            return format_comparison_csv(records)
        return format_solution_csv(records)
    return _join_blocks(blocks)


def _algorithm_names(algo: str) -> list[str]:
    """Resolve ``--algo``, including the ``all`` token, for a normal solve.

    Args:
        algo: Requested algorithm name, or ``all``.

    Returns:
        list[str]: Registered names to run.
    """
    if algo == "all":
        return public_algorithms()
    known = set(AlgorithmFactory.available())
    if algo == "custom" or algo not in known:
        known_text = ", ".join(public_algorithms())
        raise KeyError(f"unknown algorithm {algo!r}; known: {known_text}, all")
    return [algo]


def _run_one(
    instance: Instance,
    name: str,
    requested: str,
) -> tuple[Attempt | None, dict | None, str | None, int]:
    """Solve one algorithm. ``--algo all`` skips brute-force and Gurobi limits.

    A single named algorithm treats those limits as errors and returns status 1.

    Args:
        instance: Graph to solve.
        name: Registered algorithm to run.
        requested: Original ``--algo`` token, used to decide whether limits are skipped.

    Returns:
        tuple[Attempt | None, dict | None, str | None, int]: Attempt, JSON record,
        text block, and process status.
    """

    try:
        solution = AlgorithmFactory.create(name).solve(instance)
    except SolverUnavailable as exc:
        if requested != "all":
            print(f"error: {exc}", file=sys.stderr)
            return None, None, None, 1
        reason = str(exc)
        return (
            Attempt(instance, name, None, reason),
            {
                "instance": instance.name,
                "tags": list(instance.tags),
                "algorithm": name,
                "skipped": reason,
            },
            format_skip(instance, name, reason),
            0,
        )
    except AlgorithmLimitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None, None, None, 1
    except (RuntimeError, ValueError) as exc:
        print(f"error: {instance.name}: {name}: {exc}", file=sys.stderr)
        return None, None, None, 1
    return _success(instance, solution)


def _success(instance: Instance, solution: Solution) -> tuple[Attempt, dict, str, int]:
    """Package a finished solution for the terminal and the run folder.

    Args:
        instance: Graph that was solved.
        solution: Completed orientation and metrics.

    Returns:
        tuple[Attempt, dict, str, int]: Attempt, JSON record, text block, and status ``0``.
    """
    return (
        Attempt(instance, solution.algorithm, solution, None),
        format_json_record(instance, solution),
        format_text(instance, solution),
        0,
    )


def _write_run(
    root: Path,
    fmt: str,
    rendered: str,
    records: list[dict],
    blocks: list[str],
    attempts: list[Attempt],
    comparison: bool,
) -> Path:
    """Create the timestamped folder and write every artifact of this run.

    Args:
        root: Parent directory, created when it does not exist.
        fmt: Terminal format, used to decide which extra report file to store.
        rendered: Exact terminal report.
        records: Comparison JSON records when ``comparison`` is true, otherwise unused.
        blocks: Comparison text blocks when ``comparison`` is true.
        attempts: Solves and skips. Only successful solves become assignment files.
        comparison: When true, also store the comparison table as text and CSV.
            Gap and ratio in ``summary.csv`` are filled only in this case.

    Returns:
        Path: The new run directory.
    """
    run_dir = _fresh_run_dir(root)
    grouped = _assignments_by_algorithm(attempts)
    assignments = run_dir / "assignments"
    assignments.mkdir()
    used: set[str] = set()
    for algorithm, pairs in grouped.items():
        filename = assignment_filename(algorithm)
        stem = filename[:-4]
        suffix = 2
        while filename in used:
            filename = f"{stem}_{suffix}.txt"
            suffix += 1
        used.add(filename)
        _write_text(assignments / filename, format_assignment_file(pairs))
    solved = [(instance, solution) for pairs in grouped.values() for instance, solution in pairs]
    _write_text(run_dir / "loads.csv", format_loads_csv(solved))
    _write_text(run_dir / "solutions.csv", format_solution_csv(_solution_records(attempts)))
    _write_text(
        run_dir / "summary.csv",
        format_summary_csv(_summary_groups(attempts, records if comparison else None)),
    )
    if comparison:
        _write_text(run_dir / "comparison.csv", format_comparison_csv(records))
        _write_text(run_dir / "comparison.txt", _join_blocks(blocks))
        if fmt == "json":
            _write_text(run_dir / "comparison.json", rendered)
    elif fmt == "json":
        _write_text(run_dir / "solutions.json", rendered)
    elif fmt == "text":
        _write_text(run_dir / "report.txt", rendered)
    return run_dir


def _summary_groups(
    attempts: list[Attempt],
    comparison_records: list[dict] | None,
) -> dict[str, list[dict]]:
    """Per-algorithm samples used by ``summary.csv``.

    Gap and ratio are copied from the comparison records when this run
    compared algorithms. A skipped attempt is an empty sample.

    Args:
        attempts: Solves and skips in the order they were produced.
        comparison_records: Comparison JSON objects, or ``None`` for a plain solve.

    Returns:
        dict[str, list[dict]]: Algorithm name to one sample per attempt.
    """
    compared: dict[tuple[str, str], dict] = {}
    if comparison_records is not None:
        for record in comparison_records:
            instance_name = record.get("instance") or ""
            for row in record.get("rows") or []:
                compared[(instance_name, row.get("algorithm") or "")] = row
    groups: dict[str, list[dict]] = {}
    for attempt in attempts:
        name = attempt.solution.algorithm if attempt.solution is not None else attempt.algorithm
        if attempt.solution is None:
            groups.setdefault(name, []).append({})
            continue
        sample = {
            "makespan": attempt.solution.makespan,
            "lower_bound": attempt.solution.lower_bound,
            "upper_bound": attempt.solution.upper_bound,
        }
        row = compared.get((attempt.instance.name, name))
        if row is not None:
            sample["gap"] = row.get("gap")
            sample["ratio"] = row.get("ratio")
        groups.setdefault(name, []).append(sample)
    return groups


def _assignments_by_algorithm(
    attempts: list[Attempt],
) -> dict[str, list[tuple[Instance, Solution]]]:
    """Group successful orientations by algorithm, preserving run order.

    Args:
        attempts: Solves and skips in the order they were produced.

    Returns:
        dict[str, list[tuple[Instance, Solution]]]: Algorithm name to its solved instances.
    """
    grouped: dict[str, list[tuple[Instance, Solution]]] = {}
    for attempt in attempts:
        if attempt.solution is None:
            continue
        grouped.setdefault(attempt.solution.algorithm, []).append(
            (attempt.instance, attempt.solution)
        )
    return grouped


def _solution_records(attempts: list[Attempt]) -> list[dict]:
    """JSON solve records for ``solutions.csv``, including skipped algorithms.

    Args:
        attempts: Solves and skips in the order they were produced.

    Returns:
        list[dict]: One record per attempt. Skips carry ``tags`` and ``skipped``.
    """
    records: list[dict] = []
    for attempt in attempts:
        if attempt.solution is None:
            records.append(
                {
                    "instance": attempt.instance.name,
                    "tags": list(attempt.instance.tags),
                    "algorithm": attempt.algorithm,
                    "skipped": attempt.skipped or "",
                }
            )
        else:
            records.append(format_json_record(attempt.instance, attempt.solution))
    return records


def _fresh_run_dir(root: Path) -> Path:
    """Create ``root/<timestamp>``, adding a numeric suffix if that second is taken.

    Args:
        root: Parent directory. It is created when missing.

    Returns:
        Path: The empty directory for this run.
    """
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = root / stamp
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _join_blocks(blocks: list[str]) -> str:
    """Join text blocks the same way the terminal report does.

    Args:
        blocks: Report sections, without trailing newlines.

    Returns:
        str: Sections separated by a blank line, ending in a newline when not empty.
    """
    rendered = "\n\n".join(blocks)
    if rendered:
        rendered += "\n"
    return rendered


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8 with ``\\n`` line endings.

    Args:
        path: Destination file. Parent directories must already exist.
        text: Full file contents.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
