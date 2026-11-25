"""Pool comparison runs from Outputs into one summary.

Edit the timestamps below, then run this file from the IDE. A run folder is
included when its timestamp is inside DATE_FROM and DATE_TO, both minutes
included. Means are weighted by the number of instances in each run. Ratio
and gap are measured against brute force. Runs where brute did not finish
are left out of the averages and listed at the end.

The script writes three CSV files next to the run folders:

- summary_<from>_to_<to>.csv: one row per algorithm, including the instance
  that attained the worst ratio
- summary_<from>_to_<to>_by_edges.csv: mean ratio by edge count
- summary_<from>_to_<to>_by_nodes.csv: mean ratio by node count
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Edit these, then run this file. Format is YYYY-MM-DD HH:MM.
DATE_FROM = "2026-11-24 12:00"
DATE_TO = "2026-11-24 23:59"
OUTPUT_DIR = Path(__file__).resolve().parent / "Outputs"

_RUN_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")
_INSTANCE_SIZE = re.compile(r"^n(\d+)-m(\d+)-")


def main() -> None:
    """Pool every run folder in the date range and write the CSV files."""
    first = _parse_stamp(DATE_FROM)
    last = _parse_stamp(DATE_TO).replace(second=59)
    if first > last:
        raise ValueError(f"DATE_FROM {DATE_FROM} is after DATE_TO {DATE_TO}")
    runs = _runs_in_range(OUTPUT_DIR, first, last)
    if not runs:
        raise ValueError(
            f"no run folders in {OUTPUT_DIR} from {DATE_FROM} to {DATE_TO}"
        )

    pooled: dict[str, _Totals] = {}
    by_edges: dict[int, dict[str, _Totals]] = defaultdict(dict)
    by_nodes: dict[int, dict[str, _Totals]] = defaultdict(dict)
    peaks: dict[Path, dict[str, float]] = {}
    skipped: list[str] = []
    used = 0
    for run_dir in runs:
        loaded = _load_run(run_dir)
        if loaded is None:
            skipped.append(run_dir.name)
            continue
        used += 1
        nodes, edges, rows = loaded
        peaks[run_dir] = {row["algorithm"]: row["worst_ratio"] for row in rows}
        for row in rows:
            _add(pooled, row["algorithm"], row)
            if edges is not None:
                _add(by_edges[edges], row["algorithm"], row)
            if nodes is not None:
                _add(by_nodes[nodes], row["algorithm"], row)

    if not pooled:
        raise ValueError(
            f"no brute-scored runs from {DATE_FROM} to {DATE_TO}; "
            f"skipped {len(skipped)} folder(s)"
        )

    order = sorted(pooled, key=lambda name: pooled[name].mean_ratio)
    worst = _worst_instances(peaks, pooled)
    prefix = OUTPUT_DIR / (
        f"summary_{_file_stamp(DATE_FROM)}_to_{_file_stamp(DATE_TO)}"
    )
    _write_algorithms(prefix.with_suffix(".csv"), order, pooled, worst)
    _write_breakdown(
        prefix.with_name(prefix.name + "_by_edges.csv"), by_edges, order, "edges"
    )
    _write_breakdown(
        prefix.with_name(prefix.name + "_by_nodes.csv"), by_nodes, order, "nodes"
    )

    print(
        f"{used} runs from {DATE_FROM} to {DATE_TO}, {skipped and len(skipped) or 0} skipped"
    )
    print(f"{'algorithm':<44} {'ratio':>8} {'gap':>8} {'worst':>8}  instance")
    for name in order:
        rec = pooled[name]
        hit = worst[name]
        instance = hit["instance"] if hit else ""
        print(
            f"{name:<44} {rec.mean_ratio:8.4f} {rec.mean_gap:8.3f} "
            f"{rec.worst_ratio:8.4f}  {instance}"
        )
    for path in (
        prefix.with_suffix(".csv"),
        prefix.with_name(prefix.name + "_by_edges.csv"),
        prefix.with_name(prefix.name + "_by_nodes.csv"),
    ):
        print(path)
    if skipped:
        print("skipped, brute did not score them:")
        for name in skipped:
            print(f"  {name}")


class _Totals:
    """Instance-weighted ratio and gap for one algorithm."""

    def __init__(self) -> None:
        self.runs = 0
        self.instances = 0
        self._ratio_sum = 0.0
        self._gap_sum = 0.0
        self.worst_ratio = 0.0
        self.worst_gap = 0.0

    def add(
        self,
        instances: int,
        mean_ratio: float,
        mean_gap: float,
        worst_ratio: float,
        worst_gap: float,
    ) -> None:
        self.runs += 1
        self.instances += instances
        self._ratio_sum += mean_ratio * instances
        self._gap_sum += mean_gap * instances
        self.worst_ratio = max(self.worst_ratio, worst_ratio)
        self.worst_gap = max(self.worst_gap, worst_gap)

    @property
    def mean_ratio(self) -> float:
        return self._ratio_sum / self.instances

    @property
    def mean_gap(self) -> float:
        return self._gap_sum / self.instances


def _parse_stamp(text: str) -> datetime:
    """Parse ``YYYY-MM-DD HH:MM``."""
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError(f"timestamp must be YYYY-MM-DD HH:MM, got {text!r}") from exc


def _file_stamp(text: str) -> str:
    """Turn a timestamp into a filename piece, such as ``2026-11-22_1332``."""
    return _parse_stamp(text).strftime("%Y-%m-%d_%H%M")


def _runs_in_range(output_dir: Path, first: datetime, last: datetime) -> list[Path]:
    """Run folders whose timestamp falls inside the inclusive minute range."""
    chosen = []
    if not output_dir.is_dir():
        raise ValueError(f"output directory does not exist: {output_dir}")
    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue
        matched = _RUN_STAMP.match(folder.name)
        if matched is None:
            continue
        folder_stamp = datetime.strptime(
            f"{matched.group(1)} {matched.group(2)}:{matched.group(3)}:{matched.group(4)}",
            "%Y-%m-%d %H:%M:%S",
        )
        if first <= folder_stamp <= last:
            chosen.append(folder)
    chosen.sort(key=lambda folder: folder.name)
    return chosen


def _load_run(run_dir: Path) -> tuple[int | None, int | None, list[dict]] | None:
    """Per-algorithm rows for one run, or ``None`` when brute did not score it.

    Each row has ``algorithm``, ``instances``, ``mean_ratio``, ``mean_gap``,
    ``worst_ratio``, and ``worst_gap``. ``summary.csv`` supplies those totals
    when this run wrote one. Otherwise they are computed from ``comparison.csv``.
    """
    nodes, edges = _size_of(run_dir)
    summary = run_dir / "summary.csv"
    if summary.exists():
        rows = _rows_from_summary(summary)
    else:
        rows = _rows_from_comparison(run_dir / "comparison.csv")
    if not rows:
        return None
    return nodes, edges, rows


def _size_of(run_dir: Path) -> tuple[int | None, int | None]:
    """Node and edge counts from the first instance name in the comparison text."""
    comparison = run_dir / "comparison.txt"
    if not comparison.exists():
        return None, None
    with comparison.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("# instance:"):
                continue
            matched = _INSTANCE_SIZE.match(line.split(":", 1)[1].strip())
            if matched is None:
                return None, None
            return int(matched.group(1)), int(matched.group(2))
    return None, None


def _rows_from_summary(path: Path) -> list[dict]:
    """Algorithm rows from ``summary.csv``. Brute and unscored rows are omitted."""
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            name = record["algorithm"]
            if name == "brute" or not record["ratio_avg"]:
                continue
            rows.append(
                {
                    "algorithm": name,
                    "instances": int(record["solved"]),
                    "mean_ratio": float(record["ratio_avg"]),
                    "mean_gap": float(record["gap_avg"]),
                    "worst_ratio": float(record["ratio_max"]),
                    "worst_gap": float(record["gap_max"]),
                }
            )
    return rows


def _rows_from_comparison(path: Path) -> list[dict]:
    """The same rows, totaled from a comparison that has no ``summary.csv``."""
    if not path.exists():
        return []
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            name = record["algorithm"]
            if name == "brute" or record["role"] == "reference" or not record["ratio"]:
                continue
            buckets[name].append((float(record["ratio"]), float(record["gap"])))
    rows = []
    for name, samples in buckets.items():
        count = len(samples)
        rows.append(
            {
                "algorithm": name,
                "instances": count,
                "mean_ratio": sum(ratio for ratio, _gap in samples) / count,
                "mean_gap": sum(gap for _ratio, gap in samples) / count,
                "worst_ratio": max(ratio for ratio, _gap in samples),
                "worst_gap": max(gap for _ratio, gap in samples),
            }
        )
    return rows


def _add(groups: dict[str, _Totals], name: str, row: dict) -> None:
    """Add one run's totals for ``name``."""
    totals = groups.get(name)
    if totals is None:
        totals = _Totals()
        groups[name] = totals
    totals.add(
        row["instances"],
        row["mean_ratio"],
        row["mean_gap"],
        row["worst_ratio"],
        row["worst_gap"],
    )


def _worst_instances(
    peaks: dict[Path, dict[str, float]],
    pooled: dict[str, _Totals],
) -> dict[str, dict | None]:
    """Instance, if any, that attains each algorithm's worst ratio.

    Only comparison files whose own maximum reaches the pooled maximum are
    opened. Ties keep the instance with the larger gap. ``other_instances``
    lists the remaining ties.

    Args:
        peaks: Run folder to each algorithm's worst ratio in that run.
        pooled: Totals whose ``worst_ratio`` is the target for each algorithm.

    Returns:
        dict[str, dict | None]: Algorithm name to the winning instance record.
    """
    found: dict[str, list[dict]] = {name: [] for name in pooled}
    for run_dir, run_peaks in peaks.items():
        comparison = run_dir / "comparison.csv"
        if not comparison.exists():
            continue
        wanted = {
            name: pooled[name].worst_ratio
            for name, peak in run_peaks.items()
            if name in pooled and _close(peak, pooled[name].worst_ratio)
        }
        if not wanted:
            continue
        with comparison.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                name = record["algorithm"]
                if name not in wanted or not record["ratio"]:
                    continue
                ratio = float(record["ratio"])
                target = wanted[name]
                if ratio + 1e-7 < target:
                    continue
                if abs(ratio - target) > 1e-6 * max(1.0, abs(target)):
                    continue
                found[name].append(
                    {
                        "instance": record["instance"],
                        "makespan": record["makespan"],
                        "optimum": record["baseline_makespan"],
                        "gap": record["gap"],
                        "ratio": record["ratio"],
                        "run": run_dir.name,
                    }
                )
    chosen: dict[str, dict | None] = {}
    for name, hits in found.items():
        unique = _unique_hits(hits)
        if not unique:
            chosen[name] = None
            continue
        best = unique[0]
        best["tied"] = len(unique)
        best["others"] = ";".join(hit["instance"] for hit in unique[1:])
        chosen[name] = best
    return chosen


def _close(left: float, right: float) -> bool:
    """True when two ratios are the same up to summary-file rounding."""
    return abs(left - right) <= 1e-8 * max(1.0, abs(right))


def _unique_hits(hits: list[dict]) -> list[dict]:
    """One entry per instance, largest gap first."""
    hits.sort(key=lambda hit: (float(hit["gap"]), hit["instance"]), reverse=True)
    seen: set[tuple[str, str]] = set()
    unique = []
    for hit in hits:
        key = (hit["run"], hit["instance"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def _write_algorithms(
    path: Path,
    order: list[str],
    pooled: dict[str, _Totals],
    worst: dict[str, dict | None],
) -> None:
    """Write the per-algorithm summary."""
    fields = [
        "algorithm",
        "runs",
        "instances",
        "mean_ratio",
        "above_optimum",
        "mean_gap",
        "worst_ratio",
        "worst_gap",
        "worst_instance",
        "worst_makespan",
        "worst_optimum",
        "worst_instance_gap",
        "instances_at_worst_ratio",
        "other_instances",
        "run",
    ]
    rows = []
    for name in order:
        rec = pooled[name]
        hit = worst[name] or {}
        rows.append(
            {
                "algorithm": name,
                "runs": rec.runs,
                "instances": rec.instances,
                "mean_ratio": f"{rec.mean_ratio:.6f}",
                "above_optimum": f"{(rec.mean_ratio - 1) * 100:.2f}%",
                "mean_gap": f"{rec.mean_gap:.4f}",
                "worst_ratio": f"{rec.worst_ratio:.6g}",
                "worst_gap": f"{rec.worst_gap:.6g}",
                "worst_instance": hit.get("instance", ""),
                "worst_makespan": hit.get("makespan", ""),
                "worst_optimum": hit.get("optimum", ""),
                "worst_instance_gap": hit.get("gap", ""),
                "instances_at_worst_ratio": hit.get("tied", ""),
                "other_instances": hit.get("others", ""),
                "run": hit.get("run", ""),
            }
        )
    _write_csv(path, fields, rows)


def _write_breakdown(
    path: Path,
    groups: dict[int, dict[str, _Totals]],
    order: list[str],
    key_name: str,
) -> None:
    """Write mean ratio for each edge count or each node count."""
    fields = [key_name, "runs", "instances", *order]
    rows = []
    for key in sorted(groups):
        algorithms = groups[key]
        present = [algorithms[name] for name in order if name in algorithms]
        if not present:
            continue
        row = {
            key_name: key,
            "runs": max(rec.runs for rec in present),
            "instances": max(rec.instances for rec in present),
        }
        for name in order:
            rec = algorithms.get(name)
            row[name] = f"{rec.mean_ratio:.6f}" if rec is not None else ""
        rows.append(row)
    _write_csv(path, fields, rows)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write ``rows`` with ``fields`` as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
