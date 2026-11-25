"""Read and write Graph Instance Builder exports.

Text, CSV, and JSON follow the formats written by ``Graph Instance Builder.html``.
Node ids are preserved. Edge ids are ``1..m`` in file order.
"""

from __future__ import annotations

import csv
import io
import json
import re
from math import inf, isfinite
from pathlib import Path

from makespan.model import Edge, Instance, Solution, as_number


_HEADER_BASE = re.compile(
    r"#\s*graph-instances\b.*?index-base=([01])",
    re.IGNORECASE,
)


def parse(source: str | Path) -> list[Instance]:
    """Parse a builder export from a path or from the export text itself.

    Args:
        source: Path to a builder export, or the export text itself.

    Returns:
        list[Instance]: Parsed instances in file order.
    """
    if isinstance(source, Path) or _is_file_path(source):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = str(source)
    if text.startswith("\ufeff"):
        text = text[1:]
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("empty instance")
    if stripped[0] == "{":
        return parse_json(text)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first and not first.startswith("#") and "," in first:
        return parse_csv(text)
    return parse_text(text)


def parse_text(text: str) -> list[Instance]:
    """Parse one or more ``n m`` blocks, including comment headers.

    Args:
        text: Builder text export containing one or more instances.

    Returns:
        list[Instance]: Parsed instances in file order.
    """
    lines = text.splitlines()
    header_base = _header_base(lines)
    instances: list[Instance] = []
    name = ""
    tags: tuple[str, ...] = ()
    index = 0
    while index < len(lines):
        raw = lines[index].strip()
        index += 1
        if not raw:
            continue
        if raw.startswith("#"):
            name, tags = _consume_comment(raw, name, tags)
            continue
        parts = raw.split()
        if len(parts) != 2:
            raise ValueError(f"expected 'n m' but found {raw!r}")
        node_count = _parse_count(parts[0], "n")
        edge_count = _parse_count(parts[1], "m")
        rows: list[tuple[int, int, float, float]] = []
        while len(rows) < edge_count:
            if index >= len(lines):
                label = name or f"instance-{len(instances) + 1}"
                raise ValueError(
                    f"instance {label!r}: expected {edge_count} edge lines, "
                    f"found {len(rows)}"
                )
            edge_line = lines[index].strip()
            index += 1
            if not edge_line or edge_line.startswith("#"):
                if edge_line.startswith("#"):
                    name, tags = _consume_comment(edge_line, name, tags)
                continue
            rows.append(_parse_edge_row(edge_line, len(rows) + 1))
        endpoints = [node for row in rows for node in row[:2]]
        base = header_base if header_base is not None else detect_base(node_count, endpoints)
        label = name or f"instance-{len(instances) + 1}"
        instances.append(
            build_instance(label, tags, node_count, base, rows)
        )
        name = ""
        tags = ()
    if not instances:
        raise ValueError(
            "no instances found; expected 'n m' followed by m lines 'u v wu wv'"
        )
    return instances


def parse_csv(text: str) -> list[Instance]:
    """Parse the builder CSV. One logical instance may span many rows.

    Args:
        text: Builder CSV export.

    Returns:
        list[Instance]: Parsed instances in CSV order.
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("the CSV is empty")
    fields = {
        (name or "").strip().lower(): name
        for name in reader.fieldnames
        if name is not None
    }
    for required in ("u", "v", "wu", "wv"):
        if required not in fields:
            raise ValueError("the CSV header needs the columns u, v, wu and wv")

    groups: dict[str, dict] = {}
    order: list[str] = []
    for line_number, row in enumerate(reader, start=2):
        key = _cell(row, fields, "instance") or "instance"
        if key not in groups:
            tag_text = _cell(row, fields, "tags")
            tags = tuple(part for part in re.split(r"[;|]", tag_text) if part)
            groups[key] = {"name": key, "tags": tags, "n": None, "edges": []}
            order.append(key)
        group = groups[key]
        if "n" in fields and _cell(row, fields, "n"):
            count = _parse_count(_cell(row, fields, "n"), "n")
            if group["n"] is not None and group["n"] != count:
                raise ValueError(
                    f"row {line_number}: instance {key!r} has two values of n"
                )
            group["n"] = count
        values = [_cell(row, fields, column) for column in ("u", "v", "wu", "wv")]
        if all(value == "" for value in values):
            continue
        if any(value == "" for value in values):
            raise ValueError(
                f"row {line_number}: u, v, wu and wv must all be present"
            )
        try:
            group["edges"].append(_parse_edge_tokens(values, line_number))
        except ValueError as exc:
            raise ValueError(f"row {line_number}: {exc}") from exc

    if not order:
        raise ValueError("the CSV has a header but no rows")
    instances: list[Instance] = []
    for key in order:
        group = groups[key]
        rows = group["edges"]
        endpoints = [node for row in rows for node in row[:2]]
        node_count = group["n"]
        if node_count is None:
            node_count = 0
            for left, right, _, _ in rows:
                node_count = max(node_count, left + 1, right + 1)
        base = detect_base(node_count, endpoints)
        instances.append(
            build_instance(group["name"], group["tags"], node_count, base, rows)
        )
    return instances


def parse_json(text: str) -> list[Instance]:
    """Parse a builder JSON backup.

    Args:
        text: Builder JSON backup text.

    Returns:
        list[Instance]: Parsed instances from the backup.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"this is not valid JSON: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "graph-instance-builder"
        or not isinstance(payload.get("instances"), list)
    ):
        raise ValueError("this JSON is not a Graph Instance Builder backup")
    base = 0
    settings = payload.get("settings")
    if isinstance(settings, dict) and settings.get("base") is not None:
        base = _parse_count(str(settings["base"]), "base")
        if base not in (0, 1):
            raise ValueError("settings.base must be 0 or 1")
    instances: list[Instance] = []
    for position, raw in enumerate(payload["instances"], start=1):
        instances.append(_instance_from_json(raw, base, position))
    if not instances:
        raise ValueError("the backup contains no instances")
    return instances


def build_instance(
    name: str,
    tags: tuple[str, ...],
    node_count: int,
    base: int,
    rows: list[tuple[int, int, float, float]],
) -> Instance:
    """Number nodes ``base .. base + n - 1`` and edges ``1..m``.

    Args:
        name: Label stored on the instance.
        tags: Optional tags copied from the export.
        node_count: Number of nodes ``n``.
        base: First node id, typically ``0`` or ``1``.
        rows: Edge records ``(u, v, w_u, w_v)`` in file order.

    Returns:
        Instance: Graph numbered ``base .. base + n - 1`` with edges ``1..m``.
    """
    if node_count < 0:
        raise ValueError("n must be non-negative")
    if base < 0:
        raise ValueError("index base must be non-negative")
    nodes = tuple(range(base, base + node_count))
    node_set = set(nodes)
    edges: list[Edge] = []
    for edge_id, (left, right, weight_u, weight_v) in enumerate(rows, start=1):
        if left == right:
            raise ValueError(f"edge {edge_id}: u and v are the same node")
        if left not in node_set or right not in node_set:
            raise ValueError(
                f"edge {edge_id}: node ids {left} and {right} do not fit "
                f"n={node_count} numbered from {base}"
            )
        if weight_u < 0 or weight_v < 0:
            raise ValueError(f"edge {edge_id}: weights must be zero or positive")
        edges.append(Edge(edge_id, left, right, float(weight_u), float(weight_v)))
    return Instance(name, tuple(tags), nodes, tuple(edges))


def detect_base(node_count: int, endpoints: list[int]) -> int:
    """Choose 0-based or 1-based numbering when the file has no index header.

    Base 0 is used when a node id ``0`` appears and every id lies in
    ``0 .. n-1``. Base 1 is used when every id lies in ``1 .. n``.

    Args:
        node_count: Declared number of nodes ``n``.
        endpoints: Node ids that appear as edge endpoints.

    Returns:
        int: ``0`` or ``1``, the inferred index base.
    """
    if node_count < 0:
        raise ValueError("n must be non-negative")
    if not endpoints:
        return 0
    has_zero = 0 in endpoints
    zero_based = all(0 <= node < node_count for node in endpoints)
    one_based = all(1 <= node <= node_count for node in endpoints)
    if has_zero and zero_based:
        return 0
    if one_based and not has_zero:
        return 1
    raise ValueError(
        f"node ids do not fit n={node_count} numbered from 0 or from 1"
    )


def format_text(instance: Instance, solution: Solution) -> str:
    """Render one solution as a comment block plus ``id u v head weight`` lines.

    Args:
        instance: Graph whose edges are listed.
        solution: Orientation and metrics to render.

    Returns:
        str: Comment header plus ``id u v head weight`` lines.
    """
    lines = [
        f"# instance: {solution.instance_name}",
        f"# algorithm: {solution.algorithm}",
        f"# makespan: {_format_number(solution.makespan)}",
        f"# lower_bound: {_format_number(solution.lower_bound)}",
        f"# upper_bound: {_format_number(solution.upper_bound)}",
    ]
    for edge in instance.edges:
        head = solution.heads[edge.id]
        lines.append(
            f"{edge.id} {edge.u} {edge.v} {head} "
            f"{_format_number(edge.weight_toward(head))}"
        )
    return "\n".join(lines)


def format_assignment_file(entries: list[tuple[Instance, Solution]]) -> str:
    """Render one algorithm's orientations in an input-like text file.

    The file lists every instance that algorithm solved. Each instance keeps
    its name and tags, then ``n m``, then one ``u v head weight`` line per
    edge, then one ``node load`` line per node. ``head`` is the endpoint the
    edge points to, and ``weight`` is the load that choice adds. Load lines
    include isolated nodes.

    Args:
        entries: Instance and solution pairs, all from the same algorithm,
            in the order they should appear.

    Returns:
        str: A ``graph-assignments`` document ending in a newline.

    Raises:
        ValueError: ``entries`` is empty, or the solutions name more than one algorithm.
    """
    if not entries:
        raise ValueError("an assignment file needs at least one solution")
    algorithm = entries[0][1].algorithm
    for instance, solution in entries:
        if solution.algorithm != algorithm:
            raise ValueError(
                f"assignment file for {algorithm!r} also contains {solution.algorithm!r}"
            )
        if solution.instance_name != instance.name:
            raise ValueError(
                f"solution {solution.instance_name!r} does not match instance {instance.name!r}"
            )
    lines = [
        f"# graph-assignments v1 count={len(entries)} index-base={_index_base(entries)}",
        '# each instance: a name line, a tags line, and an algorithm line,',
        '# then "n m", then m lines "u v head weight", then n lines "node load"',
        "# head = the node the edge is oriented toward",
        "# weight = the load that orientation adds to head",
        "# the load lines list every node, including nodes with no incident edge",
        "# open this file in Assignment Viewer.html",
    ]
    for instance, solution in entries:
        lines.append("")
        lines.extend(_assignment_block(instance, solution))
    return "\n".join(lines) + "\n"


def assignment_filename(algorithm: str) -> str:
    """File name for one algorithm's assignment document.

    Args:
        algorithm: Registered algorithm name.

    Returns:
        str: A ``.txt`` name made of letters, digits, dots, underscores, and dashes.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", algorithm).strip("._")
    if not cleaned:
        cleaned = "algorithm"
    return f"{cleaned}.txt"


def format_json_record(instance: Instance, solution: Solution) -> dict:
    """One JSON object for an orientation, including tags and node loads.

    Args:
        instance: Graph whose edges and tags are listed.
        solution: Orientation and metrics to encode.

    Returns:
        dict: JSON object with instance metrics, one record per edge, and one load per node.
    """
    return {
        "instance": instance.name,
        "tags": list(instance.tags),
        "algorithm": solution.algorithm,
        "makespan": as_number(solution.makespan),
        "lower_bound": as_number(solution.lower_bound),
        "upper_bound": as_number(solution.upper_bound),
        "edges": [
            {
                "id": edge.id,
                "u": edge.u,
                "v": edge.v,
                "head": solution.heads[edge.id],
                "weight": as_number(edge.weight_toward(solution.heads[edge.id])),
            }
            for edge in instance.edges
        ],
        "loads": [
            {"node": node, "load": as_number(solution.loads[node])}
            for node in instance.nodes
        ],
    }


def format_skip(instance: Instance, algorithm: str, reason: str) -> str:
    """Comment block used when an algorithm is skipped instead of solved.

    Args:
        instance: Graph that would have been solved.
        algorithm: Name of the skipped algorithm.
        reason: Human-readable explanation for the skip.

    Returns:
        str: Comment block naming the instance, algorithm, and skip reason.
    """
    return "\n".join(
        [
            f"# instance: {instance.name}",
            f"# algorithm: {algorithm}",
            f"# skipped: {reason}",
        ]
    )


_SOLUTION_CSV_COLUMNS = (
    "instance",
    "tags",
    "algorithm",
    "makespan",
    "lower_bound",
    "upper_bound",
    "edge_id",
    "u",
    "v",
    "head",
    "weight",
    "skipped",
)

_LOADS_CSV_COLUMNS = (
    "instance",
    "tags",
    "algorithm",
    "node",
    "load",
    "makespan",
)


def format_solution_csv(records: list[dict]) -> str:
    """Render solve records as CSV, one row per oriented edge.

    A skipped algorithm is one row with empty edge columns and the reason in
    ``skipped``. Instance metrics are repeated on every edge row so the file
    can be filtered in a spreadsheet.

    Args:
        records: JSON solve objects, one per instance and algorithm.

    Returns:
        str: CSV text, one row per oriented edge (or one skip row).
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=_SOLUTION_CSV_COLUMNS,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for record in records:
        edges = record.get("edges") or []
        if not edges:
            writer.writerow(
                {
                    "instance": record.get("instance", ""),
                    "tags": _tags_cell(record.get("tags")),
                    "algorithm": record.get("algorithm", ""),
                    "makespan": _csv_cell(record.get("makespan")),
                    "lower_bound": _csv_cell(record.get("lower_bound")),
                    "upper_bound": _csv_cell(record.get("upper_bound")),
                    "skipped": record.get("skipped") or "",
                }
            )
            continue
        for edge in edges:
            writer.writerow(
                {
                    "instance": record.get("instance", ""),
                    "tags": _tags_cell(record.get("tags")),
                    "algorithm": record.get("algorithm", ""),
                    "makespan": _csv_cell(record.get("makespan")),
                    "lower_bound": _csv_cell(record.get("lower_bound")),
                    "upper_bound": _csv_cell(record.get("upper_bound")),
                    "edge_id": edge.get("id", ""),
                    "u": edge.get("u", ""),
                    "v": edge.get("v", ""),
                    "head": edge.get("head", ""),
                    "weight": _csv_cell(edge.get("weight")),
                    "skipped": "",
                }
            )
    return buffer.getvalue()


def format_loads_csv(entries: list[tuple[Instance, Solution]]) -> str:
    """Render node loads as CSV, one row per node.

    Args:
        entries: Instance and solution pairs, in the order rows should appear.
            Every node of each instance is written, including isolated nodes.

    Returns:
        str: CSV text with columns ``instance``, ``tags``, ``algorithm``,
        ``node``, ``load``, and ``makespan``.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=_LOADS_CSV_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for instance, solution in entries:
        for node in instance.nodes:
            writer.writerow(
                {
                    "instance": instance.name,
                    "tags": _tags_cell(instance.tags),
                    "algorithm": solution.algorithm,
                    "node": node,
                    "load": _csv_cell(as_number(solution.loads[node])),
                    "makespan": _csv_cell(as_number(solution.makespan)),
                }
            )
    return buffer.getvalue()


_SUMMARY_CSV_COLUMNS = (
    "algorithm",
    "solved",
    "skipped",
    "makespan_min",
    "makespan_avg",
    "makespan_max",
    "lower_bound_min",
    "lower_bound_avg",
    "lower_bound_max",
    "upper_bound_min",
    "upper_bound_avg",
    "upper_bound_max",
    "gap_min",
    "gap_avg",
    "gap_max",
    "ratio_min",
    "ratio_avg",
    "ratio_max",
)

_SUMMARY_METRICS = ("makespan", "lower_bound", "upper_bound", "gap", "ratio")


def format_summary_csv(groups: dict[str, list[dict]]) -> str:
    """Aggregate one row per algorithm across every instance in the run.

    A sample with a ``makespan`` counts as solved. A sample without one counts
    as skipped and adds nothing to the metrics. ``gap`` and ``ratio`` are
    filled only for samples that carry them, which is a comparison run.
    Minimum, average, and maximum use the finite values. Positive infinity
    raises the maximum to ``inf`` and stays out of the average, so one
    unbounded ratio does not hide the finite ones. A metric with no values
    stays blank.

    Args:
        groups: Algorithm name to its per-instance samples, in row order.
            Each sample may contain ``makespan``, ``lower_bound``,
            ``upper_bound``, ``gap``, and ``ratio``.

    Returns:
        str: CSV text, one row per algorithm.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=_SUMMARY_CSV_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for algorithm, samples in groups.items():
        solved = 0
        skipped = 0
        collected: dict[str, list[float]] = {metric: [] for metric in _SUMMARY_METRICS}
        for sample in samples:
            makespan = _summary_metric(sample, "makespan")
            if makespan is None:
                skipped += 1
                continue
            solved += 1
            for metric in _SUMMARY_METRICS:
                value = _summary_metric(sample, metric)
                if value is not None:
                    collected[metric].append(value)
        row = {"algorithm": algorithm, "solved": solved, "skipped": skipped}
        for metric in _SUMMARY_METRICS:
            low, average, high = _aggregate(collected[metric])
            row[f"{metric}_min"] = low
            row[f"{metric}_avg"] = average
            row[f"{metric}_max"] = high
        writer.writerow(row)
    return buffer.getvalue()


def _summary_metric(sample: dict, key: str) -> float | None:
    """Read one numeric sample field, accepting the string ``inf``.

    Args:
        sample: Per-instance metric mapping.
        key: Field name to read.

    Returns:
        float | None: The finite or infinite value, or ``None`` when absent.
    """
    if key not in sample:
        return None
    value = sample[key]
    if value is None or value == "":
        return None
    if value == "inf":
        return inf
    if value == "-inf":
        return -inf
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _aggregate(values: list[float]) -> tuple[str, str, str]:
    """Minimum, average, and maximum of one metric.

    The average uses finite values only. Positive infinity sets the maximum
    to ``inf``. Negative infinity sets the minimum to ``-inf``.

    Args:
        values: Observed metric values, possibly empty or non-finite.

    Returns:
        tuple[str, str, str]: Minimum, average, and maximum cells. Empty when
        ``values`` has no usable number.
    """
    if not values:
        return "", "", ""
    finite = [value for value in values if isfinite(value)]
    positive_inf = any(value == inf for value in values)
    negative_inf = any(value == -inf for value in values)
    if not finite:
        if positive_inf and negative_inf:
            return "-inf", "", "inf"
        if positive_inf:
            return "inf", "inf", "inf"
        if negative_inf:
            return "-inf", "-inf", "-inf"
        return "", "", ""
    low = min(finite)
    high = max(finite)
    average = sum(finite) / len(finite)
    return (
        "-inf" if negative_inf else _summary_cell(low),
        _summary_cell(average),
        "inf" if positive_inf else _summary_cell(high),
    )


def _summary_cell(value: float) -> str:
    """Render one aggregate the same way other numeric CSV cells are rendered.

    Args:
        value: Finite aggregate to render.

    Returns:
        str: Integer text when ``value`` is integral, otherwise compact digits.
    """
    return _format_number(value)


def _assignment_block(instance: Instance, solution: Solution) -> list[str]:
    """Comment header, edge lines, and load lines for one solved instance.

    Args:
        instance: Graph that was solved, including its name and tags.
        solution: Orientation and the load of every node.

    Returns:
        list[str]: Lines for this instance, without a trailing blank line.
    """
    lines = [
        f"# instance: {instance.name}",
        f"# tags: {', '.join(instance.tags)}".rstrip(),
        f"# algorithm: {solution.algorithm}",
        f"# makespan: {_format_number(solution.makespan)}",
        f"# lower_bound: {_format_number(solution.lower_bound)}",
        f"# upper_bound: {_format_number(solution.upper_bound)}",
        f"{len(instance.nodes)} {len(instance.edges)}",
    ]
    for edge in instance.edges:
        head = solution.heads[edge.id]
        lines.append(
            f"{edge.u} {edge.v} {head} {_format_number(edge.weight_toward(head))}"
        )
    lines.append("# loads")
    for node in instance.nodes:
        lines.append(f"{node} {_format_number(solution.loads[node])}")
    return lines


def _index_base(entries: list[tuple[Instance, Solution]]) -> int:
    """Smallest node id of the first instance that has a node.

    Args:
        entries: Instance and solution pairs in file order.

    Returns:
        int: First node id, or ``0`` when every instance is empty.
    """
    for instance, _solution in entries:
        if instance.nodes:
            return min(instance.nodes)
    return 0


def _tags_cell(value: object) -> str:
    """Join tags with ``;``, matching the Graph Instance Builder CSV.

    Args:
        value: A tag string, a sequence of tags, or nothing.

    Returns:
        str: Tags for one CSV cell. Missing tags become an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ";".join(str(part) for part in value if str(part) != "")
    return str(value)


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


def _format_number(value: float) -> str:
    """Render an integer-valued float as an integer, otherwise with compact digits.

    Args:
        value: Number to render.

    Returns:
        str: Integer text when ``value`` is integral, otherwise compact digits.
    """
    number = as_number(value)
    if isinstance(number, int):
        return str(number)
    return format(number, ".10g")


def _is_file_path(source: str | Path) -> bool:
    """True when ``source`` names an existing file rather than export text.

    Args:
        source: Path or export text to classify.

    Returns:
        bool: True when ``source`` names an existing file.
    """
    if isinstance(source, Path):
        return True
    if "\n" in source or "\r" in source:
        return False
    path = Path(source)
    return path.exists() and path.is_file()


def _header_base(lines: list[str]) -> int | None:
    """Read ``index-base`` from the first non-empty line, or return ``None``.

    Args:
        lines: Text-export lines, including comments.

    Returns:
        int | None: ``index-base`` from the first non-empty line, or ``None``.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = _HEADER_BASE.search(stripped)
        if match:
            return int(match.group(1))
        return None
    return None


def _consume_comment(
    raw: str, name: str, tags: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Update the current instance name or tag list from one comment line.

    Args:
        raw: One comment line, including the leading ``#``.
        name: Current instance name, kept when this line is not an instance header.
        tags: Current tag tuple, kept when this line is not a tags header.

    Returns:
        tuple[str, tuple[str, ...]]: Updated ``(name, tags)``.
    """
    body = raw[1:].strip()
    lowered = body.lower()
    if lowered.startswith("instance:"):
        return body.split(":", 1)[1].strip(), tags
    if lowered.startswith("tags:"):
        tag_text = body.split(":", 1)[1]
        parsed = tuple(part.strip() for part in tag_text.split(",") if part.strip())
        return name, parsed
    return name, tags


def _parse_edge_row(line: str, edge_number: int) -> tuple[int, int, float, float]:
    """Parse one ``u v wu wv`` line. ``edge_number`` is used in error messages.

    Args:
        line: Edge line from the text export.
        edge_number: 1-based edge index used in error messages.

    Returns:
        tuple[int, int, float, float]: ``(u, v, w_u, w_v)``.
    """
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(f"expected 'u v wu wv' but found {line!r}")
    try:
        return _parse_edge_tokens(parts, edge_number)
    except ValueError as exc:
        raise ValueError(f"edge {edge_number}: {exc}") from exc


def _parse_edge_tokens(
    parts: list[str], edge_number: int
) -> tuple[int, int, float, float]:
    """Parse four tokens into endpoints and weights. Negative weights are rejected.

    Args:
        parts: Four tokens ``u``, ``v``, ``wu``, ``wv``.
        edge_number: 1-based edge index used in error messages.

    Returns:
        tuple[int, int, float, float]: ``(u, v, w_u, w_v)``.
    """
    try:
        left = _parse_node_id(parts[0])
        right = _parse_node_id(parts[1])
        weight_u = float(parts[2])
        weight_v = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"edge {edge_number} has a non-numeric field") from exc
    if weight_u < 0 or weight_v < 0:
        raise ValueError(f"edge {edge_number}: weights must be zero or positive")
    return left, right, weight_u, weight_v


def _parse_node_id(token: str) -> int:
    """Parse a node id. Integral floats such as ``1.0`` are accepted.

    Args:
        token: Text of the node id.

    Returns:
        int: Parsed node id.
    """
    value = float(token)
    if not value.is_integer():
        raise ValueError(f"{token!r} is not an integer node id")
    return int(value)


def _parse_count(token: str, label: str) -> int:
    """Parse a non-negative integer count. ``label`` names the field in errors.

    Args:
        token: Text of the count.
        label: Field name used in error messages, such as ``n``.

    Returns:
        int: Non-negative integer count.
    """
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _cell(row: dict, fields: dict[str, str], name: str) -> str:
    """Return a stripped CSV cell, or an empty string when the column is absent.

    Args:
        row: One CSV row as a mapping from original header names to cells.
        fields: Map from lowercase column names to the original header names.
        name: Lowercase column name to read.

    Returns:
        str: Stripped cell, or ``""`` when the column is absent.
    """
    if name not in fields:
        return ""
    value = row.get(fields[name])
    if value is None:
        return ""
    return value.strip()


def _instance_from_json(raw: object, base: int, position: int) -> Instance:
    """Turn one backup instance into nodes numbered from ``base`` in array order.

    Args:
        raw: One object from the backup ``instances`` array.
        base: First node id, typically ``0`` or ``1``.
        position: 1-based instance index used in error messages.

    Returns:
        Instance: Graph numbered from ``base`` in array order.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"instance {position} in the backup is malformed")
    nodes_raw = raw.get("nodes")
    edges_raw = raw.get("edges")
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise ValueError(f"instance {position} in the backup is malformed")
    id_map: dict[object, int] = {}
    for offset, node in enumerate(nodes_raw):
        if not isinstance(node, dict) or "id" not in node:
            raise ValueError(f"instance {position} has a malformed node")
        id_map[node["id"]] = base + offset
    rows: list[tuple[int, int, float, float]] = []
    for edge_number, edge in enumerate(edges_raw, start=1):
        if not isinstance(edge, dict):
            raise ValueError(f"instance {position}, edge {edge_number} is malformed")
        if edge.get("a") not in id_map or edge.get("b") not in id_map:
            raise ValueError(
                f"instance {position}, edge {edge_number} points at a missing node"
            )
        try:
            weight_u = float(edge["wa"])
            weight_v = float(edge["wb"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"instance {position}, edge {edge_number} has an invalid weight"
            ) from exc
        rows.append((id_map[edge["a"]], id_map[edge["b"]], weight_u, weight_v))
    name = str(raw.get("name") or f"instance-{position}").strip() or f"instance-{position}"
    tags_raw = raw.get("tags") or []
    if not isinstance(tags_raw, list):
        raise ValueError(f"instance {position} has malformed tags")
    tags = tuple(str(tag).strip() for tag in tags_raw if str(tag).strip())
    try:
        return build_instance(name, tags, len(nodes_raw), base, rows)
    except ValueError as exc:
        raise ValueError(f"instance {position}: {exc}") from exc
