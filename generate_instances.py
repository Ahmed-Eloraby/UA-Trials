"""Write random orientation instances in the Graph Instance Builder text format.

Edit the variables in ``main`` and run this file from the IDE. Each size is a
``(nodes, edges)`` pair. The script writes ``instances_per_size`` graphs for
that pair. Consecutive instances share one weight distribution, and the
distribution changes after every such group. Both directions of an edge are
positive integers drawn independently from that distribution.

Graphs are simple when the requested number of edges fits in the complete
graph on those nodes. A larger request is a loopless multigraph, tagged
``parallel-edges``. Node ids are ``0 .. n-1``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Distribution:
    """A positive-integer weight law.

    Args:
        name: Label stored on each instance that uses this law.
        sample: Draws one integer that is at least 1.

    Returns:
        Distribution: Named sampler.
    """

    name: str
    sample: Callable[[random.Random], int]


def uniform(low: int, high: int) -> Distribution:
    """Uniform on the integers ``low`` through ``high``, inclusive.

    Args:
        low: Smallest weight. Must be at least 1.
        high: Largest weight. Must be at least ``low``.

    Returns:
        Distribution: Uniform sampler.
    """
    if low < 1 or high < low:
        raise ValueError(
            f"uniform bounds must satisfy 1 <= low <= high, got {low}, {high}"
        )

    def sample(rng: random.Random) -> int:
        return rng.randint(low, high)

    return Distribution(f"uniform-{low}-{high}", sample)


def geometric(success: float) -> Distribution:
    """Trials until the first success. The result is at least 1.

    Args:
        success: Success probability, strictly between 0 and 1.

    Returns:
        Distribution: Geometric sampler.
    """
    if not 0.0 < success < 1.0:
        raise ValueError(
            f"geometric success probability must be in (0, 1), got {success}"
        )
    log_complement = math.log(1.0 - success)

    def sample(rng: random.Random) -> int:
        unit = rng.random()
        while unit == 0.0:
            unit = rng.random()
        return math.floor(math.log(unit) / log_complement) + 1

    return Distribution(f"geometric-{success:g}", sample)


def poisson(mean: float) -> Distribution:
    """Poisson with this mean, redrawn while the draw is 0.

    Args:
        mean: Positive Poisson mean.

    Returns:
        Distribution: Zero-truncated Poisson sampler.
    """
    if mean <= 0.0:
        raise ValueError(f"poisson mean must be positive, got {mean}")
    limit = math.exp(-mean)

    def sample(rng: random.Random) -> int:
        while True:
            count = 0
            product = 1.0
            while product > limit:
                count += 1
                product *= rng.random()
            value = count - 1
            if value >= 1:
                return value

    return Distribution(f"poisson-{mean:g}", sample)


def exponential(mean: float) -> Distribution:
    """Ceiling of an exponential draw. The result is at least 1.

    Args:
        mean: Positive mean of the continuous exponential.

    Returns:
        Distribution: Discretized exponential sampler.
    """
    if mean <= 0.0:
        raise ValueError(f"exponential mean must be positive, got {mean}")

    def sample(rng: random.Random) -> int:
        unit = rng.random()
        while unit == 0.0:
            unit = rng.random()
        return max(1, math.ceil(-mean * math.log(unit)))

    return Distribution(f"exponential-{mean:g}", sample)


def binomial(trials: int, success: float) -> Distribution:
    """Sum of ``trials`` coin flips, redrawn while the sum is 0.

    Args:
        trials: Number of flips. Must be at least 1.
        success: Flip probability, strictly between 0 and 1.

    Returns:
        Distribution: Zero-truncated binomial sampler.
    """
    if trials < 1 or not 0.0 < success < 1.0:
        raise ValueError(
            f"binomial needs trials >= 1 and success in (0, 1), got {trials}, {success}"
        )

    def sample(rng: random.Random) -> int:
        while True:
            value = sum(rng.random() < success for _ in range(trials))
            if value >= 1:
                return value

    return Distribution(f"binomial-{trials}-{success:g}", sample)


def zipf(exponent: float, largest: int) -> Distribution:
    """Zipf on ``1 .. largest``. Mass at ``k`` is proportional to ``k ** -exponent``.

    Args:
        exponent: Positive decay. Larger values put more mass on small weights.
        largest: Largest weight in the support. Must be at least 1.

    Returns:
        Distribution: Truncated Zipf sampler.
    """
    if exponent <= 0.0 or largest < 1:
        raise ValueError(
            f"zipf needs a positive exponent and largest >= 1, got {exponent}, {largest}"
        )
    total = 0.0
    cumulative: list[float] = []
    for value in range(1, largest + 1):
        total += value ** (-exponent)
        cumulative.append(total)
    scale = cumulative[-1]

    def sample(rng: random.Random) -> int:
        unit = rng.random() * scale
        lo = 0
        hi = largest - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if unit <= cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo + 1

    return Distribution(f"zipf-{exponent:g}-{largest}", sample)


def rounded_normal(center: float, spread: float) -> Distribution:
    """Nearest integer to a normal draw, redrawn while that integer is below 1.

    Args:
        center: Mean of the normal.
        spread: Positive standard deviation.

    Returns:
        Distribution: Rounded normal sampler.
    """
    if spread <= 0.0:
        raise ValueError(f"normal spread must be positive, got {spread}")

    def sample(rng: random.Random) -> int:
        while True:
            value = int(round(rng.gauss(center, spread)))
            if value >= 1:
                return value

    return Distribution(f"normal-{center:g}-{spread:g}", sample)



def clipped_lognormal_rate(
    median_mbps: float, log_spread: float, maximum_mbps: int
) -> Distribution:
    """Sample a bounded, heterogeneous UE rate requirement in Mbps.

    The median controls the typical requested rate. ``log_spread`` controls
    user-to-user variability, and ``maximum_mbps`` prevents extreme outliers.
    The result is rounded to a positive integer Mbps.
    """
    if (
        not math.isfinite(median_mbps)
        or not math.isfinite(log_spread)
        or median_mbps <= 0.0
        or log_spread <= 0.0
        or maximum_mbps < 1
    ):
        raise ValueError(
            "clipped_lognormal_rate needs finite median > 0, log_spread > 0, "
            "and maximum_mbps >= 1"
        )

    def sample(rng: random.Random) -> int:
        value = int(round(rng.lognormvariate(math.log(median_mbps), log_spread)))
        return min(maximum_mbps, max(1, value))

    return Distribution(
        f"ue-rate-median-{median_mbps:g}-spread-{log_spread:g}-max-{maximum_mbps}",
        sample,
    )


def mixture(
    name: str, components: list[tuple[float, Distribution]]
) -> Distribution:
    """Choose a component distribution according to its relative weight.

    This is useful for representing a population containing several UE
    traffic classes or several radio conditions. Component weights need not
    sum to one.
    """
    if not name or not components:
        raise ValueError("mixture needs a non-empty name and components")
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight, _ in components
    ):
        raise ValueError("mixture weights must be finite and non-negative")
    total = sum(weight for weight, _ in components)
    if total <= 0.0:
        raise ValueError("mixture must have a positive total weight")

    def sample(rng: random.Random) -> int:
        target = rng.random() * total
        cumulative = 0.0
        for weight, distribution in components:
            cumulative += weight
            if target < cumulative:
                return distribution.sample(rng)
        return components[-1][1].sample(rng)

    return Distribution(name, sample)


def shannon_rate(
    bandwidth_mhz: float,
    median_sinr_db: float,
    shadowing_std_db: float,
    efficiency: float = 0.75,
    max_spectral_efficiency: float = 6.0,
) -> Distribution:
    """Approximate an achievable UE-to-BS link rate using Shannon's formula.

    SINR is sampled in dB from a normal distribution to represent variation
    across users and radio conditions. The returned integer is in Mbps, with
    an efficiency factor for protocol and implementation losses and a cap on
    spectral efficiency. This models per-link rate, not aggregate BS capacity.
    """
    values = (bandwidth_mhz, median_sinr_db, shadowing_std_db, efficiency, max_spectral_efficiency)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("shannon_rate parameters must be finite")
    if bandwidth_mhz <= 0.0 or shadowing_std_db < 0.0:
        raise ValueError("bandwidth must be positive and shadowing_std_db non-negative")
    if not 0.0 < efficiency <= 1.0 or max_spectral_efficiency <= 0.0:
        raise ValueError("efficiency must be in (0, 1] and spectral-efficiency cap > 0")

    def sample(rng: random.Random) -> int:
        sinr_db = rng.gauss(median_sinr_db, shadowing_std_db)
        sinr_db = min(40.0, max(-30.0, sinr_db))
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        spectral_efficiency = min(
            math.log2(1.0 + sinr_linear), max_spectral_efficiency
        )
        rate_mbps = bandwidth_mhz * spectral_efficiency * efficiency
        return max(1, int(round(rate_mbps)))

    return Distribution(
        f"shannon-{bandwidth_mhz:g}MHz-sinr-{median_sinr_db:g}dB-std-{shadowing_std_db:g}",
        sample,
    )


def negative_binomial(successes: int, success: float) -> Distribution:
    """Failures before ``successes`` successes, shifted to start at 1.

    The returned weight is ``1 + failures``. Larger ``successes`` or smaller
    ``success`` values generally produce larger weights.
    """
    if successes < 1 or not 0.0 < success < 1.0:
        raise ValueError(
            "negative_binomial needs successes >= 1 and success in (0, 1)"
        )
    log_complement = math.log(1.0 - success)

    def sample(rng: random.Random) -> int:
        failures = 0
        for _ in range(successes):
            unit = rng.random()
            while unit == 0.0:
                unit = rng.random()
            failures += math.floor(math.log(unit) / log_complement)
        return failures + 1

    return Distribution(f"negative-binomial-{successes}-{success:g}", sample)


def lognormal(log_mean: float, log_spread: float) -> Distribution:
    """Ceiling of a lognormal draw, producing a positive integer.

    ``log_mean`` and ``log_spread`` are the mean and standard deviation of the
    underlying normal distribution, not of the resulting weights.
    """
    if log_spread <= 0.0:
        raise ValueError("lognormal log_spread must be positive")

    def sample(rng: random.Random) -> int:
        return max(1, math.ceil(rng.lognormvariate(log_mean, log_spread)))

    return Distribution(f"lognormal-{log_mean:g}-{log_spread:g}", sample)


def gamma(shape: float, scale: float) -> Distribution:
    """Ceiling of a Gamma(shape, scale) draw."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("gamma shape and scale must be positive")

    def sample(rng: random.Random) -> int:
        return max(1, math.ceil(rng.gammavariate(shape, scale)))

    return Distribution(f"gamma-{shape:g}-{scale:g}", sample)


def weibull(shape: float, scale: float) -> Distribution:
    """Ceiling of a Weibull draw, with positive shape and scale parameters."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("weibull shape and scale must be positive")

    def sample(rng: random.Random) -> int:
        unit = rng.random()
        while unit == 0.0:
            unit = rng.random()
        value = scale * (-math.log(unit)) ** (1.0 / shape)
        return max(1, math.ceil(value))

    return Distribution(f"weibull-{shape:g}-{scale:g}", sample)


def pareto(shape: float, minimum: float = 1.0) -> Distribution:
    """Ceiling of a Pareto-I draw with the given minimum and shape."""
    if shape <= 0.0 or minimum < 1.0:
        raise ValueError("pareto shape must be positive and minimum must be >= 1")

    def sample(rng: random.Random) -> int:
        return max(1, math.ceil(minimum * rng.paretovariate(shape)))

    return Distribution(f"pareto-{shape:g}-{minimum:g}", sample)


def triangular(low: int, mode: float, high: int) -> Distribution:
    """Rounded triangular distribution over integer weights in ``[low, high]``."""
    if low < 1 or high < low or not low <= mode <= high:
        raise ValueError("triangular needs 1 <= low <= mode <= high")

    def sample(rng: random.Random) -> int:
        value = int(round(rng.triangular(low - 0.5, high + 0.5, mode)))
        return min(high, max(low, value))

    return Distribution(f"triangular-{low}-{mode:g}-{high}", sample)


def beta_binomial(trials: int, alpha: float, beta: float) -> Distribution:
    """Binomial draw with a fresh Beta(alpha, beta) probability per sample.

    Zero outcomes are redrawn so that every generated weight is positive.
    This produces a bounded, overdispersed alternative to a standard binomial.
    """
    if trials < 1 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta_binomial needs trials >= 1 and alpha, beta > 0")

    def sample(rng: random.Random) -> int:
        while True:
            probability = rng.betavariate(alpha, beta)
            value = sum(rng.random() < probability for _ in range(trials))
            if value >= 1:
                return value

    return Distribution(f"beta-binomial-{trials}-{alpha:g}-{beta:g}", sample)

def two_point(
    spike: int, spike_probability: float, low: int, high: int
) -> Distribution:
    """``spike`` with the given probability, otherwise uniform on ``low .. high``.

    Args:
        spike: Weight used for the point mass. Must be at least 1.
        spike_probability: Probability of drawing ``spike``, in ``[0, 1]``.
        low: Smallest weight of the uniform alternative. Must be at least 1.
        high: Largest weight of the uniform alternative.

    Returns:
        Distribution: Two-point sampler.
    """
    if spike < 1 or low < 1 or high < low or not 0.0 <= spike_probability <= 1.0:
        raise ValueError(
            "two-point needs spike >= 1, 1 <= low <= high, and a probability in [0, 1]"
        )

    def sample(rng: random.Random) -> int:
        if rng.random() < spike_probability:
            return spike
        return rng.randint(low, high)

    return Distribution(
        f"two-point-{spike}-{spike_probability:g}-else-{low}-{high}",
        sample,
    )


def generate_instances(
    sizes: list[tuple[int, int]],
    instances_per_size: int,
    instances_per_distribution: int,
    seed: int,
    output_dir: Path,
    distributions: list[Distribution],
) -> list[Path]:
    """Write one text file per ``(nodes, edges)`` pair.

    Instance ``index`` inside a size uses
    ``distributions[(index // instances_per_distribution) % len(distributions)]``.
    With ``instances_per_distribution`` equal to 2, each couple shares a law and
    the next couple takes the next law.

    Args:
        sizes: Each pair is ``(number of nodes, number of edges)``.
        instances_per_size: How many graphs to write for each pair.
        instances_per_distribution: How many consecutive graphs share one law.
        seed: Master seed. Each graph has its own stream derived from this seed,
            its size, and its index, so editing one pair does not reshuffle the others.
        output_dir: Directory created if needed. One file is ``n{n}_m{m}.txt``.
        distributions: Weight laws, used in order and then cycled.

    Returns:
        list[Path]: Files written, in the same order as ``sizes``.
    """
    _validate_request(
        sizes,
        instances_per_size,
        instances_per_distribution,
        distributions,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for nodes, edges in sizes:
        blocks = [
            _instance_block(
                nodes,
                edges,
                index,
                index // instances_per_distribution,
                _distribution_at(index, instances_per_distribution, distributions),
                random.Random(f"{seed}|{nodes}|{edges}|{index}"),
            )
            for index in range(instances_per_size)
        ]
        path = output_dir / f"n{nodes}_m{edges}.txt"
        path.write_text(_file_text(blocks, seed), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    """Edit these variables, then run this file."""
    sizes = [(i, j) for i in range(4, 6) for j in range(i, 24)]
    instances_per_size = 10000
    instances_per_distribution = 50
    seed = 1
    output_dir = Path(__file__).resolve().parent / "generated"
    distributions = [
        # General-purpose weight distributions retained from the original set.
        uniform(1, 10),
        uniform(1, 100),
        uniform(50, 60),
        geometric(0.3),
        geometric(0.05),
        poisson(4),
        poisson(15),
        exponential(8),
        binomial(20, 0.4),
        zipf(1.5, 200),
        rounded_normal(30, 8),
        two_point(1, 0.8, 40, 120),
        negative_binomial(4, 0.35),
        lognormal(3.0, 0.6),
        gamma(2.5, 10),
        weibull(1.5, 20),
        pareto(2.5, minimum=5),
        triangular(1, 25, 100),
        beta_binomial(40, 2, 5),

        # Synthetic UE requested-rate profiles, in integer Mbps.
        # The profiles represent low, moderate, high, and very high rate needs.
        clipped_lognormal_rate(2, 0.40, 8),
        clipped_lognormal_rate(5, 0.50, 20),
        clipped_lognormal_rate(15, 0.55, 60),
        clipped_lognormal_rate(40, 0.60, 150),
        mixture(
            "ue-demand-mix",
            [
                (0.20, clipped_lognormal_rate(2, 0.40, 8)),
                (0.40, clipped_lognormal_rate(5, 0.50, 20)),
                (0.30, clipped_lognormal_rate(15, 0.55, 60)),
                (0.10, clipped_lognormal_rate(40, 0.60, 150)),
            ],
        ),

        # Per-UE achievable link rates from different BS bandwidth/SINR settings.
        # These are link rates, not total cell capacity.
        shannon_rate(10, 0, 6),
        shannon_rate(20, 5, 6),
        shannon_rate(20, 12, 4),
        shannon_rate(40, 8, 6),
        shannon_rate(100, 10, 6),
        shannon_rate(100, 18, 4),
        mixture(
            "mixed-bs-link-rates",
            [
                (0.25, shannon_rate(10, 0, 6)),
                (0.45, shannon_rate(20, 8, 6)),
                (0.20, shannon_rate(40, 12, 5)),
                (0.10, shannon_rate(100, 16, 4)),
            ],
        ),
    ]

    written = generate_instances(
        sizes,
        instances_per_size,
        instances_per_distribution,
        seed,
        output_dir,
        distributions,
    )
    print(
        f"{instances_per_size} instances for each of {len(sizes)} sizes, "
        f"{instances_per_distribution} instances per distribution"
    )
    for path in written:
        print(path)


def _validate_request(
    sizes: list[tuple[int, int]],
    instances_per_size: int,
    instances_per_distribution: int,
    distributions: list[Distribution],
) -> None:
    """Reject a configuration that cannot produce the requested files."""
    if not sizes:
        raise ValueError("sizes is empty")
    if not distributions:
        raise ValueError("distributions is empty")
    if instances_per_size < 1:
        raise ValueError("instances_per_size must be at least 1")
    if instances_per_distribution < 1:
        raise ValueError("instances_per_distribution must be at least 1")
    seen: set[tuple[int, int]] = set()
    for nodes, edges in sizes:
        if not isinstance(nodes, int) or not isinstance(edges, int):
            raise ValueError(f"size {(nodes, edges)} must be a pair of integers")
        if isinstance(nodes, bool) or isinstance(edges, bool):
            raise ValueError(f"size {(nodes, edges)} must be a pair of integers")
        if nodes < 1 or edges < 0:
            raise ValueError(f"size {(nodes, edges)} needs n >= 1 and m >= 0")
        if edges > 0 and nodes < 2:
            raise ValueError(
                f"size {(nodes, edges)} needs at least 2 nodes to hold an edge"
            )
        if (nodes, edges) in seen:
            raise ValueError(f"duplicate size {(nodes, edges)}")
        seen.add((nodes, edges))


def _distribution_at(
    index: int,
    instances_per_distribution: int,
    distributions: list[Distribution],
) -> Distribution:
    """Law shared by the group that contains this 0-based instance index."""
    group = index // instances_per_distribution
    return distributions[group % len(distributions)]


def _instance_block(
    nodes: int,
    edge_count: int,
    index: int,
    group: int,
    distribution: Distribution,
    rng: random.Random,
) -> str:
    """One instance: name, tags, ``n m``, and ``m`` edge lines.

    Args:
        nodes: Node count ``n``.
        edge_count: Edge count ``m``.
        index: 0-based index of this graph inside its size.
        group: Index of the shared-distribution group. A couple is group mates.
        distribution: Law used for both directions of every edge.
        rng: Stream for this graph only.
    """
    endpoints = _sample_endpoints(nodes, edge_count, rng)
    parallel = edge_count > nodes * (nodes - 1) // 2
    name = f"n{nodes}-m{edge_count}-{index:03d}"
    lines = [
        f"# instance: {name}",
        "# tags: " + ", ".join(_tags(distribution.name, group, parallel)),
        f"{nodes} {edge_count}",
    ]
    for left, right in endpoints:
        weight_left = _draw(distribution, rng)
        weight_right = _draw(distribution, rng)
        lines.append(f"{left} {right} {weight_left} {weight_right}")
    return "\n".join(lines)


def _tags(distribution_name: str, group: int, parallel: bool) -> list[str]:
    """Tags recorded on one instance.

    Instances in one group carry the same ``dist`` and the same ``group``.
    With two instances per distribution, group ``k`` is the ``k``-th couple.

    Args:
        distribution_name: Label of the weight law.
        group: Shared-distribution group index.
        parallel: True when the graph had to use parallel edges.

    Returns:
        list[str]: Tag list, comma-joined by the caller.
    """
    tags = ["random", f"dist={distribution_name}", f"group={group}"]
    if parallel:
        tags.append("parallel-edges")
    return tags


def _sample_endpoints(
    nodes: int, edge_count: int, rng: random.Random
) -> list[tuple[int, int]]:
    """``edge_count`` loopless endpoint pairs. Simple when that fits, otherwise parallel."""
    capacity = nodes * (nodes - 1) // 2
    if edge_count <= capacity:
        pairs = _sample_simple(nodes, edge_count, capacity, rng)
    else:
        pairs = _sample_parallel(nodes, edge_count, rng)
    edges: list[tuple[int, int]] = []
    for left, right in pairs:
        if rng.randrange(2) == 1:
            left, right = right, left
        edges.append((left, right))
    return edges


def _sample_simple(
    nodes: int, edge_count: int, capacity: int, rng: random.Random
) -> list[tuple[int, int]]:
    """``edge_count`` distinct unordered pairs on ``nodes`` nodes."""
    if capacity <= 1_000_000:
        universe = [
            (left, right) for left in range(nodes) for right in range(left + 1, nodes)
        ]
        return rng.sample(universe, edge_count)
    if edge_count <= capacity / 2:
        chosen: set[tuple[int, int]] = set()
        while len(chosen) < edge_count:
            chosen.add(_random_pair(nodes, rng))
        return list(chosen)
    excluded: set[tuple[int, int]] = set()
    while len(excluded) < capacity - edge_count:
        excluded.add(_random_pair(nodes, rng))
    return [
        (left, right)
        for left in range(nodes)
        for right in range(left + 1, nodes)
        if (left, right) not in excluded
    ]


def _sample_parallel(
    nodes: int, edge_count: int, rng: random.Random
) -> list[tuple[int, int]]:
    """``edge_count`` unordered pairs, repeats allowed, loops excluded."""
    return [_random_pair(nodes, rng) for _ in range(edge_count)]


def _random_pair(nodes: int, rng: random.Random) -> tuple[int, int]:
    """One unordered pair of distinct nodes, ordered so the first id is smaller."""
    left = rng.randrange(nodes)
    right = rng.randrange(nodes - 1)
    if right >= left:
        right += 1
    if left > right:
        left, right = right, left
    return left, right


def _draw(distribution: Distribution, rng: random.Random) -> int:
    """One weight, checked to be an integer of at least 1."""
    value = distribution.sample(rng)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"distribution {distribution.name!r} returned {value!r}; expected an integer >= 1"
        )
    return value


def _file_text(blocks: list[str], seed: int) -> str:
    """Builder header plus the instance blocks."""
    header = "\n".join(
        [
            f"# graph-instances v1 count={len(blocks)} index-base=0",
            '# each instance: a name line and a tags line (both comments), then "n m", then m lines "u v wu wv"',
            "# wu = weight if the edge is oriented toward u, wv = weight if it is oriented toward v",
            f"# generated by generate_instances.py seed={seed}",
            "# weights are positive integers; consecutive instances share a distribution for instances_per_distribution graphs",
            "",
        ]
    )
    return header + "\n\n".join(blocks) + "\n"


if __name__ == "__main__":
    main()
