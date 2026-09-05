"""Break measurement error down by the things that plausibly cause it.

An aggregate MAE tells you a pipeline is imperfect but not which objects it
fails on. A mean of 1.8 m sitting next to a p95 of 9.3 m is not uniform noise -
it is a small population of badly wrong labels hiding inside a mostly good one,
and the fix depends entirely on which population that is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISTANCE_EDGES = (0.0, 10.0, 20.0, 30.0, 50.0, float("inf"))


@dataclass(frozen=True)
class Record:
    """One annotation and what the pipeline made of it."""

    truth: float          # ground-truth distance, metres
    estimate: float       # measured distance; nan when no label was produced
    visibility: int       # 1..4
    num_lidar_pts: int    # nuScenes' own point count for the 3D box
    reliable: bool
    brightness: float = float("nan")   # mean pixel intensity inside the box, 0-255

    @property
    def error(self) -> float:
        return self.estimate - self.truth


@dataclass(frozen=True)
class Bucket:
    label: str
    total: int
    produced: int
    mae: float
    median_ae: float
    bias: float
    p95: float

    @property
    def coverage(self) -> float:
        return self.produced / self.total if self.total else 0.0


def summarize(records: list[Record], label: str) -> Bucket:
    """Aggregate one group. Median absolute error sits beside the mean on purpose:
    when the two diverge sharply, the mean is being set by a tail rather than by
    typical behaviour, and quoting only the mean misrepresents the pipeline."""
    kept = np.array([r.error for r in records if r.reliable and np.isfinite(r.estimate)])
    if kept.size == 0:
        return Bucket(label, len(records), 0, float("nan"), float("nan"),
                      float("nan"), float("nan"))
    absolute = np.abs(kept)
    return Bucket(
        label=label,
        total=len(records),
        produced=int(kept.size),
        mae=float(absolute.mean()),
        median_ae=float(np.median(absolute)),
        bias=float(kept.mean()),
        p95=float(np.percentile(absolute, 95)),
    )


def by_distance(records: list[Record], edges=DISTANCE_EDGES) -> list[Bucket]:
    buckets = []
    for low, high in zip(edges, edges[1:]):
        name = f"{low:.0f}-{high:.0f} m" if np.isfinite(high) else f"{low:.0f} m+"
        buckets.append(summarize([r for r in records if low <= r.truth < high], name))
    return buckets


def by_visibility(records: list[Record]) -> list[Bucket]:
    names = {1: "1  (<40% visible)", 2: "2  (40-60%)", 3: "3  (60-80%)", 4: "4  (>80%)"}
    return [
        summarize([r for r in records if r.visibility == level], names[level])
        for level in (1, 2, 3, 4)
    ]


def by_point_count(records: list[Record]) -> list[Bucket]:
    edges = [(0, 10), (10, 30), (30, 100), (100, 10**9)]
    return [
        summarize(
            [r for r in records if low <= r.num_lidar_pts < high],
            f"{low}-{high} pts" if high < 10**9 else f"{low}+ pts",
        )
        for low, high in edges
    ]


def by_brightness(records: list[Record]) -> list[Bucket]:
    """Group by how dark the object appears.

    Tests a specific suspicion: LiDAR runs near 905 nm, where black automotive
    paint reflects poorly, so dark vehicles return fewer points than light ones
    at the same distance. Fewer points means more of them fail the reliability
    gate, which means they are missing from the labels - and a camera model
    trained on those labels inherits a colour bias from a sensor that is not
    even present at inference.

    If coverage falls monotonically as boxes get darker, that is the mechanism.
    """
    usable = [r for r in records if np.isfinite(r.brightness)]
    if not usable:
        return []

    edges = np.percentile([r.brightness for r in usable], [0, 25, 50, 75, 100])
    labels = ["darkest 25%", "25-50%", "50-75%", "lightest 25%"]
    buckets = []
    for i, name in enumerate(labels):
        low, high = edges[i], edges[i + 1]
        selected = [
            r for r in usable
            if low <= r.brightness < high or (i == 3 and r.brightness == high)
        ]
        buckets.append(summarize(selected, f"{name} ({low:.0f}-{high:.0f})"))
    return buckets


def render(buckets: list[Bucket], heading: str) -> str:
    lines = [
        "",
        heading,
        f"{'group':<20}{'n':>6}{'kept':>7}{'cov':>7}{'MAE':>8}{'median':>8}{'bias':>8}{'p95':>8}",
        "-" * 72,
    ]
    for b in buckets:
        if b.total == 0:
            continue
        if b.produced == 0:
            lines.append(f"{b.label:<20}{b.total:>6}{0:>7}{'0%':>7}{'-':>8}{'-':>8}{'-':>8}{'-':>8}")
            continue
        lines.append(
            f"{b.label:<20}{b.total:>6}{b.produced:>7}{b.coverage:>6.0%}"
            f"{b.mae:>8.2f}{b.median_ae:>8.2f}{b.bias:>+8.2f}{b.p95:>8.2f}"
        )
    return "\n".join(lines)
