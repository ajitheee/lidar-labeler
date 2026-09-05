"""Turn a 2D box plus projected LiDAR points into one reliable distance label.

This is the part that decides whether your dataset is any good. A 2D box does
not tightly enclose the object, so the points that fall inside it are a mixture
of three things:

  1. the object's own surface  - a dense cluster at roughly one depth
  2. background behind it      - road, buildings, sky-adjacent returns, farther
  3. occluders in front        - poles, mirrors, other traffic, nearer

Taking a naive mean or median of everything inside the box biases the label
toward the background and quietly poisons the training set. The strategy here
is to find the dominant *near* cluster, on the reasoning that an object occludes
whatever is behind it, so its surface is the nearest substantial cluster in the
box - while still requiring that cluster to be substantial, so a single stray
foreground return cannot drag the label toward the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .projection import ProjectedCloud

# Spans a small pedestrian through a lorry. Deliberately generous: this is a
# sanity check on physical possibility, not a per-class size model.
MIN_PLAUSIBLE_HEIGHT = 0.8
MAX_PLAUSIBLE_HEIGHT = 5.0


@dataclass(frozen=True)
class BoxDistance:
    """A distance label with enough context to decide whether to trust it."""

    distance: float          # metres, forward depth; nan when unresolved
    n_points: int            # points inside the shrunk box
    n_cluster: int           # points in the winning cluster
    spread: float            # std dev within the cluster, metres
    inlier_fraction: float   # n_cluster / n_points
    n_beyond: int = 0        # points sitting well behind the winning cluster
    implied_height: float = float("nan")  # object height this distance implies, metres

    @property
    def implausible_size(self) -> bool:
        """Does this distance imply an object of impossible physical size?

        The strongest available check, because it needs no ground truth. A box
        h pixels tall at distance d implies a real height of h*d/f. Measure an
        occluder instead of the car behind it and the implied height collapses -
        a 30 px box read at 14.7 m instead of 44.7 m implies a 0.6 m car.

        This catches what the point-count heuristic cannot: a *dense* occluder.
        Nearer things return more points, so the intruder usually wins on count
        as well as proximity, and only geometry gives it away.
        """
        return np.isfinite(self.implied_height) and not (
            MIN_PLAUSIBLE_HEIGHT <= self.implied_height <= MAX_PLAUSIBLE_HEIGHT
        )

    @property
    def looks_occluded(self) -> bool:
        """A small near cluster with a crowd behind it is probably an occluder.

        This is the pipeline's main failure mode, and it is directional: it
        always reports objects as NEARER than they are, because the thing in
        front is what gets measured. Measured on nuScenes, the bias reached
        -11 m beyond 50 m, where a 32-beam sweep leaves a handful of returns on
        a car and any foreground intruder outvotes them.

        A genuine object in front of a background wall can trip this too. That
        is the acceptable direction to be wrong in - it costs coverage, which
        more frames recover, instead of injecting a confident wrong label.
        """
        return self.n_cluster < 12 and self.n_beyond > 3 * self.n_cluster

    @property
    def is_reliable(self) -> bool:
        """Conservative gate. Unreliable boxes should be dropped, not guessed at."""
        return (
            np.isfinite(self.distance)
            and self.n_cluster >= 8
            and self.inlier_fraction >= 0.35
            and self.spread <= 2.5
            and not self.looks_occluded
            and not self.implausible_size
        )


def shrink_box(
    box: tuple[float, float, float, float], scale: float = 0.75
) -> tuple[float, float, float, float]:
    """Contract a box toward its centre.

    Detector boxes include a margin of background at every edge, and that margin
    is where most of the contamination lives. Shrinking costs a few object points
    and removes a great many background ones, which is a trade worth making.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _nearest_dominant_cluster(
    depths: np.ndarray, band: float, dominance: float = 0.6
) -> np.ndarray:
    """Return the depths belonging to the nearest substantial cluster.

    Slides a window of width `band` over the sorted depths, finds the window
    holding the most points, then accepts the *nearest* window that still holds
    at least `dominance` of that maximum. Preferring near clusters encodes the
    occlusion argument; the dominance floor stops a handful of foreground
    returns from winning.
    """
    if depths.size == 0:
        return depths

    order = np.sort(depths)
    # counts[i] = number of points within [order[i], order[i] + band]
    upper = np.searchsorted(order, order + band, side="right")
    counts = upper - np.arange(order.size)

    threshold = counts.max() * dominance
    first_good = int(np.argmax(counts >= threshold))  # nearest qualifying window
    start = order[first_good]
    return order[(order >= start) & (order <= start + band)]


def estimate_box_distance(
    box: tuple[float, float, float, float],
    cloud: ProjectedCloud,
    shrink: float = 0.75,
    band: float | None = None,
    min_points: int = 5,
    focal_length: float | None = None,
) -> BoxDistance:
    """Estimate the distance to the object in `box`.

    Args:
        box: (x1, y1, x2, y2) in pixels.
        cloud: points already projected into this image.
        shrink: box contraction factor before collecting points.
        band: cluster width in metres. Defaults to an adaptive value, because a
            car at 60 m deserves a wider tolerance than one at 5 m - LiDAR range
            noise and the object's own extent both grow with distance.
        min_points: below this many points in the box, refuse to answer.
        focal_length: pixels. When given, enables the implied-height check,
            which is the only thing that reliably catches a dense occluder.

    Returns:
        BoxDistance; check `.is_reliable` before writing it into a dataset.
    """
    x1, y1, x2, y2 = shrink_box(box, shrink)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    depths = cloud.depth[inside]

    if depths.size < min_points:
        return BoxDistance(float("nan"), int(depths.size), 0, float("nan"), 0.0, 0)

    if band is None:
        # ~8% of the rough object depth, floored at 1 m and capped at 4 m.
        anchor = float(np.percentile(depths, 20))
        band = float(np.clip(0.08 * anchor, 1.0, 4.0))

    cluster = _nearest_dominant_cluster(depths, band)
    if cluster.size == 0:
        return BoxDistance(float("nan"), int(depths.size), 0, float("nan"), 0.0, 0)

    distance = float(np.median(cluster))
    box_height_px = box[3] - box[1]
    implied = (
        box_height_px * distance / focal_length
        if focal_length and focal_length > 0
        else float("nan")
    )

    return BoxDistance(
        distance=distance,
        n_points=int(depths.size),
        n_cluster=int(cluster.size),
        spread=float(np.std(cluster)),
        inlier_fraction=float(cluster.size / depths.size),
        n_beyond=int((depths > cluster.max() + band).sum()),
        implied_height=implied,
    )
