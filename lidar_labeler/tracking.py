"""Time-to-collision from a sequence of distance estimates.

Distance alone does not tell a driver anything actionable - a car 30 m ahead
travelling at your speed is safe indefinitely, and one 30 m ahead that has
stopped is roughly two seconds away. TTC is the number that matters, and it
needs the distance to change over time, which means tracking.

Closing speed comes from a least-squares slope over a short window rather than
differencing consecutive frames. Per-frame distance carries about half a metre
of noise; differencing two noisy samples over a 30 ms gap produces a speed
estimate that swings wildly and an alert that flickers.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

# Below this the object is not meaningfully approaching, and dividing by a
# near-zero closing speed yields a spuriously dramatic countdown.
MIN_CLOSING_SPEED = 0.5   # m/s


class TimeToCollision:
    """Per-track distance history, turned into a closing speed and a TTC."""

    def __init__(self, window: int = 8, min_samples: int = 4):
        if min_samples < 2:
            raise ValueError("need at least two samples to estimate a slope")
        self.window = window
        self.min_samples = min_samples
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, distance: float, timestamp: float) -> float | None:
        """Record an observation and return TTC in seconds, or None.

        None means "no meaningful answer": too few samples yet, or the object is
        stationary or receding. Returning None rather than a large number keeps
        the caller from rendering a countdown for a car that is pulling away.
        """
        if not np.isfinite(distance):
            return None

        history = self._history[track_id]
        history.append((timestamp, distance))
        if len(history) < self.min_samples:
            return None

        times = np.array([t for t, _ in history])
        distances = np.array([d for _, d in history])
        if np.ptp(times) <= 0:
            return None

        slope = float(np.polyfit(times, distances, 1)[0])   # metres per second
        closing = -slope
        if closing < MIN_CLOSING_SPEED:
            return None

        return float(distances[-1] / closing)

    def forget(self, active_ids: set[int]) -> None:
        """Drop tracks that have left the scene, so memory does not grow."""
        for track_id in list(self._history):
            if track_id not in active_ids:
                del self._history[track_id]


def severity(ttc: float | None, warn: float = 4.0, critical: float = 2.0) -> str:
    """Bucket a TTC for display. Thresholds are roughly human reaction time."""
    if ttc is None:
        return "none"
    if ttc <= critical:
        return "critical"
    if ttc <= warn:
        return "warn"
    return "ok"


def in_ego_path(
    box: tuple[float, float, float, float],
    distance: float,
    image_width: int,
    focal_length: float = 1250.0,
    lane_half_width: float = 1.9,
) -> bool:
    """Is this object inside the corridor the vehicle is about to drive through?

    Distance alone makes a bad alert. Driving past a parked car closes the gap
    to it every frame, so an unfiltered TTC fires on street furniture while
    saying nothing about the vehicle you are actually following. The maths is
    right and the meaning is wrong.

    A lane is roughly 3.5 m wide, so half of it subtends f*1.75/d pixels at
    distance d - wide near the bumper, narrowing toward the vanishing point.
    An object whose centre falls outside that wedge is beside the vehicle's
    path, not in it.

    Assumes a straight path. On a curve the corridor should follow the steering
    angle or the lane markings; that needs signals this pipeline does not have,
    and pretending otherwise would be worse than stating the limit.
    """
    if not np.isfinite(distance) or distance <= 0:
        return False

    centre_x = (box[0] + box[2]) / 2.0
    half_corridor_px = focal_length * lane_half_width / distance
    return abs(centre_x - image_width / 2.0) < half_corridor_px
