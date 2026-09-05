"""Predict distance from the camera alone, learned from the LiDAR labels.

This is where the project pays off. The LiDAR was only ever a teacher: once a
model can read distance off box geometry, the sensor can be removed and a $20
camera does the job.

The physics is a one-liner. An object of real height H at distance d projects to
a box of pixel height h = f*H/d, so d = (f*H)/h. Everything class-specific
collapses into a single constant C = f*H, and the estimator is d = C/h.

That deliberately beats a learned regression here. With a few hundred labels a
neural head would fit the noise; a one-parameter physical model cannot, it
extrapolates sanely past the distances it saw, and when it is wrong you can say
why - the object was an unusual height, or the box was clipped by the frame edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DistanceModel:
    """One constant per class: distance = coefficient / box_height_px."""

    coefficients: dict[str, float]
    fallback: float
    n_samples: dict[str, int] = field(default_factory=dict)

    def predict(self, class_name: str, box: tuple[float, float, float, float]) -> float:
        height = box[3] - box[1]
        if height <= 0:
            return float("nan")
        return self.coefficients.get(class_name, self.fallback) / height

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "coefficients": self.coefficients,
            "fallback": self.fallback,
            "n_samples": self.n_samples,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DistanceModel":
        data = json.loads(Path(path).read_text())
        return cls(data["coefficients"], data["fallback"], data.get("n_samples", {}))


def fit(samples: list[tuple[str, float, float]], min_samples: int = 15) -> DistanceModel:
    """Fit the per-class constant from (class_name, box_height_px, distance).

    Uses the median of d*h rather than least squares. The training labels come
    from a pipeline that we know produces occasional confident errors, and a
    least-squares fit hands those outliers disproportionate influence. The median
    ignores them. Classes with too few examples fall back to the global
    constant instead of fitting a coefficient to six cars.
    """
    if not samples:
        raise ValueError("no samples to fit")

    products = [(name, h * d) for name, h, d in samples if h > 0 and np.isfinite(d)]
    if not products:
        raise ValueError("no usable samples: every box had zero height or nan distance")

    fallback = float(np.median([p for _, p in products]))

    coefficients: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in {n for n, _ in products}:
        values = [p for n, p in products if n == name]
        counts[name] = len(values)
        if len(values) >= min_samples:
            coefficients[name] = float(np.median(values))

    return DistanceModel(coefficients, fallback, counts)


def evaluate(model: DistanceModel, samples: list[tuple[str, float, float]]) -> dict:
    """Score the camera-only estimator against the LiDAR-derived distances."""
    errors = []
    for name, height, truth in samples:
        if height <= 0 or not np.isfinite(truth):
            continue
        predicted = model.coefficients.get(name, model.fallback) / height
        errors.append(predicted - truth)

    err = np.array(errors)
    absolute = np.abs(err)
    return {
        "n": int(err.size),
        "mae": float(absolute.mean()),
        "median_ae": float(np.median(absolute)),
        "bias": float(err.mean()),
        "p95": float(np.percentile(absolute, 95)),
    }


def samples_from_sidecar(path: str | Path) -> list[tuple[str, float, float]]:
    """Read (class, box height, distance) triples out of a distances.json."""
    data = json.loads(Path(path).read_text())
    samples = []
    for objects in data.values():
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            samples.append((obj["class_name"], float(y2 - y1), float(obj["distance"])))
    return samples
