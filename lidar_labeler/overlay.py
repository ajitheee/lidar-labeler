"""Visual sanity checks. Look at these before trusting a single label."""

from __future__ import annotations

import cv2
import numpy as np

from .depth import BoxDistance
from .projection import ProjectedCloud


def draw_points(
    image: np.ndarray,
    cloud: ProjectedCloud,
    radius: int = 2,
    max_depth: float = 60.0,
) -> np.ndarray:
    """Overlay projected points, coloured near-to-far.

    If your calibration is wrong this is where you find out: the points will sit
    beside the cars rather than on them. Do not skip this step.
    """
    canvas = image.copy()
    if len(cloud) == 0:
        return canvas

    normalized = np.clip(cloud.depth / max_depth, 0.0, 1.0)
    colors = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
    ).reshape(-1, 3)

    for (u, v), color in zip(cloud.uv.astype(int), colors):
        cv2.circle(canvas, (int(u), int(v)), radius, color.tolist(), -1)
    return canvas


def draw_box_with_distance(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    result: BoxDistance,
) -> np.ndarray:
    """Draw one box annotated with its distance; unreliable boxes are greyed out."""
    canvas = image.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    color = (0, 200, 0) if result.is_reliable else (128, 128, 128)
    text = (
        f"{label} {result.distance:.1f}m"
        if np.isfinite(result.distance)
        else f"{label} ?"
    )

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        canvas, text, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas
