"""KITTI calibration parsing and the LiDAR -> image projection matrix.

The KITTI object-detection benchmark ships one calib txt per frame:

    P0..P3       3x4  projection matrices for each rectified camera
    R0_rect      3x3  rectifying rotation for the reference camera
    Tr_velo_to_cam 3x4 rigid transform, Velodyne frame -> reference camera frame

A Velodyne point is projected into camera 2 (the left colour camera, which is
what the RGB images come from) as:

    x_img_homogeneous = P2 @ R0_rect_4x4 @ Tr_velo_to_cam_4x4 @ [x, y, z, 1]

The third component of the result is the depth along the camera's optical
axis, which is the number we actually want as a distance label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _to_4x4(mat: np.ndarray) -> np.ndarray:
    """Promote a 3x3 rotation or 3x4 rigid transform to a 4x4 homogeneous matrix."""
    out = np.eye(4, dtype=np.float64)
    rows, cols = mat.shape
    out[:rows, :cols] = mat
    return out


@dataclass(frozen=True)
class Calibration:
    """Composed calibration for one KITTI frame."""

    P2: np.ndarray            # 3x4
    R0_rect: np.ndarray       # 3x3
    Tr_velo_to_cam: np.ndarray  # 3x4

    @property
    def velo_to_image(self) -> np.ndarray:
        """3x4 matrix mapping homogeneous Velodyne points to homogeneous image points."""
        return self.P2 @ _to_4x4(self.R0_rect) @ _to_4x4(self.Tr_velo_to_cam)

    @property
    def velo_to_cam(self) -> np.ndarray:
        """4x4 matrix mapping Velodyne points into the rectified camera frame."""
        return _to_4x4(self.R0_rect) @ _to_4x4(self.Tr_velo_to_cam)

    @property
    def focal_length(self) -> float:
        """Horizontal focal length in pixels, for monocular distance work later."""
        return float(self.P2[0, 0])

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.P2[0, 2]), float(self.P2[1, 2])


def parse_calib_file(path: str | Path) -> Calibration:
    """Read a KITTI object-detection calib txt into a Calibration."""
    values: dict[str, np.ndarray] = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, payload = line.partition(":")
            try:
                values[key.strip()] = np.array(
                    [float(tok) for tok in payload.split()], dtype=np.float64
                )
            except ValueError:
                continue  # non-numeric rows (e.g. "calib_time") are ignored

    missing = {"P2", "R0_rect", "Tr_velo_to_cam"} - values.keys()
    if missing:
        raise KeyError(
            f"{path}: calib file is missing {sorted(missing)}. "
            "Note the raw-data calib files use different key names than the "
            "object-detection ones; this parser expects the object-detection format."
        )

    return Calibration(
        P2=values["P2"].reshape(3, 4),
        R0_rect=values["R0_rect"].reshape(3, 3),
        Tr_velo_to_cam=values["Tr_velo_to_cam"].reshape(3, 4),
    )
