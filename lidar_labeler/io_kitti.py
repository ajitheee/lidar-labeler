"""Readers for KITTI on-disk formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_velodyne_bin(path: str | Path) -> np.ndarray:
    """Load a KITTI Velodyne scan.

    The .bin files are flat float32 records of (x, y, z, reflectance) in the
    Velodyne frame: x forward, y left, z up, metres.
    """
    scan = np.fromfile(str(path), dtype=np.float32)
    if scan.size % 4 != 0:
        raise ValueError(
            f"{path}: expected a multiple of 4 float32 values, got {scan.size}. "
            "This usually means the file is truncated or is not a KITTI .bin."
        )
    return scan.reshape(-1, 4)


from dataclasses import dataclass


@dataclass(frozen=True)
class KittiLabel:
    """One object from a KITTI label_2 txt."""

    type: str
    truncated: float
    occluded: int
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    dimensions: tuple[float, float, float]   # height, width, length in metres
    location: tuple[float, float, float]     # x, y, z of the 3D box bottom-centre, camera frame
    rotation_y: float

    @property
    def center_depth(self) -> float:
        """Forward distance to the object's centre."""
        return self.location[2]

    @property
    def near_face_depth(self) -> float:
        """Forward distance to the face pointing at the camera.

        This is the honest comparison for a LiDAR-derived label. The LiDAR only
        ever sees the near surface, so it will read roughly half an object-length
        closer than KITTI's centre annotation. Comparing against `center_depth`
        instead makes your labels look biased when they are in fact correct.
        """
        return self.location[2] - self.dimensions[2] / 2.0


def load_labels(path: str | Path) -> list[KittiLabel]:
    """Parse a KITTI label_2 txt. 'DontCare' entries are skipped."""
    labels: list[KittiLabel] = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 15 or parts[0] == "DontCare":
                continue
            values = [float(p) for p in parts[1:15]]
            labels.append(
                KittiLabel(
                    type=parts[0],
                    truncated=values[0],
                    occluded=int(values[1]),
                    box=(values[3], values[4], values[5], values[6]),
                    dimensions=(values[7], values[8], values[9]),
                    location=(values[10], values[11], values[12]),
                    rotation_y=values[13],
                )
            )
    return labels
