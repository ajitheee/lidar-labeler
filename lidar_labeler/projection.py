"""Project Velodyne points into the camera image plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SupportsProjection(Protocol):
    """Anything that can place LiDAR points in a camera image.

    Both the KITTI `Calibration` and the nuScenes `SensorCalibration` satisfy
    this, which is why nothing below the calibration layer is dataset-specific.
    """

    @property
    def velo_to_cam(self) -> np.ndarray: ...      # 4x4

    @property
    def velo_to_image(self) -> np.ndarray: ...    # 3x4

    @property
    def focal_length(self) -> float: ...          # pixels


@dataclass(frozen=True)
class ProjectedCloud:
    """Points that survived projection into the visible image."""

    uv: np.ndarray      # (N, 2) float pixel coordinates
    depth: np.ndarray   # (N,) metres along the camera optical axis (forward distance)
    xyz_cam: np.ndarray  # (N, 3) point positions in the rectified camera frame
    mask: np.ndarray    # (M,) bool selecting which of the M input points survived

    def __len__(self) -> int:
        return int(self.uv.shape[0])

    @property
    def euclidean_range(self) -> np.ndarray:
        """Straight-line distance from the camera centre, as opposed to forward depth.

        Forward depth is the right quantity for time-to-collision; euclidean range
        is the right one if you are comparing against a spec sheet's "detection
        range". They differ noticeably for objects far off to the side.
        """
        return np.linalg.norm(self.xyz_cam, axis=1)


def project_to_image(
    points_velo: np.ndarray,
    calib: SupportsProjection,
    image_shape: tuple[int, int],
    min_depth: float = 1.0,
    max_depth: float = 120.0,
) -> ProjectedCloud:
    """Project LiDAR points into image pixels, keeping only what the camera can see.

    Args:
        points_velo: (M, 3) or (M, 4) array in the Velodyne frame. A 4th column
            (reflectance, as KITTI stores it) is ignored.
        calib: calibration for this frame.
        image_shape: (height, width) of the RGB image.
        min_depth: drop points closer than this. Points at or behind the camera
            plane must be dropped BEFORE the perspective divide, otherwise they
            reproject to mirrored pixel coordinates and land somewhere plausible
            looking but completely wrong.
        max_depth: drop points beyond this; KITTI's Velodyne returns get sparse
            and noisy past ~80 m.

    Returns:
        ProjectedCloud containing only in-frame points.
    """
    points = np.asarray(points_velo, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"expected (M, 3+) point array, got {points.shape}")
    xyz = points[:, :3]

    homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])          # (M, 4)
    xyz_cam = (calib.velo_to_cam @ homogeneous.T).T[:, :3]              # (M, 3)
    projected = (calib.velo_to_image @ homogeneous.T).T                 # (M, 3)

    depth = projected[:, 2]
    in_front = (depth > min_depth) & (depth < max_depth)

    # Perspective divide only where depth is safely positive.
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    safe = in_front
    uv[safe] = projected[safe, :2] / depth[safe, None]

    height, width = image_shape
    in_frame = (
        safe
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )

    return ProjectedCloud(
        uv=uv[in_frame],
        depth=depth[in_frame],
        xyz_cam=xyz_cam[in_frame],
        mask=in_frame,
    )
