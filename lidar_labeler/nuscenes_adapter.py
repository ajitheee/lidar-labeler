"""nuScenes -> the same Calibration interface the rest of this repo already uses.

KITTI hands you one static LiDAR-to-camera transform because its sensors are
hardware-synchronised. nuScenes does not: the LiDAR sweep and the camera shutter
happen at different moments, and the car has moved in between. Projecting with a
single static extrinsic gives an overlay that looks *almost* right - points
drift a fraction of a metre along the direction of travel, worst at speed - and
quietly biases every distance label you produce.

The fix is to route through global coordinates, four transforms in total:

    LiDAR frame
      -> ego frame   at the LiDAR timestamp   (sensor extrinsics)
      -> global                               (ego pose at LiDAR time)
      -> ego frame   at the camera timestamp  (inverse ego pose at camera time)
      -> camera frame                         (inverse sensor extrinsics)

Only then do you apply the camera intrinsics. `build_calibration` composes all
of that into the same two matrices `project_to_image` already expects, so
nothing downstream changes.

Requires `pip install nuscenes-devkit` at runtime; the transform maths in
`transforms.py` is independent of it and tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import Detection
from .transforms import transform_matrix

# nuScenes class names are long and hierarchical. Collapse them to the handful
# that matter for a forward-facing driving model.
CLASS_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "truck",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
}


@dataclass(frozen=True)
class SensorCalibration:
    """The two matrices `project_to_image` needs, from any dataset.

    Deliberately matches the property names on the KITTI `Calibration` class so
    the two are interchangeable everywhere downstream.
    """

    velo_to_cam: np.ndarray    # 4x4
    velo_to_image: np.ndarray  # 3x4

    @property
    def focal_length(self) -> float:
        return float(self.velo_to_image[0, 0])

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.velo_to_image[0, 2]), float(self.velo_to_image[1, 2])


def build_calibration(nusc, lidar_token: str, cam_token: str) -> SensorCalibration:
    """Compose the motion-compensated LiDAR -> camera -> image transform."""
    lidar_sd = nusc.get("sample_data", lidar_token)
    cam_sd = nusc.get("sample_data", cam_token)

    lidar_calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    cam_calib = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    to_cam = (
        transform_matrix(cam_calib["translation"], cam_calib["rotation"], inverse=True)
        @ transform_matrix(cam_pose["translation"], cam_pose["rotation"], inverse=True)
        @ transform_matrix(lidar_pose["translation"], lidar_pose["rotation"])
        @ transform_matrix(lidar_calib["translation"], lidar_calib["rotation"])
    )

    intrinsic = np.zeros((3, 4), dtype=np.float64)
    intrinsic[:3, :3] = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)

    return SensorCalibration(velo_to_cam=to_cam, velo_to_image=intrinsic @ to_cam)


def load_pointcloud(path: str | Path) -> np.ndarray:
    """Load a nuScenes .pcd.bin sweep.

    Five float32 columns - x, y, z, intensity, ring - where KITTI has four.
    Reshaping to (-1, 4) here does not error, it just silently scrambles every
    point, so the column count is asserted rather than assumed.
    """
    scan = np.fromfile(str(path), dtype=np.float32)
    if scan.size % 5 != 0:
        raise ValueError(
            f"{path}: expected a multiple of 5 float32 values, got {scan.size}. "
            "nuScenes sweeps have 5 columns (x, y, z, intensity, ring)."
        )
    return scan.reshape(-1, 5)[:, :4]


@dataclass(frozen=True)
class GroundTruth:
    """A detection paired with everything needed to judge the label it produces."""

    detection: Detection
    depth: float            # nearest corner of the 3D box, metres
    visibility: int         # 1..4, roughly <40% .. >80% visible
    num_lidar_pts: int      # nuScenes' own count of points in the 3D box


def annotated_detections(
    nusc, cam_token: str, min_visibility: int = 2
) -> list[GroundTruth]:
    """2D boxes from nuScenes 3D annotations, each paired with its true depth.

    nuScenes has no native 2D boxes, so each 3D box is projected and its corners
    reduced to an axis-aligned extent. Boxes with any corner behind the image
    plane are dropped rather than clipped - a partially-behind box projects to
    nonsense coordinates that still look like a valid box.

    The depth returned is the nearest corner of the 3D box, which is the surface
    the LiDAR actually sees. Pairing it with the detection here, rather than
    matching them up later, is what keeps validation honest.
    """
    from nuscenes.utils.geometry_utils import view_points

    _, boxes, intrinsic = nusc.get_sample_data(cam_token)
    cam_sd = nusc.get("sample_data", cam_token)
    width, height = cam_sd["width"], cam_sd["height"]

    results: list[GroundTruth] = []
    for box in boxes:
        name = CLASS_MAP.get(box.name)
        if name is None:
            continue

        annotation = nusc.get("sample_annotation", box.token)
        # The visibility TOKEN is the 1-4 level. The `level` field is a string
        # like "v0-40" / "v80-100" - every one of which ends in "0", so parsing
        # its last character silently rejects every annotation in the dataset.
        if int(annotation["visibility_token"]) < min_visibility:
            continue

        corners = box.corners()                       # 3x8, camera frame
        if (corners[2, :] <= 0.1).any():              # any corner at/behind the lens
            continue

        projected = view_points(corners, intrinsic, normalize=True)[:2, :]
        x1, y1 = projected.min(axis=1)
        x2, y2 = projected.max(axis=1)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 - x1 < 4 or y2 - y1 < 4:                # degenerate after clipping
            continue

        results.append(GroundTruth(
            detection=Detection(
                box=(float(x1), float(y1), float(x2), float(y2)), class_name=name
            ),
            depth=float(corners[2, :].min()),
            visibility=int(annotation["visibility_token"]),
            num_lidar_pts=int(annotation.get("num_lidar_pts", 0)),
        ))
    return results


def detections_from_annotations(
    nusc, cam_token: str, min_visibility: int = 2
) -> list[Detection]:
    """Just the boxes, for the labelling path that does not need ground truth."""
    return [gt.detection for gt in annotated_detections(nusc, cam_token, min_visibility)]
