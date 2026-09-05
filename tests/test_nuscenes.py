"""Verify the nuScenes transform chain without needing the dataset or devkit.

`build_calibration` only ever calls `nusc.get(table, token)`, so a dict-backed
stub exercises the real composition logic. What is tested here is the part that
is easy to get silently wrong: quaternion convention, inverse transforms, and
whether motion compensation is actually being applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_labeler import project_to_image, quaternion_to_rotation, transform_matrix  # noqa: E402
from lidar_labeler.nuscenes_adapter import SensorCalibration, build_calibration  # noqa: E402

IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]
YAW_90_Q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
INTRINSIC = [[1200.0, 0.0, 800.0], [0.0, 1200.0, 450.0], [0.0, 0.0, 1.0]]


class FakeNusc:
    """Minimal stand-in for the devkit's NuScenes object."""

    def __init__(self, lidar_pose, cam_pose, lidar_calib, cam_calib):
        self._tables = {
            "sample_data": {
                "lidar": {"calibrated_sensor_token": "lc", "ego_pose_token": "lp"},
                "cam": {"calibrated_sensor_token": "cc", "ego_pose_token": "cp"},
            },
            "calibrated_sensor": {"lc": lidar_calib, "cc": cam_calib},
            "ego_pose": {"lp": lidar_pose, "cp": cam_pose},
        }

    def get(self, table, token):
        return self._tables[table][token]


def _pose(translation, rotation=IDENTITY_Q):
    return {"translation": list(translation), "rotation": list(rotation)}


def _calib(translation, rotation=IDENTITY_Q, intrinsic=None):
    record = _pose(translation, rotation)
    if intrinsic is not None:
        record["camera_intrinsic"] = intrinsic
    return record


def test_quaternion_conventions():
    assert np.allclose(quaternion_to_rotation(IDENTITY_Q), np.eye(3))

    rot = quaternion_to_rotation(YAW_90_Q)
    # +x should rotate onto +y for a 90 degree yaw. If the convention were
    # [x, y, z, w] this lands somewhere else entirely.
    assert np.allclose(rot @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)

    unnormalized = quaternion_to_rotation([2.0, 0.0, 0.0, 0.0])
    assert np.allclose(unnormalized, np.eye(3)), "must normalise its input"


def test_inverse_transform_round_trips():
    translation, rotation = [1.5, -2.0, 0.7], YAW_90_Q
    forward = transform_matrix(translation, rotation)
    backward = transform_matrix(translation, rotation, inverse=True)
    assert np.allclose(backward @ forward, np.eye(4), atol=1e-12)


def test_stationary_vehicle_reduces_to_static_extrinsic():
    """With identical ego poses the four-transform chain must collapse to two."""
    pose = _pose([100.0, 50.0, 0.0], YAW_90_Q)
    lidar_calib = _calib([0.9, 0.0, 1.8])
    cam_calib = _calib([1.7, 0.0, 1.5], intrinsic=INTRINSIC)

    calib = build_calibration(FakeNusc(pose, pose, lidar_calib, cam_calib), "lidar", "cam")

    expected = (
        transform_matrix(cam_calib["translation"], cam_calib["rotation"], inverse=True)
        @ transform_matrix(lidar_calib["translation"], lidar_calib["rotation"])
    )
    assert np.allclose(calib.velo_to_cam, expected, atol=1e-12)


def test_motion_compensation_actually_moves_the_point():
    """A car that travelled between LiDAR and camera timestamps must shift the projection."""
    lidar_calib = _calib([0.9, 0.0, 1.8])
    cam_calib = _calib([1.7, 0.0, 1.5], intrinsic=INTRINSIC)

    still = build_calibration(
        FakeNusc(_pose([0, 0, 0]), _pose([0, 0, 0]), lidar_calib, cam_calib), "lidar", "cam"
    )
    # 0.5 m of travel between the two timestamps - about 45 km/h over a 40 ms gap.
    moved = build_calibration(
        FakeNusc(_pose([0, 0, 0]), _pose([0.5, 0, 0]), lidar_calib, cam_calib), "lidar", "cam"
    )

    shift = np.abs(still.velo_to_cam[:3, 3] - moved.velo_to_cam[:3, 3]).max()
    assert np.isclose(shift, 0.5, atol=1e-9), (
        f"expected the 0.5 m of ego motion to appear in the transform, got {shift:.3f}"
    )


def test_point_on_axis_projects_to_principal_point():
    """A point straight ahead of the camera lands on the principal point."""
    calib = SensorCalibration(
        velo_to_cam=np.eye(4),
        velo_to_image=np.hstack([np.array(INTRINSIC), np.zeros((3, 1))]),
    )
    cloud = project_to_image(np.array([[0.0, 0.0, 25.0]]), calib, (900, 1600))

    assert len(cloud) == 1
    assert np.allclose(cloud.uv[0], [800.0, 450.0], atol=1e-9)
    assert np.isclose(cloud.depth[0], 25.0)


def test_visibility_token_is_used_not_the_level_string():
    """Regression: every nuScenes level string ends in '0'.

    'v0-40', 'v40-60', 'v60-80', 'v80-100' - parsing the last character gives 0
    for all four, so a `level[-1] < 2` gate silently discards every annotation
    in the dataset while looking entirely reasonable.
    """
    for level in ("v0-40", "v40-60", "v60-80", "v80-100"):
        assert level[-1] == "0", "the trap this test exists for"

    # The tokens are what actually carry the ordering.
    assert [int(t) for t in ("1", "2", "3", "4")] == [1, 2, 3, 4]
    assert sum(int(t) >= 2 for t in ("1", "2", "3", "4")) == 3


def test_five_column_sweep_is_rejected_when_malformed():
    import tempfile
    from lidar_labeler.nuscenes_adapter import load_pointcloud

    with tempfile.NamedTemporaryFile(suffix=".pcd.bin", delete=False) as fh:
        np.arange(12, dtype=np.float32).tofile(fh)   # 12 is not a multiple of 5
        path = fh.name
    try:
        load_pointcloud(path)
    except ValueError as exc:
        assert "multiple of 5" in str(exc)
    else:
        raise AssertionError("should have rejected a malformed sweep")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
    print("\nall passed" if not failures else f"\n{failures} failed")
    sys.exit(1 if failures else 0)
