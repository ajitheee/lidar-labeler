"""Synthetic-scene verification.

Real KITTI calibration values, a scene with known geometry, and assertions that
the recovered distance matches the truth we constructed. This is what lets you
trust the pipeline before you have looked at a single real frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_labeler import (  # noqa: E402
    Calibration, estimate_box_distance, project_to_image,
)

IMAGE_SHAPE = (375, 1242)  # KITTI's typical rectified size

# Genuine KITTI calibration values, so the test exercises a realistic transform
# rather than a convenient identity one.
P2 = np.array([
    [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
    [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
    [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
])
R0_RECT = np.array([
    [9.999239e-01, 9.837760e-03, -7.445048e-03],
    [-9.869795e-03, 9.999421e-01, -4.278459e-03],
    [7.402527e-03, 4.351614e-03, 9.999631e-01],
])
TR_VELO_TO_CAM = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
])

CALIB = Calibration(P2=P2, R0_rect=R0_RECT, Tr_velo_to_cam=TR_VELO_TO_CAM)


def _grid(x, y_range, z_range, n=40):
    """A vertical plane of points at forward distance x, in the Velodyne frame."""
    ys = np.linspace(*y_range, n)
    zs = np.linspace(*z_range, n)
    yy, zz = np.meshgrid(ys, zs)
    return np.stack([np.full(yy.size, x), yy.ravel(), zz.ravel()], axis=1)


def _true_depth(points_velo: np.ndarray) -> float:
    """Exact camera-frame forward depth of a point set, straight from the calibration."""
    homogeneous = np.hstack([points_velo, np.ones((len(points_velo), 1))])
    return float(np.median((CALIB.velo_to_cam @ homogeneous.T).T[:, 2]))


def _scene():
    """A car at 15 m, a wall at 40 m behind it, and a ground plane."""
    car = _grid(15.0, (-0.9, 0.9), (-1.4, 0.2))
    wall = _grid(40.0, (-8.0, 8.0), (-1.5, 3.0), n=60)
    gx, gy = np.meshgrid(np.linspace(4, 50, 90), np.linspace(-9, 9, 60))
    ground = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.73)], axis=1)
    return car, np.vstack([car, wall, ground])


def _loose_box(points_velo, pad=0.15):
    """The box a real detector would emit: the object, plus a margin of background."""
    cloud = project_to_image(points_velo, CALIB, IMAGE_SHAPE)
    x1, y1 = cloud.uv.min(axis=0)
    x2, y2 = cloud.uv.max(axis=0)
    w, h = x2 - x1, y2 - y1
    return (x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h)


def test_projection_lands_in_frame():
    _, scene = _scene()
    cloud = project_to_image(scene, CALIB, IMAGE_SHAPE)
    assert len(cloud) > 1000, "expected most of the scene to be visible"
    assert cloud.uv[:, 0].min() >= 0 and cloud.uv[:, 0].max() < IMAGE_SHAPE[1]
    assert cloud.uv[:, 1].min() >= 0 and cloud.uv[:, 1].max() < IMAGE_SHAPE[0]
    assert (cloud.depth > 0).all(), "no point behind the camera should survive"


def test_recovers_object_distance_despite_background():
    car, scene = _scene()
    cloud = project_to_image(scene, CALIB, IMAGE_SHAPE)
    result = estimate_box_distance(_loose_box(car), cloud)

    truth = _true_depth(car)
    assert result.is_reliable, f"should be confident here: {result}"
    assert abs(result.distance - truth) < 0.5, (
        f"got {result.distance:.2f} m, truth {truth:.2f} m"
    )
    assert result.distance < 20.0, "must not be dragged toward the 40 m wall"


def test_naive_median_would_have_been_wrong():
    """Documents why the clustering exists: the naive answer is materially worse."""
    car, scene = _scene()
    cloud = project_to_image(scene, CALIB, IMAGE_SHAPE)
    box = _loose_box(car)

    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    inside = (u >= box[0]) & (u <= box[2]) & (v >= box[1]) & (v <= box[3])
    naive = float(np.median(cloud.depth[inside]))

    truth = _true_depth(car)
    robust = estimate_box_distance(box, cloud).distance
    assert abs(robust - truth) < abs(naive - truth), (
        f"robust {robust:.2f} vs naive {naive:.2f}, truth {truth:.2f}"
    )


def test_stray_foreground_points_do_not_win():
    """A few near returns (a mirror, a pole edge) must not capture the label."""
    car, scene = _scene()
    stray = _grid(6.0, (-0.05, 0.05), (-0.4, 0.0), n=3)  # 9 points, very close
    cloud = project_to_image(np.vstack([scene, stray]), CALIB, IMAGE_SHAPE)
    result = estimate_box_distance(_loose_box(car), cloud)

    truth = _true_depth(car)
    assert abs(result.distance - truth) < 1.0, (
        f"stray foreground pulled the label to {result.distance:.2f} m"
    )


def test_occluded_distant_object_is_refused_not_guessed():
    """The failure this gate exists for, reproduced.

    A far car with few returns, and a small occluder in front of it. The near
    cluster wins the vote and the reported distance is the occluder's - so the
    label must be marked unreliable rather than published as a confident lie.
    """
    _, scene = _scene()
    far_car = _grid(45.0, (-0.9, 0.9), (-1.4, 0.2), n=6)      # ~36 sparse points
    occluder = _grid(12.0, (-0.12, 0.12), (-1.2, 0.6), n=3)   # 9 points, in front

    cloud = project_to_image(np.vstack([scene, far_car, occluder]), CALIB, IMAGE_SHAPE)
    result = estimate_box_distance(
        _loose_box(far_car), cloud, focal_length=CALIB.focal_length
    )

    truth = _true_depth(far_car)
    if result.is_reliable:
        assert abs(result.distance - truth) < 3.0, (
            f"published {result.distance:.1f} m as reliable when truth is "
            f"{truth:.1f} m - this is the -11 m bias seen on nuScenes"
        )


def test_occlusion_heuristic_flags_the_right_shape():
    """A small near cluster with a crowd behind it is the occluder signature."""
    from lidar_labeler.depth import BoxDistance

    occluder = BoxDistance(12.0, 60, 9, 0.2, 0.15, n_beyond=50)
    assert occluder.looks_occluded and not occluder.is_reliable

    # Same small cluster, but nothing behind it: a genuine sparse object.
    sparse = BoxDistance(12.0, 12, 9, 0.2, 0.75, n_beyond=1)
    assert not sparse.looks_occluded

    # Large cluster with background behind it: an ordinary well-observed object.
    solid = BoxDistance(14.7, 900, 500, 0.3, 0.55, n_beyond=300)
    assert not solid.looks_occluded and solid.is_reliable


def test_sparse_box_refuses_to_answer():
    _, scene = _scene()
    cloud = project_to_image(scene, CALIB, IMAGE_SHAPE)
    result = estimate_box_distance((5.0, 5.0, 12.0, 12.0), cloud)  # empty sky region
    assert not np.isfinite(result.distance)
    assert not result.is_reliable


def test_distance_ordering_is_monotonic():
    """Objects placed farther away must label as farther away."""
    _, scene = _scene()
    distances = []
    for x in (10.0, 20.0, 30.0):
        target = _grid(x, (-0.9, 0.9), (-1.4, 0.2))
        cloud = project_to_image(np.vstack([scene, target]), CALIB, IMAGE_SHAPE)
        distances.append(estimate_box_distance(_loose_box(target), cloud).distance)
    assert distances == sorted(distances), distances


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
