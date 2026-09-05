#!/usr/bin/env python3
"""The whole pipeline on nuScenes v1.0-mini.

    pip install nuscenes-devkit
    python run_nuscenes.py overlay  --root /data/nuscenes
    python run_nuscenes.py validate --root /data/nuscenes --limit 200
    python run_nuscenes.py export   --root /data/nuscenes --out dataset_out

overlay   draw projected points and measured boxes on one frame
validate  score LiDAR-derived distances against nuScenes 3D annotations
diagnose  break that error down by distance, visibility and point count
export    label frames and write the COCO + sidecar dataset

Uses CAM_FRONT and LIDAR_TOP. nuScenes mini is 10 scenes, ~400 keyframes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lidar_labeler import (
    draw_box_with_distance, draw_points, estimate_box_distance,
    export_dataset, label_frame, project_to_image,
)
from lidar_labeler.nuscenes_adapter import (
    annotated_detections, build_calibration, detections_from_annotations,
    load_pointcloud,
)
from lidar_labeler.report import (
    Record, by_brightness, by_distance, by_point_count, by_visibility, render,
)

CAM, LIDAR = "CAM_FRONT", "LIDAR_TOP"


def open_dataset(root: str, version: str = "v1.0-mini"):
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=root, verbose=False)


def read_frame(nusc, sample):
    """Image, points, calibration and detections for one keyframe."""
    cam_token = sample["data"][CAM]
    lidar_token = sample["data"][LIDAR]

    image = cv2.imread(str(Path(nusc.dataroot) / nusc.get("sample_data", cam_token)["filename"]))
    points = load_pointcloud(Path(nusc.dataroot) / nusc.get("sample_data", lidar_token)["filename"])
    calib = build_calibration(nusc, lidar_token, cam_token)
    return image, points, calib, cam_token


def cmd_overlay(args):
    nusc = open_dataset(args.root, args.version)
    sample = nusc.sample[args.index]
    image, points, calib, cam_token = read_frame(nusc, sample)

    cloud = project_to_image(points, calib, image.shape[:2])
    canvas = draw_points(image, cloud)
    for detection in detections_from_annotations(nusc, cam_token):
        result = estimate_box_distance(detection.box, cloud)
        canvas = draw_box_with_distance(canvas, detection.box, detection.class_name, result)

    out = Path(args.out or f"overlay_nuscenes_{args.index}.png")
    cv2.imwrite(str(out), canvas)
    print(f"{len(cloud)} points projected -> {out}")
    print("Points must sit ON the vehicles. A consistent smear along the direction "
          "of travel means motion compensation is not being applied.")


def cmd_validate(args):
    """Score LiDAR-derived distances against the 3D annotations.

    Each detection is compared against the box it was derived from. An earlier
    version matched each result to whichever annotation had the nearest depth,
    which is circular - it picks the box that makes the error look smallest and
    reports a flatteringly wrong MAE no matter how bad the pipeline is.
    """
    nusc = open_dataset(args.root, args.version)
    errors: list[float] = []
    total = unreliable = no_points = 0

    for sample in nusc.sample[: args.limit]:
        image, points, calib, cam_token = read_frame(nusc, sample)
        if image is None:
            continue
        cloud = project_to_image(points, calib, image.shape[:2])

        for gt in annotated_detections(nusc, cam_token):
            total += 1
            result = estimate_box_distance(
                gt.detection.box, cloud, focal_length=calib.focal_length
            )
            if not result.is_reliable:
                unreliable += 1
                no_points += result.n_points == 0
                continue
            errors.append(result.distance - gt.depth)

    if total == 0:
        raise SystemExit(
            "No annotations survived filtering. Either the class map matches "
            "nothing in this split, or every box was rejected as too small or "
            "behind the camera. Check the overlay for drawn boxes."
        )
    if not errors:
        raise SystemExit(
            f"{total} annotations found, but none produced a reliable distance "
            f"({no_points} had no LiDAR points inside the box at all).\n"
            "No points inside any box means the projection is misaligned - "
            "inspect the overlay. Points present but rejected means the gate in "
            "depth.py is too strict for this data; loosen min_points."
        )

    err = np.array(errors)
    print(f"\nannotations        : {total}")
    print(f"labels produced    : {len(err)}  ({len(err)/total:.0%} coverage)")
    print(f"dropped as unsure  : {unreliable}")
    print(f"\nMAE   {np.abs(err).mean():.2f} m")
    print(f"bias  {err.mean():+.2f} m")
    print(f"p95   {np.percentile(np.abs(err), 95):.2f} m")


def _mean_brightness(image, box) -> float:
    """Mean grey level inside a box, as a stand-in for how dark the object is."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2 = min(x2, image.shape[1]); y2 = min(y2, image.shape[0])
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    return float(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).mean())


def cmd_diagnose(args):
    """Where the error actually lives.

    Run this instead of tuning blind. The aggregate MAE cannot distinguish
    "uniformly mediocre" from "excellent except for one population", and those
    two want opposite fixes.
    """
    nusc = open_dataset(args.root, args.version)
    records: list[Record] = []

    for sample in nusc.sample[: args.limit]:
        image, points, calib, cam_token = read_frame(nusc, sample)
        if image is None:
            continue
        cloud = project_to_image(points, calib, image.shape[:2])

        for gt in annotated_detections(nusc, cam_token, min_visibility=1):
            result = estimate_box_distance(
                gt.detection.box, cloud, focal_length=calib.focal_length
            )
            records.append(Record(
                truth=gt.depth,
                estimate=result.distance,
                visibility=gt.visibility,
                num_lidar_pts=gt.num_lidar_pts,
                reliable=result.is_reliable,
                brightness=_mean_brightness(image, gt.detection.box),
            ))

    if not records:
        raise SystemExit("no annotations found - check the overlay first")

    print(f"\n{len(records)} annotations across {args.limit} keyframes "
          "(visibility filter off, so every level is represented)")
    print(render(by_distance(records), "ERROR BY DISTANCE"))
    print(render(by_visibility(records), "ERROR BY VISIBILITY"))
    print(render(by_point_count(records), "ERROR BY LIDAR POINT COUNT"))
    print(render(by_brightness(records), "ERROR BY OBJECT BRIGHTNESS"))
    print("\nRead the median column against MAE. Where they diverge, a tail is "
          "setting the mean.\nWhere coverage collapses, the reliability gate is "
          "starving rather than the projection failing.")


def cmd_export(args):
    nusc = open_dataset(args.root, args.version)
    out_images = Path(args.out) / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    frames, empty = [], 0
    for index, sample in enumerate(nusc.sample[: args.limit]):
        image, points, calib, cam_token = read_frame(nusc, sample)
        if image is None:
            continue
        def detector(_img, token=cam_token):
            return [
                gt.detection
                for gt in annotated_detections(nusc, token, args.min_visibility)
                if gt.depth <= args.max_distance
            ]

        objects = label_frame(image, points, calib, detector)
        if not objects:
            empty += 1
            continue
        filename = f"{index:06d}.jpg"
        cv2.imwrite(str(out_images / filename), image)
        frames.append((filename, image.shape[:2], objects))

    summary = export_dataset(frames, args.out, use_binned_classes=args.binned_classes)
    print(f"\n{summary['images']} images, {summary['annotations']} boxes, "
          f"{summary['classes']} classes -> {summary['out_dir']}")
    print(f"{empty} frames yielded no reliable label.")
    print(f"envelope: <= {args.max_distance:.0f} m, visibility >= {args.min_visibility}")


def main():
    # --root is accepted either before or after the subcommand. argparse makes
    # this awkward: a top-level optional normally has to precede the subcommand,
    # and a subparser's defaults overwrite whatever the top level already set.
    # SUPPRESS defaults mean an absent flag sets no attribute at all, so
    # whichever position the user actually used is the one that survives.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS,
                        help="nuScenes dataroot (the folder holding samples/ and v1.0-mini/)")
    common.add_argument("--version", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("overlay", parents=[common])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("validate", parents=[common])
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("diagnose", parents=[common])
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("export", parents=[common])
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--out", default="dataset_out")
    p.add_argument("--binned-classes", action="store_true")
    # Defaults are the operating envelope measured by `diagnose` on nuScenes
    # mini: inside these bounds the labels are sub-metre and unbiased; outside
    # them a 32-beam sweep leaves too few returns to measure anything honestly.
    p.add_argument("--max-distance", type=float, default=30.0,
                   help="drop annotations beyond this distance (default 30 m)")
    p.add_argument("--min-visibility", type=int, default=4, choices=[1, 2, 3, 4],
                   help="nuScenes visibility level, 4 = >80%% visible (default 4)")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if not getattr(args, "root", None):
        parser.error("--root is required (the folder holding samples/ and v1.0-mini/)")
    args.version = getattr(args, "version", "v1.0-mini")
    args.func(args)


if __name__ == "__main__":
    main()
