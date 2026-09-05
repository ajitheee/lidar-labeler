#!/usr/bin/env python3
"""Saturday hour zero: prove the projection is right, then measure how good it is.

Two modes:

  overlay   Draw projected LiDAR points on the image. Look at it. The points
            must sit ON the cars, not beside them. Everything downstream is
            worthless if this picture is wrong.

  validate  Use KITTI's own 2D boxes, extract distances from LiDAR only, and
            compare against KITTI's 3D annotations. Gives you a real accuracy
            number before you have wired up any detector at all.

Usage:
    python demo.py overlay  --root /data/kitti --frame 000008
    python demo.py validate --root /data/kitti --limit 200

Expects the KITTI object-detection layout:
    root/image_2/000000.png  root/velodyne/000000.bin
    root/calib/000000.txt    root/label_2/000000.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lidar_labeler import (
    draw_box_with_distance, draw_points, estimate_box_distance,
    load_labels, load_velodyne_bin, parse_calib_file, project_to_image,
)


def load_frame(root: Path, frame: str):
    image = cv2.imread(str(root / "image_2" / f"{frame}.png"))
    if image is None:
        raise FileNotFoundError(f"no image for frame {frame} under {root}")
    points = load_velodyne_bin(root / "velodyne" / f"{frame}.bin")
    calib = parse_calib_file(root / "calib" / f"{frame}.txt")
    return image, points, calib


def cmd_overlay(args):
    root = Path(args.root)
    image, points, calib = load_frame(root, args.frame)
    cloud = project_to_image(points, calib, image.shape[:2])
    canvas = draw_points(image, cloud)

    label_path = root / "label_2" / f"{args.frame}.txt"
    if label_path.exists():
        for label in load_labels(label_path):
            result = estimate_box_distance(label.box, cloud)
            canvas = draw_box_with_distance(canvas, label.box, label.type, result)

    out = Path(args.out or f"overlay_{args.frame}.png")
    cv2.imwrite(str(out), canvas)
    print(f"{len(cloud)} points projected into frame -> {out}")
    print("Check that points land on objects. If they are offset, the calibration "
          "is being read or composed wrongly and nothing below this matters.")


def cmd_validate(args):
    root = Path(args.root)
    frames = sorted(p.stem for p in (root / "velodyne").glob("*.bin"))[: args.limit]
    if not frames:
        raise SystemExit(f"no .bin files under {root / 'velodyne'}")

    errors, skipped, total = [], 0, 0
    for frame in frames:
        try:
            image, points, calib = load_frame(root, frame)
        except FileNotFoundError:
            continue
        cloud = project_to_image(points, calib, image.shape[:2])
        for label in load_labels(root / "label_2" / f"{frame}.txt"):
            if label.type == "DontCare" or label.truncated > 0.5:
                continue
            total += 1
            result = estimate_box_distance(label.box, cloud)
            if not result.is_reliable:
                skipped += 1
                continue
            errors.append(result.distance - label.near_face_depth)

    if not errors:
        raise SystemExit("no reliable labels produced - inspect an overlay first")

    err = np.array(errors)
    print(f"\nobjects considered : {total}")
    print(f"labels produced    : {len(err)}  ({len(err)/total:.0%} coverage)")
    print(f"dropped as unsure  : {skipped}")
    print(f"\nMAE   {np.abs(err).mean():.2f} m")
    print(f"bias  {err.mean():+.2f} m   (persistent sign means a systematic offset,"
          " often the near-face vs centre convention)")
    print(f"p95   {np.percentile(np.abs(err), 95):.2f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_overlay = sub.add_parser("overlay", help="draw projected points and boxes")
    p_overlay.add_argument("--root", required=True, help="KITTI object-detection root")
    p_overlay.add_argument("--frame", default="000008")
    p_overlay.add_argument("--out", default=None)
    p_overlay.set_defaults(func=cmd_overlay)

    p_validate = sub.add_parser("validate", help="score LiDAR labels against KITTI 3D truth")
    p_validate.add_argument("--root", required=True)
    p_validate.add_argument("--limit", type=int, default=200)
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
