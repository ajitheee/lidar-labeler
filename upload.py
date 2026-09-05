#!/usr/bin/env python3
"""Label a KITTI split with LiDAR and push it to Roboflow.

    export ROBOFLOW_API_KEY=...
    python upload.py --root /path/to/kitti --limit 500 --project lidar-distance

Runs with KITTI's own boxes by default so you can exercise the whole path
before wiring up a zero-shot detector. Swap `kitti_detector` for Grounding DINO
/ SAM 3 / Rapid and nothing else changes - that is the point of the Detector
being a plain callable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from lidar_labeler import (
    Detection, export_dataset, label_frame, load_labels,
    load_velodyne_bin, parse_calib_file,
)

# KITTI classes worth keeping. Its long tail ('Misc', 'DontCare') is noise here.
KEEP = {"Car", "Van", "Truck", "Pedestrian", "Cyclist"}


def kitti_detector(label_path: Path):
    """Stand-in detector: replay KITTI's 2D boxes. Replace with a zero-shot model."""
    def detect(_image):
        return [
            Detection(box=l.box, class_name=l.type.lower(), confidence=1.0)
            for l in load_labels(label_path)
            if l.type in KEEP and l.truncated <= 0.5
        ]
    return detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="KITTI root with image_2/ etc")
    parser.add_argument("--out", default="dataset_out")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--binned-classes", action="store_true",
                        help="fold distance into class names as a comparison run")
    parser.add_argument("--project", default=None,
                        help="Roboflow project id; omit to export locally only")
    args = parser.parse_args()

    root = Path(args.root)
    out_images = Path(args.out) / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    frames_data, empty = [], 0
    for frame in sorted(p.stem for p in (root / "velodyne").glob("*.bin"))[: args.limit]:
        image_path = root / "image_2" / f"{frame}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        objects = label_frame(
            image,
            load_velodyne_bin(root / "velodyne" / f"{frame}.bin"),
            parse_calib_file(root / "calib" / f"{frame}.txt"),
            kitti_detector(root / "label_2" / f"{frame}.txt"),
        )
        if not objects:
            empty += 1
            continue

        cv2.imwrite(str(out_images / f"{frame}.png"), image)
        frames_data.append((f"{frame}.png", image.shape[:2], objects))

    summary = export_dataset(frames_data, args.out, use_binned_classes=args.binned_classes)
    print(f"\n{summary['images']} images, {summary['annotations']} boxes, "
          f"{summary['classes']} classes -> {summary['out_dir']}")
    print(f"{empty} frames yielded no reliable label and were skipped.")
    print("distances.json holds exact metres; the COCO file is what Roboflow ingests.")

    if not args.project:
        print("\nNo --project given, so nothing was uploaded.")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("set ROBOFLOW_API_KEY (a private key) before uploading")

    from roboflow import Roboflow  # imported late so local export needs no SDK

    project = Roboflow(api_key=api_key).workspace().project(args.project)
    annotations = str(Path(args.out) / "_annotations.coco.json")
    for filename, _shape, _objects in frames_data:
        project.upload(
            image_path=str(out_images / filename),
            annotation_path=annotations,
            batch_name="lidar-auto-labeled",
        )
    print(f"uploaded {len(frames_data)} images to {args.project}")
    print("Keep distances.json - Roboflow will not preserve the exact values.")


if __name__ == "__main__":
    main()
