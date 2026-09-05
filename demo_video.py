#!/usr/bin/env python3
"""The demo: detections, camera-only distance, and a time-to-collision warning.

    pip install inference supervision
    python demo_video.py --video drive.mp4 --model your-workspace/your-model/1

Runs the same on a laptop and on a Jetson - `inference` picks the backend. The
LiDAR is absent here by design: it taught the distance model and is no longer
needed, which is the whole argument of the project.

Confidence defaults to 0.40 rather than the F1-optimal threshold. F1 treats a
missed car and a spurious box as equally costly; a collision warning does not.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from lidar_labeler.monocular import DistanceModel
from lidar_labeler.tracking import TimeToCollision, in_ego_path, severity

COLORS = {                      # BGR
    "ok": (120, 190, 120),
    "warn": (60, 170, 235),
    "critical": (60, 60, 225),
    "none": (190, 170, 120),
}


def load_backend(model_id: str, api_key: str):
    """Return an infer(frame, confidence) callable, local if available.

    Local `inference` runs the weights on this machine and is what you want on
    a Jetson, where measuring on-device FPS is the entire point. It depends on
    torch, which will not install on a stock Windows Python: torch ships nested
    third_party directories that exceed the 260-character MAX_PATH limit.

    The hosted client needs no torch and is fine for producing a demo video -
    it just costs a network round trip per frame, so a 400-frame clip takes
    minutes rather than seconds.
    """
    try:
        from inference import get_model
    except ImportError:
        pass
    else:
        model = get_model(model_id=model_id, api_key=api_key)
        return (lambda frame, conf: model.infer(frame, confidence=conf)[0]), "local"

    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise SystemExit(
            "install one of:\n"
            "  pip install inference        (local, needs torch)\n"
            "  pip install inference-sdk    (hosted, no torch)"
        ) from exc

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com", api_key=api_key
    )
    return (
        lambda frame, conf: client.infer(frame, model_id=model_id),
        "hosted (network round trip per frame - use --max-frames to keep it short)",
    )


def annotate(frame, box, label, band):
    x1, y1, x2, y2 = (int(v) for v in box)
    color = COLORS[band]
    thickness = 3 if band == "critical" else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 9), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="Roboflow model id, e.g. project/1")
    parser.add_argument("--distance-model", default="distance_model.json")
    parser.add_argument("--out", default="demo.mp4")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--focal", type=float, default=1250.0,
                        help="camera focal length in pixels, for the path corridor")
    parser.add_argument("--lane-half-width", type=float, default=1.9,
                        help="metres either side of centre counted as in-path")
    parser.add_argument("--no-path-filter", action="store_true",
                        help="warn on every object, in path or not")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("set ROBOFLOW_API_KEY first")

    infer, backend = load_backend(args.model, api_key)
    print(f"inference backend: {backend}")
    distances = DistanceModel.load(args.distance_model)
    tracker = sv.ByteTrack()
    ttc = TimeToCollision()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    index, alerts = 0, 0
    while True:
        ok, frame = capture.read()
        if not ok or (args.max_frames and index >= args.max_frames):
            break

        detections = tracker.update_with_detections(
            sv.Detections.from_inference(infer(frame, args.confidence))
        )
        timestamp = index / fps

        for i in range(len(detections)):
            box = detections.xyxy[i]
            name = detections.data.get("class_name", ["object"] * len(detections))[i]
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else i

            metres = distances.predict(name, tuple(box))
            label = f"{name} {metres:.0f}m"

            # Only objects in the driving corridor get a countdown. Everything
            # else is drawn plainly - present, measured, not a threat.
            if args.no_path_filter or in_ego_path(
                tuple(box), metres, width, args.focal, args.lane_half_width
            ):
                seconds = ttc.update(track_id, metres, timestamp)
                band = severity(seconds)
                alerts += band == "critical"
                if seconds is not None:
                    label += f"  TTC {seconds:.1f}s"
            else:
                band = "none"

            frame = annotate(frame, box, label, band)

        if detections.tracker_id is not None:
            tracker_ids = set(int(t) for t in detections.tracker_id)
            ttc.forget(tracker_ids)

        cv2.putText(frame, "camera only - no LiDAR at inference", (14, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        writer.write(frame)
        index += 1
        if index % 50 == 0:
            print(f"  {index} frames")

    capture.release()
    writer.release()
    print(f"\n{index} frames -> {args.out}")
    print(f"{alerts} critical-TTC detections (under 2 s to contact)")


if __name__ == "__main__":
    main()
