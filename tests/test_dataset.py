"""Verify the export carries distance through intact."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidar_labeler import Detection, binned_class_name, export_dataset, label_frame  # noqa: E402
from lidar_labeler.dataset import LabeledObject  # noqa: E402
from test_synthetic import CALIB, IMAGE_SHAPE, _loose_box, _scene, _true_depth  # noqa: E402


def test_binning_boundaries():
    assert binned_class_name("car", 5.0) == "car_0-10m"
    assert binned_class_name("car", 14.0) == "car_10-20m"
    assert binned_class_name("car", 10.0) == "car_10-20m"     # boundary goes up
    assert binned_class_name("car", 999.0) == "car_50m+"      # unbounded top bin
    assert binned_class_name("pedestrian", 0.0) == "pedestrian_0-10m"


def test_label_frame_measures_and_filters():
    car, scene = _scene()
    image = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)
    detector = lambda _img: [
        Detection(box=_loose_box(car), class_name="car", confidence=0.9),
        Detection(box=(5.0, 5.0, 12.0, 12.0), class_name="car", confidence=0.8),  # empty sky
    ]

    labeled = label_frame(image, scene, CALIB, detector)
    assert len(labeled) == 1, "the unmeasurable box should have been dropped"
    assert abs(labeled[0].distance - _true_depth(car)) < 0.5
    assert labeled[0].class_name == "car"


def test_export_roundtrip_preserves_distance():
    objects = [
        LabeledObject((10, 20, 110, 120), "car", 0.9, 14.72, 900, 0.55, 0.3),
        LabeledObject((300, 40, 360, 140), "pedestrian", 0.8, 31.4, 120, 0.42, 0.9),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        summary = export_dataset([("000008.png", IMAGE_SHAPE, objects)], tmp)
        assert summary == {
            "images": 1, "annotations": 2, "classes": 2, "out_dir": tmp,
        }

        coco = json.loads((Path(tmp) / "_annotations.coco.json").read_text())
        # COCO stores xywh, not xyxy - getting this wrong shifts every box.
        assert coco["annotations"][0]["bbox"] == [10, 20, 100, 100]
        assert coco["annotations"][0]["distance_m"] == 14.72

        sidecar = json.loads((Path(tmp) / "distances.json").read_text())
        distances = [o["distance"] for o in sidecar["000008.png"]]
        assert distances == [14.72, 31.4], "sidecar must keep exact values"


def test_binned_export_creates_distance_classes():
    objects = [
        LabeledObject((10, 20, 110, 120), "car", 0.9, 14.72, 900, 0.55, 0.3),
        LabeledObject((200, 20, 260, 90), "car", 0.9, 42.0, 300, 0.50, 0.8),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        export_dataset([("a.png", IMAGE_SHAPE, objects)], tmp, use_binned_classes=True)
        coco = json.loads((Path(tmp) / "_annotations.coco.json").read_text())
        names = sorted(c["name"] for c in coco["categories"])
        assert names == ["car_10-20m", "car_30-50m"], names


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
