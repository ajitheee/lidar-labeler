"""Turn labelled frames into a dataset Roboflow can actually ingest.

The awkward part of this project is that no standard annotation format has a
field for "distance". Three ways to smuggle it through, and the right answer is
to use two of them at once:

  1. Bin it into the class name  - "car_10-20m". Survives every format, and
     RF-DETR learns it with no custom head at all. Costs you class count and
     resolution, and confuses the detection metric with the distance metric.
  2. A sidecar JSON keyed by image  - exact metres, nothing lost. Roboflow will
     not display it, but it is what you fit the monocular distance model on.
  3. A COCO custom attribute  - clean in principle, but importers are entitled
     to drop fields they do not recognise, so never let it be your only copy.

`export_dataset` writes clean class names for the detector AND the sidecar with
exact values. Train detection on the first, fit distance on the second. Use the
binned variant only if you want the "no custom head" demo as a comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .depth import BoxDistance, estimate_box_distance
from .projection import SupportsProjection, project_to_image

# A detector is anything that turns an image into boxes. Keeping it a plain
# callable means the pipeline does not care whether you use Grounding DINO,
# SAM 3, Roboflow Rapid, or KITTI's own labels while you are still testing.
Detector = Callable[[np.ndarray], Sequence["Detection"]]

DEFAULT_BINS = (0.0, 10.0, 20.0, 30.0, 50.0, float("inf"))


@dataclass(frozen=True)
class Detection:
    """A box from whatever named the object."""

    box: tuple[float, float, float, float]
    class_name: str
    confidence: float = 1.0


@dataclass(frozen=True)
class LabeledObject:
    """A detection that the LiDAR has now measured."""

    box: tuple[float, float, float, float]
    class_name: str
    confidence: float
    distance: float
    n_points: int
    inlier_fraction: float
    spread: float

    @classmethod
    def build(cls, detection: Detection, result: BoxDistance) -> "LabeledObject":
        return cls(
            box=detection.box,
            class_name=detection.class_name,
            confidence=detection.confidence,
            distance=result.distance,
            n_points=result.n_points,
            inlier_fraction=result.inlier_fraction,
            spread=result.spread,
        )


def binned_class_name(
    class_name: str, distance: float, bins: Sequence[float] = DEFAULT_BINS
) -> str:
    """Fold distance into the class name, e.g. 'car' at 14 m -> 'car_10-20m'.

    Bins are half-open [low, high), so a value sitting exactly on a boundary
    goes into the upper bin - matching what the printed label "0-10m" implies.
    """
    index = int(np.searchsorted(np.asarray(bins[1:]), distance, side="right"))
    index = min(index, len(bins) - 2)
    low, high = bins[index], bins[index + 1]
    suffix = f"{low:.0f}-{high:.0f}m" if np.isfinite(high) else f"{low:.0f}m+"
    return f"{class_name}_{suffix}"


def label_frame(
    image: np.ndarray,
    points_velo: np.ndarray,
    calib: "SupportsProjection",
    detector: Detector,
    keep_unreliable: bool = False,
) -> list[LabeledObject]:
    """Detect objects, measure each one with the LiDAR, drop what cannot be trusted."""
    cloud = project_to_image(points_velo, calib, image.shape[:2])
    labeled: list[LabeledObject] = []
    focal = getattr(calib, "focal_length", None)
    for detection in detector(image):
        result = estimate_box_distance(detection.box, cloud, focal_length=focal)
        if result.is_reliable or keep_unreliable:
            labeled.append(LabeledObject.build(detection, result))
    return labeled


def export_dataset(
    frames: Iterable[tuple[str, tuple[int, int], Sequence[LabeledObject]]],
    out_dir: str | Path,
    use_binned_classes: bool = False,
) -> dict:
    """Write COCO annotations plus a distance sidecar.

    Args:
        frames: (image_filename, (height, width), labelled objects) per frame.
        out_dir: directory to write `_annotations.coco.json` and `distances.json`.
        use_binned_classes: fold distance into class names. Off by default -
            keep detection and distance as separate problems unless you are
            deliberately running the no-custom-head comparison.

    Returns:
        Summary counts, worth logging so a silently empty export is obvious.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, annotations, sidecar = [], [], {}
    categories: dict[str, int] = {}
    annotation_id = 1

    for image_id, (filename, (height, width), objects) in enumerate(frames, start=1):
        images.append(
            {"id": image_id, "file_name": filename, "height": height, "width": width}
        )
        sidecar[filename] = [asdict(obj) for obj in objects]

        for obj in objects:
            name = (
                binned_class_name(obj.class_name, obj.distance)
                if use_binned_classes
                else obj.class_name
            )
            category_id = categories.setdefault(name, len(categories) + 1)

            x1, y1, x2, y2 = obj.box
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],   # COCO wants xywh
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                # Best-effort extras. The sidecar is the copy that is guaranteed
                # to survive whatever the importer decides to keep.
                "distance_m": round(obj.distance, 3),
                "distance_n_points": obj.n_points,
            })
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for name, cid in categories.items()
        ],
    }

    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2))
    (out_dir / "distances.json").write_text(json.dumps(sidecar, indent=2))

    return {
        "images": len(images),
        "annotations": len(annotations),
        "classes": len(categories),
        "out_dir": str(out_dir),
    }
