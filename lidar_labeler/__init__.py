"""LiDAR-supervised auto-labelling: let the point cloud measure what the detector names."""

from .calib import Calibration, parse_calib_file
from .dataset import (
    Detection, LabeledObject, binned_class_name, export_dataset, label_frame,
)
from .depth import BoxDistance, estimate_box_distance, shrink_box
from .io_kitti import KittiLabel, load_labels, load_velodyne_bin
from .overlay import draw_box_with_distance, draw_points
from .monocular import DistanceModel, evaluate as evaluate_distance, fit as fit_distance
from .tracking import TimeToCollision, severity
from .projection import ProjectedCloud, SupportsProjection, project_to_image
from .transforms import quaternion_to_rotation, transform_matrix

__all__ = [
    "Calibration", "parse_calib_file",
    "ProjectedCloud", "project_to_image", "SupportsProjection",
    "quaternion_to_rotation", "transform_matrix",
    "DistanceModel", "fit_distance", "evaluate_distance",
    "TimeToCollision", "severity",
    "BoxDistance", "estimate_box_distance", "shrink_box",
    "load_velodyne_bin", "Detection", "LabeledObject",
    "label_frame", "export_dataset", "binned_class_name", "load_labels", "KittiLabel",
    "draw_points", "draw_box_with_distance",
]
