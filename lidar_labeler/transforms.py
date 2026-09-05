"""Rigid-transform helpers, written out rather than pulled from a devkit.

nuScenes stores every pose as a translation plus a quaternion in [w, x, y, z]
order. Scipy and most other libraries expect [x, y, z, w]. Feeding one to the
other produces a rotation that is wrong but not obviously wrong, which is the
worst kind of bug in this pipeline - so the conversion lives here, explicitly.
"""

from __future__ import annotations

import numpy as np


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a nuScenes quaternion, which is [w, x, y, z]."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("zero-length quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def transform_matrix(
    translation: np.ndarray, rotation: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """4x4 homogeneous transform from a translation and a [w, x, y, z] quaternion.

    With `inverse=True` returns the inverse transform directly, which is what
    you want going *into* a sensor frame rather than out of one.
    """
    matrix = np.eye(4, dtype=np.float64)
    rot = quaternion_to_rotation(rotation)
    translation = np.asarray(translation, dtype=np.float64)

    if inverse:
        matrix[:3, :3] = rot.T
        matrix[:3, 3] = rot.T @ -translation
    else:
        matrix[:3, :3] = rot
        matrix[:3, 3] = translation
    return matrix
