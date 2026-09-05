"""The diagnostic aggregation, tested on distributions with known shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_labeler.report import (  # noqa: E402
    Record, by_brightness, by_distance, by_point_count, by_visibility, summarize,
)


def _record(truth, estimate, visibility=4, pts=100, reliable=True, brightness=float("nan")):
    return Record(truth, estimate, visibility, pts, reliable, brightness)


def test_unreliable_and_nan_are_excluded():
    records = [
        _record(10.0, 10.5),
        _record(10.0, 99.0, reliable=False),      # gated out
        _record(10.0, float("nan")),              # no measurement
    ]
    bucket = summarize(records, "x")
    assert bucket.total == 3 and bucket.produced == 1
    assert np.isclose(bucket.mae, 0.5)
    assert np.isclose(bucket.coverage, 1 / 3)


def test_empty_group_does_not_explode():
    bucket = summarize([], "empty")
    assert bucket.total == 0 and bucket.produced == 0
    assert np.isnan(bucket.mae) and bucket.coverage == 0.0


def test_median_separates_a_tail_from_uniform_noise():
    """The reason median sits next to MAE in the output."""
    tight = [_record(20.0, 20.0 + e) for e in np.linspace(-0.5, 0.5, 99)]
    tailed = tight + [_record(20.0, 60.0)]      # one catastrophic label

    a, b = summarize(tight, "tight"), summarize(tailed, "tailed")
    assert b.mae > a.mae * 1.5, "one bad label must move the mean"
    assert np.isclose(b.median_ae, a.median_ae, atol=0.02), "but not the median"


def test_bias_detects_a_systematic_offset():
    offset = [_record(t, t - 2.0) for t in np.linspace(5, 50, 40)]
    bucket = summarize(offset, "offset")
    assert np.isclose(bucket.bias, -2.0, atol=1e-9)
    assert np.isclose(bucket.mae, 2.0, atol=1e-9)


def test_distance_buckets_are_half_open_and_cover_everything():
    records = [_record(t, t) for t in (0.0, 9.9, 10.0, 29.9, 30.0, 49.9, 50.0, 500.0)]
    buckets = by_distance(records)
    assert sum(b.total for b in buckets) == len(records), "no record may be lost"
    # 0.0, 9.9 | 10.0 | 29.9 | 30.0, 49.9 | 50.0, 500.0
    assert [b.total for b in buckets] == [2, 1, 1, 2, 2]


def test_brightness_quartiles_partition_and_detect_a_coverage_gradient():
    """The diagnostic must actually surface a brightness-linked coverage drop.

    This is the shape that would confirm the reflectance hypothesis: dark
    objects returning too few LiDAR points to pass the gate, and so never
    reaching the training set at all.
    """
    dark = [_record(20.0, float("nan"), reliable=False, brightness=b)
            for b in np.linspace(20, 60, 40)]
    light = [_record(20.0, 20.2, brightness=b) for b in np.linspace(140, 200, 40)]

    buckets = by_brightness(dark + light)
    assert sum(b.total for b in buckets) == 80, "quartiles must cover every record"
    assert buckets[0].coverage == 0.0, "darkest quartile produced no labels"
    assert buckets[-1].coverage == 1.0, "lightest quartile produced all of them"


def test_brightness_grouping_ignores_records_without_a_measurement():
    assert by_brightness([_record(20.0, 20.0)]) == []


def test_visibility_and_point_groupings_partition_the_input():
    records = [_record(20.0, 20.0, visibility=v, pts=p)
               for v in (1, 2, 3, 4) for p in (5, 20, 60, 500)]
    assert sum(b.total for b in by_visibility(records)) == len(records)
    assert sum(b.total for b in by_point_count(records)) == len(records)


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
