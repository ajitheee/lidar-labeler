"""The camera-only distance model and the TTC estimator."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_labeler.monocular import DistanceModel, evaluate, fit  # noqa: E402
from lidar_labeler.tracking import (  # noqa: E402
    TimeToCollision, in_ego_path, severity,
)

FOCAL, CAR_HEIGHT = 1250.0, 1.6


def _samples(n=300, seed=0, noise=0.5, name="car", height_m=CAR_HEIGHT):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.uniform(5, 30)
        out.append((name, FOCAL * height_m / d, d + rng.normal(0, noise)))
    return out


def test_recovers_the_physical_constant():
    model = fit(_samples())
    assert abs(model.coefficients["car"] - FOCAL * CAR_HEIGHT) < 40


def test_median_fit_survives_outliers_that_would_wreck_least_squares():
    """The labels contain occasional confident errors; the fit must ignore them."""
    clean = _samples(200)
    poisoned = clean + [("car", FOCAL * CAR_HEIGHT / 40, 8.0) for _ in range(20)]

    honest, damaged = fit(clean), fit(poisoned)
    drift = abs(damaged.coefficients["car"] - honest.coefficients["car"])
    assert drift < honest.coefficients["car"] * 0.1, f"outliers moved the fit by {drift:.0f}"


def test_rare_classes_fall_back_instead_of_fitting_noise():
    samples = _samples(200) + [("bicycle", FOCAL * 1.7 / 12, 12.0)] * 5
    model = fit(samples, min_samples=15)
    assert "car" in model.coefficients
    assert "bicycle" not in model.coefficients, "5 examples must not earn a coefficient"
    assert model.predict("bicycle", (0, 0, 10, 100)) == model.fallback / 100


def test_evaluate_reports_sane_error_on_clean_data():
    model = fit(_samples())
    scores = evaluate(model, [("car", FOCAL * CAR_HEIGHT / d, d) for d in np.linspace(5, 30, 60)])
    assert scores["mae"] < 0.5 and abs(scores["bias"]) < 0.5


def test_model_round_trips_through_disk():
    model = fit(_samples())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        model.save(path)
        assert DistanceModel.load(path).coefficients == model.coefficients


def test_ttc_on_a_closing_object():
    """50 m out, closing at 10 m/s, should read about 5 seconds."""
    ttc = TimeToCollision()
    result = None
    for i in range(8):
        result = ttc.update(1, 50.0 - i * 0.5, i * 0.05)   # 10 m/s over 20 fps
    assert result is not None and abs(result - (50 - 3.5) / 10) < 0.5


def test_stationary_and_receding_objects_give_no_countdown():
    ttc = TimeToCollision()
    for i in range(8):
        stationary = ttc.update(1, 30.0, i * 0.05)
        receding = ttc.update(2, 30.0 + i * 0.5, i * 0.05)
    assert stationary is None and receding is None


def test_no_answer_until_enough_samples():
    ttc = TimeToCollision(min_samples=4)
    assert ttc.update(1, 50.0, 0.0) is None
    assert ttc.update(1, 49.5, 0.05) is None
    assert ttc.update(1, 49.0, 0.10) is None
    assert ttc.update(1, 48.5, 0.15) is not None


def test_slope_fit_is_steadier_than_frame_differencing():
    """Why the window exists: noisy distances break naive differencing."""
    rng = np.random.default_rng(3)
    true = [50.0 - i * 0.5 for i in range(8)]
    noisy = [d + rng.normal(0, 0.5) for d in true]

    ttc = TimeToCollision()
    result = None
    for i, d in enumerate(noisy):
        result = ttc.update(1, d, i * 0.05)

    naive_speed = (noisy[-2] - noisy[-1]) / 0.05
    fitted_error = abs(result - noisy[-1] / 10.0)
    naive_error = abs(noisy[-1] / max(naive_speed, 1e-6) - noisy[-1] / 10.0)
    assert fitted_error < naive_error


def test_parked_cars_beside_the_road_are_out_of_path():
    """The demo's actual bug: warning about street furniture.

    Coordinates taken from a real frame - 1600 px wide, a parked car on the
    left kerb at 7 m, and a parked car on the right at 33 m. Neither is in the
    driving corridor, so neither should produce a countdown.
    """
    assert not in_ego_path((0, 500, 320, 830), 7.0, 1600)
    assert not in_ego_path((1035, 450, 1130, 540), 33.0, 1600)


def test_lead_vehicle_is_in_path():
    """The car actually being followed, near centre at ~15 m."""
    assert in_ego_path((890, 455, 1000, 545), 15.0, 1600)


def test_corridor_narrows_with_distance():
    """A fixed lateral offset leaves the corridor as the object gets further away."""
    box = (900, 400, 1000, 500)      # centre 950, i.e. 150 px right of centre
    assert in_ego_path(box, 10.0, 1600), "at 10 m the corridor is ~237 px"
    assert not in_ego_path(box, 30.0, 1600), "at 30 m it is only ~79 px"


def test_nonsense_distances_are_not_in_path():
    assert not in_ego_path((790, 400, 810, 500), float("nan"), 1600)
    assert not in_ego_path((790, 400, 810, 500), 0.0, 1600)
    assert not in_ego_path((790, 400, 810, 500), -5.0, 1600)


def test_severity_bands():
    assert severity(None) == "none"
    assert severity(1.5) == "critical"
    assert severity(3.0) == "warn"
    assert severity(9.0) == "ok"


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
