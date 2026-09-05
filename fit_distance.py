#!/usr/bin/env python3
"""Fit the camera-only distance model from the LiDAR labels, and score it.

    python fit_distance.py --sidecar dataset_out\distances.json

Holds out 20% so the reported error is on data the fit never saw. That number -
distance error from a single camera, no LiDAR at inference time - is the
project's headline result.
"""

from __future__ import annotations

import argparse
import random

from lidar_labeler.monocular import evaluate, fit, samples_from_sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", default="dataset_out/distances.json")
    parser.add_argument("--out", default="distance_model.json")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = samples_from_sidecar(args.sidecar)
    if len(samples) < 30:
        raise SystemExit(f"only {len(samples)} samples - export more frames first")

    random.Random(args.seed).shuffle(samples)
    split = int(len(samples) * (1 - args.holdout))
    train, test = samples[:split], samples[split:]

    model = fit(train)
    model.save(args.out)

    print(f"\nfitted on {len(train)} labels, held out {len(test)}")
    print(f"\n{'class':<14}{'n':>7}{'coefficient':>14}{'implied height':>16}")
    print("-" * 51)
    for name, count in sorted(model.n_samples.items(), key=lambda kv: -kv[1]):
        coefficient = model.coefficients.get(name)
        if coefficient is None:
            print(f"{name:<14}{count:>7}{'(fallback)':>14}{'':>16}")
        else:
            # Coefficient is f*H, so dividing by a plausible focal length recovers
            # the object height the data implies. A car landing near 1.5 m is a
            # strong sign the whole chain is sound.
            print(f"{name:<14}{count:>7}{coefficient:>14.0f}{coefficient/1250:>15.2f}m")

    print("\nHELD-OUT PERFORMANCE (camera only, no LiDAR)")
    for key, value in evaluate(model, test).items():
        print(f"  {key:<10}{value:.2f}" if isinstance(value, float) else f"  {key:<10}{value}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
