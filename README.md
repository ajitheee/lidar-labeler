# lidar-labeler

Auto-label camera images using a LiDAR that's already on the vehicle. The
detector names the object; the point cloud measures it. No human annotation.

This is the labeling half of the pipeline — the part that produces a dataset
where every 2D box carries a ground-truth distance in metres.

## Why the distance extraction isn't just a median

A 2D box doesn't tightly enclose its object, so the points inside it are a
mixture of the object's surface, background behind it, and occluders in front.
On the synthetic test scene (a car at 14.7 m with a wall 40 m behind it):

| method | estimate | error |
|---|---|---|
| median of all points in box | 25.42 m | **10.70 m** |
| nearest dominant cluster | 14.72 m | **0.00 m** |

The naive answer isn't slightly worse, it's unusable — and it fails *silently*,
producing a plausible number that poisons every model trained on it. Hence
`depth.py`: shrink the box, find the nearest substantial depth cluster, and
refuse to answer when the evidence is thin.

Every label comes back with `n_points`, `inlier_fraction` and `spread`, and an
`is_reliable` gate. **Drop unreliable boxes rather than guessing.** Coverage is
cheap to recover by processing more frames; a corrupted label is not.

## Datasets

Works with **nuScenes v1.0-mini** (~4 GB, recommended) or **KITTI** object
detection (~41 GB). Only the calibration layer differs; the projection and
distance code is shared.

```bash
pip install nuscenes-devkit
python run_nuscenes.py overlay  --root /data/nuscenes
python run_nuscenes.py validate --root /data/nuscenes --limit 200
python run_nuscenes.py export   --root /data/nuscenes --out dataset_out
```

### Why nuScenes needs four transforms and KITTI needs one

KITTI's sensors are hardware-synchronised, so one static LiDAR-to-camera
extrinsic is correct. nuScenes' are not: the LiDAR sweep and the camera shutter
happen at different moments and the car moves in between. Projecting with a
single static extrinsic produces an overlay that looks *almost* right - points
smeared slightly along the direction of travel, worst at speed - and biases
every distance label you generate.

So the chain routes through global coordinates:

```
LiDAR -> ego@lidar_time -> global -> ego@camera_time -> camera -> image
```

`test_nuscenes.py` asserts this directly: with identical ego poses the chain
must collapse to the static case, and with 0.5 m of ego motion that 0.5 m must
appear in the transform. Two other traps it covers: nuScenes quaternions are
`[w, x, y, z]` while most libraries expect `[x, y, z, w]`, and nuScenes sweeps
have five columns where KITTI has four - reshaping to `(-1, 4)` doesn't error,
it just silently scrambles every point.

## Quickstart

```bash
pip install numpy opencv-python
python tests/test_synthetic.py          # verifies the math, no data needed

# 1. Look at the overlay FIRST. Points must land on cars, not beside them.
python demo.py overlay  --root /path/to/kitti --frame 000008

# 2. Score LiDAR-derived distances against KITTI's own 3D annotations.
python demo.py validate --root /path/to/kitti --limit 200
```

`validate` gives you a real accuracy number on day one, before any detector is
wired up. That number is also the credibility plot for your writeup.

Expects the KITTI object-detection layout:

```
root/image_2/000000.png   root/velodyne/000000.bin
root/calib/000000.txt     root/label_2/000000.txt
```

## Two things that will bite you

**Near face vs. centre.** KITTI's `location` is the *centre* of the 3D box.
LiDAR only sees the surface facing the camera, so your labels read about half an
object-length closer. Compare against `label.near_face_depth`, not
`center_depth`, or you'll spend an hour chasing a bias you invented. A constant
non-zero `bias` in the validate output usually means a convention mismatch, not
a broken pipeline.

**Points behind the camera.** They must be dropped *before* the perspective
divide. Divide by a negative depth and they reproject to mirrored pixels that
land somewhere entirely believable. `project_to_image` handles this; if you
reimplement it, don't skip the check.

## Carrying distance into Roboflow

No annotation format has a "distance" field, so `dataset.py` writes two things:

- `_annotations.coco.json` - clean class names, what Roboflow ingests
- `distances.json` - exact metres per box, what you fit the distance model on

Keep both. Importers are entitled to drop fields they don't recognise, so the
sidecar is the copy guaranteed to survive.

`--binned-classes` offers a third route: fold distance into the class name
(`car_10-20m`). RF-DETR then learns distance with no custom head at all. Worth
running as a comparison; not worth making your primary dataset, since it
entangles the detection metric with the distance metric.

```bash
python upload.py --root /path/to/kitti --limit 500            # local export
python upload.py --root /path/to/kitti --project my-project   # and upload
```

## Layout

```
lidar_labeler/
  calib.py       KITTI calib parsing, composed P2 · R0_rect · Tr_velo_to_cam
  projection.py  Velodyne -> image, with FOV and depth filtering
  depth.py       robust per-box distance + reliability gating
  overlay.py     visual sanity checks
  io_kitti.py    .bin scans and label_2 parsing
  dataset.py     detector-agnostic labelling + COCO/sidecar export
  transforms.py  quaternion and rigid-transform helpers
  nuscenes_adapter.py  nuScenes -> the same calibration interface
tests/
  test_synthetic.py   real KITTI calibration, known geometry, exact assertions
  test_dataset.py     export round-trips, bin boundaries, filtering
  test_nuscenes.py    transform chain, quaternion convention, motion compensation
demo.py          overlay | validate
upload.py        KITTI: label a split and push it to Roboflow
run_nuscenes.py  nuScenes: overlay | validate | export
```

## Next

Swap KITTI's boxes for zero-shot ones (Autodistill / Grounding DINO / SAM 3) and
the pipeline becomes fully unsupervised: nothing human-labeled anywhere. Then
train RF-DETR on the result, and fit camera-only distance so the LiDAR can be
removed at inference time.
