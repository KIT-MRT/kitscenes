"""Small scenes in the KITScenes release format, for tests that read through `kitscenes`."""

import json
from pathlib import Path

import numpy as np

FRAME_S = 0.1
SPEED = 10.0                      # m/s along +x
START = np.array([100.0, 200.0, 150.0])
ORIGIN = (49.0144, 8.4172)
# A pinhole camera at the base origin, looking along +x (OpenCV axes), 64 x 48 px.
CAMERA_T = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]


def car_at(index):
    return START + [SPEED * FRAME_S * index, 0.0, 0.0]


def encode_rle(mask):
    """COCO compressed RLE of an (H, W) bool mask (pycocotools' rleToString)."""
    flat = mask.reshape(-1, order="F").astype(np.uint8)
    counts, current, run = [], 0, 0
    for v in flat:
        if v != current:
            counts.append(run)
            current, run = v, 0
        run += 1
    counts.append(run)
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def write_scene(root, points_of, frames=12, *, split="train", scene_id="s", camera_rgb=None, mask_class_of=None):
    """A scene folder: `points_of(i)` gives frame i's lidar points in the scene frame.

    The car drives along +x; `lidar_top` sits at the base origin. With
    `camera_rgb`, `camera_ring_front` sees a plain image of that colour; with
    `mask_class_of(i)`, it also has one seamseg instance covering its whole
    image (class None: a mask file without instances).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = Path(root) / "data" / split / scene_id
    (d / "maps").mkdir(parents=True)
    (d / "maps" / "origin.json").write_text(json.dumps({"latitude": ORIGIN[0], "longitude": ORIGIN[1]}))
    (d / "timestamp.reference.txt").write_text("".join("%.4f\n" % (i * FRAME_S) for i in range(frames)))
    (d / "poses.txt").write_text("".join(
        "%.9f %.4f %.4f %.4f 0 0 0 1\n" % (i * FRAME_S, *car_at(i)) for i in range(frames)))
    calib = {"reference": "base_frame", "lidar_top": {"T_to_reference": np.eye(4).tolist()}}
    if camera_rgb is not None:
        calib["camera_ring_front"] = {
            "T_to_reference": CAMERA_T, "camera_model": "pinhole",
            "intrinsics": {"focal_length": 50.0, "principal_point_u": 32.0, "principal_point_v": 24.0},
            "resolution": {"width": 64, "height": 48},
        }
    (d / "calibration").mkdir()
    (d / "calibration" / "calib.json").write_text(json.dumps(calib))
    (d / "lidar_top").mkdir()
    meta = {"discretization_resolution": "0.005", "point_type_name": "PointKITScenes"}
    for i in range(frames):
        local = np.asarray(points_of(i), np.float64) - car_at(i)
        grid = np.round(local / 0.005).astype(np.int32)
        table = pa.Table.from_pydict({
            "x": grid[:, 0], "y": grid[:, 1], "z": grid[:, 2],
            "reflectivity": np.full(len(grid), 0.5, np.float32),
            "timestamp": np.full(len(grid), i * FRAME_S),
            "sensor_id": np.zeros(len(grid), np.uint8),
        })
        pq.write_table(table.cast(table.schema.with_metadata(meta)), d / "lidar_top" / ("%010d.parquet" % i))
    if camera_rgb is not None:
        from PIL import Image
        (d / "camera_ring_front").mkdir()
        for i in range(frames):
            Image.new("RGB", (64, 48), tuple(camera_rgb)).save(d / "camera_ring_front" / ("%010d.jpg" % i), quality=95)
    if mask_class_of is not None:
        folder = d / "processed" / "seamseg_instances" / "camera_ring_front"
        folder.mkdir(parents=True)
        full = encode_rle(np.ones((48, 64), bool))
        for i in range(frames):
            cls = mask_class_of(i)
            instances = [] if cls is None else [
                {"id": 0, "class": cls, "objectness": 0.9, "bbox": [0, 0, 63, 47], "mask_rle": full}]
            (folder / ("%010d.json" % i)).write_text(json.dumps(
                {"image_info": {"original_size": [48, 64]}, "instances": instances}))
    return d
