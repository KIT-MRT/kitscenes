"""Synthetic scenes in the release format, for tests.

Timestamps are seconds since the scene start (9 decimals in
``timestamp.reference.txt`` and ``poses.txt``, as in v2.0.0), per-point times share that
clock, sensor files are named by frame index, and ``calib.json`` follows the
released schema.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from kitscenes.rig import Camera

START_NS = 46_000_000  # frame 0 about 46 ms after the scene start, as in the release
STEP_NS = 100_000_000  # 10 Hz


def seconds(ns: int, decimals: int = 4) -> str:
    """Exact decimal seconds string of a non-negative nanosecond count."""
    return f"{ns // 10**9}.{ns % 10**9:09d}"[: (-9 + decimals) or None]


def frame_ns(index: int) -> int:
    return START_NS + index * STEP_NS


def write_timestamps(scene: Path, n: int) -> None:
    (scene / "timestamp.reference.txt").write_text("".join(seconds(frame_ns(i), 9) + "\n" for i in range(n)))


def write_poses(
    scene: Path, n: int, *, speed: float = 10.0, yaw_rate: float = 0.0, origin: Sequence[float] = (100.0, 200.0, 150.0),
) -> None:
    """Poses moving at *speed* m/s along the heading, turning at *yaw_rate* rad/s."""
    lines = []
    x, y, z = origin
    for i in range(n):
        t = i * STEP_NS / 1e9
        yaw = yaw_rate * t
        if yaw_rate:
            px = x + speed / yaw_rate * np.sin(yaw)
            py = y + speed / yaw_rate * (1 - np.cos(yaw))
        else:
            px, py = x + speed * t, y
        lines.append(
            f"{seconds(frame_ns(i), 9)} {px:.6f} {py:.6f} {z:.6f} 0.0 0.0 {np.sin(yaw / 2):.9f} {np.cos(yaw / 2):.9f}"
        )
    (scene / "poses.txt").write_text("\n".join(lines) + "\n")


def transform(translation: Sequence[float] = (0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> list[list[float]]:
    yaw = np.deg2rad(yaw_deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:3, 3] = translation
    return T.tolist()


def camera_entry(width: int = 64, height: int = 48, focal: float = 50.0, T: Optional[list] = None) -> dict:
    """A pinhole camera looking along base +x (OpenCV: z forward, x right, y down)."""
    if T is None:
        T = [[0, 0, 1, 1.5], [-1, 0, 0, 0], [0, -1, 0, 1.6], [0, 0, 0, 1]]
    return {
        "T_to_reference": T,
        "camera_model": "pinhole",
        "intrinsics": {"focal_length": focal, "principal_point_u": width / 2, "principal_point_v": height / 2},
        "resolution": {"width": width, "height": height},
    }


def write_calib(scene: Path, entries: dict) -> None:
    (scene / "calibration").mkdir(parents=True, exist_ok=True)
    calib = {"reference": "base_frame", **entries}
    (scene / "calibration" / "calib.json").write_text(json.dumps(calib))


def write_images(scene: Path, camera: str, indices: Iterable[int], width: int = 64, height: int = 48) -> None:
    from PIL import Image

    (scene / camera).mkdir(parents=True, exist_ok=True)
    for i in indices:
        pixels = np.full((height, width, 3), (i * 10 % 256, 128, 64), dtype=np.uint8)
        pixels[: height // 2, : width // 2] = 255  # an asymmetric block to check orientation
        Image.fromarray(pixels).save(scene / camera / f"{i:010d}.jpg", quality=95)


def write_lidar(
    scene: Path,
    lidar: str,
    indices: Iterable[int],
    *,
    n: int = 20,
    zero_rows: Sequence[int] = (),
    sweep_s: float = 0.0,
    no_return_rows: Sequence[int] = (),
) -> None:
    """Lidar parquet files as stored: 5 mm integer grid, per-point times on the reference clock.

    Args:
        zero_rows: Slots without a point (xyz = 0, reflectivity 0), kept by the loader.
        sweep_s: Spread the point times over this long after the frame time.
        no_return_rows: No-return rows (xyz = 0, reflectivity -1), dropped by the loader.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    (scene / lidar).mkdir(parents=True, exist_ok=True)
    for i in indices:
        rng = np.random.RandomState(i)
        x = rng.randint(1000, 4000, n).astype(np.int32)  # 5 to 20 m ahead
        y = rng.randint(-1000, 1000, n).astype(np.int32)
        z = rng.randint(-200, 200, n).astype(np.int32)
        refl = rng.uniform(1, 100, n).astype(np.float32)
        for rows, value in ((zero_rows, 0.0), (no_return_rows, -1.0)):
            rows = list(rows)
            x[rows], y[rows], z[rows], refl[rows] = 0, 0, 0, value
        t = frame_ns(i) / 1e9 + np.linspace(0.0, sweep_s, n)
        table = pa.Table.from_pydict({
            "x": x, "y": y, "z": z, "reflectivity": refl, "timestamp": t,
            "ring": np.arange(n, dtype=np.uint8) % 32,
        })
        meta = {"discretization_resolution": "0.005", "point_type_name": "PointKITScenes"}
        pq.write_table(table.cast(table.schema.with_metadata(meta)), scene / lidar / f"{i:010d}.parquet")


def write_organized_lidar(
    scene: Path,
    lidar: str,
    index: int,
    *,
    sensor_id: int,
    rings: int = 4,
    columns: int = 5,
    no_return_slots: Sequence[tuple[int, int]] = (),
    empty_slots: Sequence[tuple[int, int]] = (),
) -> np.ndarray:
    """An organized sweep as the release stores it; returns the per-row (ring, column, return) table.

    ``sensor_id`` 7 (OT128) or 1/2 (XT32): column-major with two returns,
    ``row = (column * 2 + return) * rings + ring``, ``pcl_width = 2 * columns``;
    0 (Velodyne): ring-major, ``row = ring * columns + column``. Every point
    lies on +x at ``0.005 * (2000 + 1000 * ring + 2 * column + return)`` m, so its
    range names its slot; its time is the frame time + 0.1 ms per column.
    ``no_return_slots`` / ``empty_slots`` are (ring, column) slots stored as
    (0, 0, 0) with reflectivity -1 / 0 (both returns for the Hesai).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    hesai = sensor_id != 0
    returns = 2 if hesai else 1
    n = rings * columns * returns
    rows = np.arange(n)
    if hesai:
        ring, column, ret = rows % rings, rows // rings // 2, (rows // rings) % 2
    else:
        ring, column, ret = rows // columns, rows % columns, np.zeros(n, dtype=np.int64)
    x = (2000 + 1000 * ring + 2 * column + ret).astype(np.int32)
    refl = np.full(n, 0.5, np.float32)
    t = frame_ns(index) / 1e9 + column * 1e-4
    for slots, value in ((no_return_slots, -1.0), (empty_slots, 0.0)):
        for r, c in slots:
            hit = (ring == r) & (column == c)
            x[hit], refl[hit] = 0, value
            if not hesai and value == 0.0:
                t[hit] = 0.0  # empty Velodyne grid cells have time 0
    table = pa.Table.from_pydict({
        "x": x, "y": np.zeros(n, np.int32), "z": np.zeros(n, np.int32), "reflectivity": refl, "timestamp": t,
        "ring": ring.astype(np.uint8), "sensor_id": np.full(n, sensor_id, np.uint8),
        "sensor_specific_data": np.where(ret == 1, 256, 0).astype(np.uint16),
    })
    meta = {"discretization_resolution": "0.005000", "point_type_name": "PointKITScenes",
            "pcl_width": str(columns * returns), "pcl_height": str(rings)}
    (scene / lidar).mkdir(parents=True, exist_ok=True)
    pq.write_table(table.cast(table.schema.with_metadata(meta)), scene / lidar / f"{index:010d}.parquet")
    return np.column_stack([ring, column, ret])


def write_radar(
    scene: Path, radar: str, indices: Iterable[int], *, n: int = 5, overrides: Optional[dict] = None,
) -> None:
    """Radar parquet files; clouds are captured 19 ms before the frame time, as in the release."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (scene / radar).mkdir(parents=True, exist_ok=True)
    for i in indices:
        rng = np.random.RandomState(i + 1000)
        x = rng.uniform(5, 50, n).astype(np.float32)
        y = rng.uniform(-10, 10, n).astype(np.float32)
        z = rng.uniform(-1, 1, n).astype(np.float32)
        rng_ = np.sqrt(x**2 + y**2 + z**2).astype(np.float32)
        cols = {
            "x": x, "y": y, "z": z,
            "range": rng_, "azimuth": np.arctan2(y, x).astype(np.float32),
            "elevation": np.arcsin(z / rng_).astype(np.float32),
            "range_rate": rng.uniform(-5, 5, n).astype(np.float32),
            "rcs": rng.uniform(-10, 30, n).astype(np.float32),
            "timestamp": np.full(n, frame_ns(i) / 1e9 - 0.019),
        }
        if overrides and i in overrides:
            cols.update(overrides[i])
        table = pa.Table.from_pydict(cols)
        meta = {"point_type_name": "PointKITScenesRadar"}
        pq.write_table(table.cast(table.schema.with_metadata(meta)), scene / radar / f"{i:010d}.parquet")


def write_ground_labels(scene: Path, lidar: str, index: int, is_ground: Sequence[bool]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = scene / "processed" / "ground_segmentation" / lidar
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict({"isground": list(is_ground)}), folder / f"{index:010d}.parquet")


# One northbound road lanelet around the local origin (49.0144 N, 8.4172 E).
MAP_OSM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <osm version="0.6">
      <node id="1" lat="49.01440" lon="8.41720" />
      <node id="2" lat="49.01450" lon="8.41720" />
      <node id="3" lat="49.01440" lon="8.41725" />
      <node id="4" lat="49.01450" lon="8.41725" />
      <way id="10"><nd ref="1" /><nd ref="2" /><tag k="type" v="line_thin" /><tag k="subtype" v="solid" /></way>
      <way id="11"><nd ref="3" /><nd ref="4" /><tag k="type" v="line_thin" /><tag k="subtype" v="solid" /></way>
      <relation id="100">
        <member type="way" ref="10" role="left" /><member type="way" ref="11" role="right" />
        <tag k="type" v="lanelet" /><tag k="subtype" v="road" /><tag k="location" v="urban" />
      </relation>
    </osm>
""")


def write_map(scene: Path) -> None:
    (scene / "maps").mkdir(parents=True, exist_ok=True)
    (scene / "maps" / "map.osm").write_text(MAP_OSM)
    (scene / "maps" / "origin.json").write_text(json.dumps({"latitude": 49.0144, "longitude": 8.4172}))


def make_scene(
    path: Path,
    *,
    n: int = 3,
    cameras: Sequence[str] = ("camera_ring_front",),
    lidars: Sequence[str] = ("lidar_top",),
    radars: Sequence[str] = ("radar_front",),
    calib_only: Sequence[str] = (),
    poses: bool = True,
    maps: bool = False,
    lidar_T: Optional[list] = None,
    radar_T: Optional[list] = None,
) -> Path:
    """A complete scene folder.

    Args:
        calib_only: Sensors listed in ``calib.json`` without data (not downloaded).
        lidar_T, radar_T: Extrinsics (default: lidar 1.2 m up, radar at the front bumper).
    """
    path.mkdir(parents=True, exist_ok=True)
    write_timestamps(path, n)
    if poses:
        write_poses(path, n)
    entries: dict = {}
    for name in [*cameras, *lidars, *radars, *calib_only]:
        if name.startswith("camera"):
            entries[name] = camera_entry()
        elif name.startswith("lidar"):
            entries[name] = {"T_to_reference": lidar_T or transform((0.0, 0.0, 1.2))}
        else:
            entries[name] = {"T_to_reference": radar_T or transform((3.5, 0.0, 0.5))}
    write_calib(path, entries)
    for cam in cameras:
        write_images(path, cam, range(n))
    for lidar in lidars:
        write_lidar(path, lidar, range(n))
    for radar in radars:
        write_radar(path, radar, range(n))
    if maps:
        write_map(path)
    return path


def make_dataset(root: Path, scenes: dict[str, Sequence[str]], **scene_kwargs) -> Path:
    """``{split: [scene_id, ...]}`` under ``root/data``."""
    for split, ids in scenes.items():
        for scene_id in ids:
            make_scene(root / "data" / split / scene_id, **scene_kwargs)
    return root


def pack_scene(scene: Path) -> Path:
    """Turn a scene folder into the release archive format in place (:func:`kitscenes.pack.pack_scene`)."""
    import os
    import shutil

    from kitscenes.pack import pack_scene as pack

    staging = scene.parent / f".packing-{scene.name}"
    pack(scene, staging / scene.name)
    for hidden in (p for p in scene.iterdir() if p.name.startswith(".")):  # the API's bookkeeping is not packed
        os.replace(hidden, staging / scene.name / hidden.name)
    shutil.rmtree(scene)
    os.replace(staging / scene.name, scene)
    staging.rmdir()
    return scene


def write_messages(scene: Path, device: str, topic: str, messages: list[dict], *,
                   frames: dict[int, int] | None = None, t_sync_s: list[float] | None = None,
                   index: list | None = None, metadata: dict | None = None) -> Path:
    """One GNSS/INS topic as the release stores it: ``<device>/<topic>.parquet`` (:mod:`kitscenes.messages`).

    *messages* are nested dicts (flattened to dotted columns, ``nan`` strings to
    NaN); *frames* maps a message number to the frame it is assigned to;
    *index* replaces the message numbers (None for a row without one),
    *metadata* is added to the schema metadata.
    """
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    special = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}

    def flatten(value, prefix, out):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f"{prefix}.{key}" if prefix else key, out)
        elif isinstance(value, list):
            out[prefix] = [special.get(v, v) if isinstance(v, str) else v for v in value]
        else:
            out[prefix] = special.get(value, value) if isinstance(value, str) else value

    rows = []
    for message in messages:
        flat: dict = {}
        flatten(message, "", flat)
        rows.append(flat)
    names = list(dict.fromkeys(k for row in rows for k in row))
    t_sync_s = t_sync_s or [0.0462 + 0.02 * k for k in range(len(messages))]
    frames = frames or {}
    table = pa.table({
        "index": pa.array(index if index is not None else range(len(messages)), pa.int64()),
        "t_ns": pa.array([round(row["header.stamp"] * 1e9) for row in rows], pa.int64()),
        "t_sync_ns": pa.array([round(t * 1e9) for t in t_sync_s], pa.int64()),
        "frame_index": pa.array([frames.get(k) for k in range(len(messages))], pa.int32()),
        **{name: pa.array([row.get(name) for row in rows]) for name in names},
    }).replace_schema_metadata({"kitscenes_format": "gnss_topic_v1", "device": device, "topic": topic, **(metadata or {})})
    (scene / device).mkdir(parents=True, exist_ok=True)
    pq.write_table(table, scene / device / f"{topic}.parquet", compression="zstd")
    return scene / device / f"{topic}.parquet"


def imu_message(t: float, gyro_z: float) -> dict:
    """One IMU message (IMU axes: z down)."""
    return {
        "header": {"seq": 1, "stamp": round(t, 9), "frame_id": "imu"},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "orientation_covariance": [0.0] * 9,
        "angular_velocity": {"x": 0.1, "y": 0.2, "z": gyro_z},
        "linear_acceleration": {"x": 0.0, "y": 0.0, "z": -9.81},
    }


# World points that do not move; the car drives along x at 10 m/s from (100, 200, 150).
STATIC = np.array([[120.0, 205.0, 151.0], [130.0, 195.0, 149.5], [140.0, 200.0, 150.0]])


def write_static_lidar(scene: Path, n: int, points: np.ndarray = STATIC) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (scene / "lidar_top").mkdir(exist_ok=True)
    for i in range(n):
        car = np.array([100.0 + 1.0 * i, 200.0, 150.0])
        sensor = points - car - [0.0, 0.0, 1.2]  # lidar 1.2 m above the base origin, no rotation
        sensor = np.vstack([sensor, np.zeros(3)])  # plus one row without a point
        grid = np.round(sensor / 0.005).astype(np.int32)
        table = pa.Table.from_pydict({
            "x": grid[:, 0], "y": grid[:, 1], "z": grid[:, 2],
            "reflectivity": np.array([10, 20, 30, 0], np.float32),
            "timestamp": np.full(4, frame_ns(i) / 1e9),
        })
        meta = {"discretization_resolution": "0.005", "point_type_name": "PointKITScenes"}
        pq.write_table(table.cast(table.schema.with_metadata(meta)), scene / "lidar_top" / f"{i:010d}.parquet")


def encode_rle(mask: np.ndarray) -> str:
    """COCO compressed RLE (pycocotools' rleToString) of an (H, W) bool mask."""
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


def instances_file(masks_scores_classes, size=(48, 64)) -> str:
    """An instance-mask file of (mask, score, class) triples, as the release stores it."""
    h, w = size
    insts = []
    for k, (mask, score, cls) in enumerate(masks_scores_classes):
        ys, xs = np.nonzero(mask)
        insts.append({"id": k, "class": cls, "objectness": score,
                      "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                      "mask_rle": encode_rle(mask)})
    return json.dumps({"image_info": {"rel_path": "0000003040.jpg", "original_size": [h, w]}, "instances": insts})


def real_camera():
    """A ring-camera-like pinhole at 1.6 m, looking along base +x: focal 1000 px, 2000 x 1200."""
    T = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.6], [0, 0, 0, 1]], float)
    K = np.array([[1000.0, 0, 1000.0], [0, 1000.0, 600.0], [0, 0, 1]])
    return Camera("camera_ring_front", T, K=K, image_size=(2000, 1200))
