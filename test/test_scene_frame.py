"""Scene metadata and availability; Frame accessors on synthetic release-format scenes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes import Availability, LabelsUnavailable, MapUnavailable, PoseUnavailable, Scene, SensorUnavailable
from kitscenes.errors import CalibrationError
from release_fixture import (
    STEP_NS,
    camera_entry,
    frame_ns,
    make_scene,
    transform,
    write_calib,
    write_ground_labels,
    write_lidar,
    write_poses,
    write_radar,
)


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    return make_scene(
        tmp_path / "data" / "train" / "scene_a",
        cameras=("camera_ring_front",), lidars=("lidar_top", "lidar_front"), radars=("radar_front",),
        calib_only=("camera_ring_rear",),
    )


@pytest.fixture
def scene(scene_dir: Path) -> Scene:
    return Scene(scene_dir, split="train")


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class TestScene:
    def test_metadata(self, scene: Scene) -> None:
        assert scene.id == "scene_a" and scene.split_info.has_maps
        assert scene.timestamps_ns.tolist() == [46_000_000, 146_000_000, 246_000_000]
        assert not scene.timestamps_ns.flags.writeable
        assert len(scene) == 3 and scene.duration_s == pytest.approx(0.2)
        assert [p.t_ns for p in scene.track] == scene.timestamps_ns.tolist()
        assert scene.pose_frame == "local_utm"
        assert scene.availability("pose") is Availability.PRESENT
        assert scene.sensors == ("camera_ring_front", "lidar_top", "lidar_front", "radar_front")
        assert set(scene.rig.cameras) == {"camera_ring_front", "camera_ring_rear"}

    def test_frame_token(self, scene: Scene) -> None:
        assert [f.token for f in scene] == ["scene_a_0000", "scene_a_0001", "scene_a_0002"]
        assert [f.token.rsplit("_", 1)[0] for f in scene] == ["scene_a"] * 3  # the scene id splits off

    def test_availability(self, scene: Scene, scene_dir: Path) -> None:
        (scene_dir / "lidar_top" / "0000000001.parquet").unlink()
        assert scene.availability("lidar_top") is Availability.PRESENT
        assert scene.availability("lidar_top", 0) is Availability.PRESENT
        assert scene.availability("lidar_top", 1) is Availability.DROPPED
        assert scene.availability("camera_ring_rear", 0) is Availability.NOT_DOWNLOADED
        assert scene.availability("lidar_rear", 0) is Availability.NOT_DOWNLOADED  # not even in calib.json
        assert scene.frame_indices("lidar_top").tolist() == [0, 2]
        with pytest.raises(IndexError):
            scene.availability("lidar_top", 3)
        with pytest.raises(KeyError):
            scene.availability("lidar_roof")

    def test_without_poses(self, scene_dir: Path) -> None:
        (scene_dir / "poses.txt").unlink()
        scene = Scene(scene_dir)
        assert len(scene.track) == 0
        assert scene.pose_frame is None
        with pytest.raises(PoseUnavailable, match="scene_a/poses.txt") as err:
            scene.frame(0).pose
        assert err.value.state is Availability.NOT_DOWNLOADED
        assert scene.availability("pose") is Availability.NOT_DOWNLOADED
        with pytest.raises(PoseUnavailable):
            scene.ego(0)

    def test_pose_frame_outside_a_known_split(self, tmp_path: Path) -> None:
        scene_dir = make_scene(tmp_path / "s")
        write_poses(scene_dir, 3, origin=(0.0, 0.0, 0.0))
        assert Scene(scene_dir).pose_frame is None  # no release folder says which
        assert Scene(scene_dir, split="test-e2e").pose_frame == "first_pose"
        assert Scene(scene_dir, split="train").pose_frame == "local_utm"

    def test_map_unavailable_reasons(self, tmp_path: Path) -> None:
        withheld = Scene(make_scene(tmp_path / "t"), split="test")
        with pytest.raises(MapUnavailable) as err:
            withheld.map
        assert err.value.state is Availability.WITHHELD_BY_SPLIT and "test split ships no maps" in str(err.value)
        assert withheld.availability("map") is Availability.WITHHELD_BY_SPLIT
        missing = Scene(make_scene(tmp_path / "v"), split="val")
        with pytest.raises(MapUnavailable) as err:
            missing.map
        assert err.value.state is Availability.NOT_DOWNLOADED
        with pytest.raises(MapUnavailable):
            missing.origin

    def test_map_loads(self, tmp_path: Path) -> None:
        pytest.importorskip("lanelet2")
        scene = Scene(make_scene(tmp_path / "m", maps=True), split="train")
        assert len(scene.map.lanelet_map.laneletLayer) == 1
        assert scene.map is scene.map
        clone = pickle.loads(pickle.dumps(scene))
        assert "_map" not in clone.__dict__

    def test_pickle(self, scene: Scene) -> None:
        scene.rig
        clone = pickle.loads(pickle.dumps(scene))
        assert (clone.id, clone.split, clone.num_frames) == ("scene_a", "train", 3)

    def test_frames(self, scene: Scene) -> None:
        assert [f.index for f in scene] == [0, 1, 2]
        assert scene.frame(-1).index == 2
        with pytest.raises(IndexError):
            scene.frame(3)


# ---------------------------------------------------------------------------
# Frame: time, pose, cameras
# ---------------------------------------------------------------------------


class TestFrameBasics:
    def test_time_and_pose(self, scene: Scene) -> None:
        f = scene.frame(1)
        assert f.t_ns == 146_000_000 and f.t_s == pytest.approx(0.146)
        np.testing.assert_allclose(f.pose.xyz, [101.0, 200.0, 150.0], atol=1e-6)  # 10 m/s for 0.1 s
        assert f.future(horizon_s=1.0).xyz_ego[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
        np.testing.assert_allclose(scene.ego(f.t_ns).T_world_base, f.pose.T_world_base, atol=1e-9)
        np.testing.assert_allclose(f.T_world_base, f.pose.T_world_base)
        assert scene.ego([0, f.t_ns]).T_world_base.shape == (2, 4, 4)
        assert f.sensors() == ("camera_ring_front", "lidar_top", "lidar_front", "radar_front")
        assert f.sensors("lidar") == ("lidar_top", "lidar_front")

    def test_image(self, scene: Scene) -> None:
        f = scene.frame(0)
        img = f.image("camera_ring_front")
        assert img.shape == (48, 64, 3) and img.dtype == np.uint8 and img.flags.writeable
        assert img[5, 5].min() > 240 and img[40, 60].max() < 200  # not flipped
        small = f.image("camera_ring_front", reduce=2)
        assert small.shape == (24, 32, 3)
        assert f.image_bytes("camera_ring_front")[:2] == b"\xff\xd8"
        with pytest.raises(ValueError):
            f.image("camera_ring_front", reduce=3)

    def test_missing_sensor_is_typed(self, scene: Scene) -> None:
        f = scene.frame(0)
        with pytest.raises(SensorUnavailable) as err:
            f.image("camera_ring_rear")
        assert err.value.state is Availability.NOT_DOWNLOADED and err.value.what == "camera_ring_rear"
        assert f.image("camera_ring_rear", missing="none") is None
        with pytest.raises(SensorUnavailable, match="--components lidar_rear") as err:
            f.lidar("lidar_rear")  # neither data nor a calib.json entry: still a partial download
        assert err.value.state is Availability.NOT_DOWNLOADED
        assert isinstance(err.value, LookupError)
        with pytest.raises(KeyError, match="not a lidar"):
            f.lidar("radar_front")

    def test_payloads_are_fresh_without_cache(self, scene: Scene) -> None:
        f = scene.frame(0)
        a, b = f.lidar("lidar_top"), f.lidar("lidar_top")
        assert a is not b and a.xyz.flags.writeable
        a.xyz[:] = 1e6
        assert f.lidar("lidar_top").xyz.max() < 1e3

    def test_cache_shares_read_only_payloads(self, scene_dir: Path) -> None:
        ds = kitscenes.open(scene_dir.parents[2], cache_mb=64)
        f = ds[0].frame(0)
        a, b = f.lidar("lidar_top"), f.lidar("lidar_top")
        assert a is b and not a.xyz.flags.writeable
        with pytest.raises(ValueError):
            a.xyz[0] = 0
        img = f.image("camera_ring_front")
        assert img is f.image("camera_ring_front") and not img.flags.writeable
        assert f.lidar("lidar_top", frame="sensor") is not a


# ---------------------------------------------------------------------------
# Lidar
# ---------------------------------------------------------------------------


def _set_lidar_extrinsic(scene_dir: Path, name: str, translation, yaw_deg: float = 0.0) -> None:
    path = scene_dir / "calibration" / "calib.json"
    calib = json.loads(path.read_text())
    calib[name] = {"T_to_reference": transform(translation, yaw_deg)}
    path.write_text(json.dumps(calib))


class TestLidar:
    def test_base_frame_deskewed(self, scene: Scene) -> None:
        cloud = scene.frame(1).lidar("lidar_top")
        raw = scene.frame(1).lidar("lidar_top", frame="sensor", deskew=False)
        assert cloud.frame == "base" and cloud.motion_compensated and cloud.reason is None
        assert cloud.sensor == "lidar_top" and cloud.t_ns == frame_ns(1)
        # All points share the frame time and the extrinsic is 1.2 m up.
        np.testing.assert_allclose(cloud.xyz, raw.xyz + [0, 0, 1.2], atol=1e-4)
        assert set(cloud.attrs) == {"reflectivity", "timestamp", "ring", "row"}

    def test_deskew_moves_late_points(self, scene_dir: Path) -> None:
        write_lidar(scene_dir, "lidar_top", [1], n=10, sweep_s=0.09)
        f = Scene(scene_dir).frame(1)
        cloud, raw = f.lidar("lidar_top"), f.lidar("lidar_top", deskew=False)
        assert not raw.motion_compensated and raw.reason == "deskewing disabled" and raw.frame == "base"
        # 10 m/s: a point measured 0.09 s after the frame time is seen 0.9 m further ahead.
        shift = cloud.xyz[:, 0] - raw.xyz[:, 0]
        np.testing.assert_allclose(shift, 10.0 * np.linspace(0, 0.09, 10), atol=2e-3)

    def test_without_poses_points_stay_in_base_frame(self, scene_dir: Path, caplog) -> None:
        _set_lidar_extrinsic(scene_dir, "lidar_top", (1.0, 2.0, 3.0), yaw_deg=90.0)
        (scene_dir / "poses.txt").unlink()
        f = Scene(scene_dir).frame(1)
        cloud = f.lidar("lidar_top")
        raw = f.lidar("lidar_top", frame="sensor")
        assert cloud.frame == "base" and not cloud.motion_compensated and cloud.reason == "no poses.txt"
        np.testing.assert_allclose(cloud.xyz[:, 0], -raw.xyz[:, 1] + 1.0, atol=1e-4)
        np.testing.assert_allclose(cloud.xyz[:, 1], raw.xyz[:, 0] + 2.0, atol=1e-4)
        f.lidar("lidar_front")
        assert caplog.text.count("not motion compensated") == 1  # once per scene

    def test_sensor_frame_deskewed(self, scene_dir: Path) -> None:
        _set_lidar_extrinsic(scene_dir, "lidar_top", (1.0, 2.0, 3.0), yaw_deg=30.0)
        write_lidar(scene_dir, "lidar_top", [1], n=10, sweep_s=0.09)
        f = Scene(scene_dir).frame(1)
        base, sensor = f.lidar("lidar_top"), f.lidar("lidar_top", frame="sensor")
        assert sensor.frame == "sensor" and sensor.motion_compensated
        T = f.scene.rig.lidars["lidar_top"].T_base_sensor
        np.testing.assert_allclose(sensor.xyz @ T[:3, :3].T + T[:3, 3], base.xyz, atol=1e-3)

    def test_measured_rows_and_no_returns(self, scene_dir: Path) -> None:
        write_lidar(scene_dir, "lidar_top", [0], n=10, zero_rows=(2, 5), no_return_rows=(7,))
        f = Scene(scene_dir).frame(0)
        cloud = f.lidar("lidar_top", rows=True)
        assert len(cloud) == 9  # the stored rows without the no-return row
        assert (~cloud.measured).sum() == 2
        np.testing.assert_array_equal(cloud.xyz[~cloud.measured], 0.0)
        points = f.lidar("lidar_top")  # the measured points only
        assert len(points) == 7 and points.measured.all()
        np.testing.assert_array_equal(points.xyz, cloud.compact().xyz)
        np.testing.assert_array_equal(points.row, cloud.row[cloud.measured])

    def test_deskew_inputs_match(self, scene_dir: Path) -> None:
        from kitscenes.motion import apply_deskew_inputs

        write_lidar(scene_dir, "lidar_top", [1], n=10, sweep_s=0.09)
        f = Scene(scene_dir).frame(1)
        np.testing.assert_allclose(apply_deskew_inputs(f.lidar_deskew_inputs("lidar_top")),
                                   f.lidar("lidar_top", rows=True).xyz)

    def test_fused(self, scene_dir: Path) -> None:
        _set_lidar_extrinsic(scene_dir, "lidar_front", (3.0, 0.0, 0.5))
        write_lidar(scene_dir, "lidar_front", [0], n=8, zero_rows=(0,))
        f = Scene(scene_dir).frame(0)
        fused = f.fused_lidar()
        assert fused.sensor == ("lidar_top", "lidar_front") and len(fused) == 20 + 7
        assert fused.source_index.tolist() == [0] * 20 + [1] * 7
        np.testing.assert_allclose(fused.xyz[20:], f.lidar("lidar_front").xyz)
        assert f.fused_lidar(["lidar_front"]).sensor == ("lidar_front",)

    def test_frame_timestamp_by_index_with_gaps(self, scene_dir: Path) -> None:
        (scene_dir / "lidar_top" / "0000000001.parquet").unlink()
        (scene_dir / "radar_front" / "0000000000.parquet").unlink()
        scene = Scene(scene_dir)
        assert scene.frame(2).lidar("lidar_top").t_ns == frame_ns(2)
        assert scene.frame(1).radar("radar_front").t_ns == frame_ns(1)

    def test_invalid_calibration_fails_the_rig(self, scene_dir: Path) -> None:
        """An invalid entry is one error when the rig loads, not a sensor left out."""
        calib = json.loads((scene_dir / "calibration" / "calib.json").read_text())
        calib["lidar_top"] = {"no_transform": []}
        (scene_dir / "calibration" / "calib.json").write_text(json.dumps(calib))
        scene = Scene(scene_dir)
        with pytest.raises(CalibrationError, match="calib.json.*lidar_top.*T_to_reference"):
            scene.rig
        f = scene.frame(0)
        for load in (lambda: f.lidar("lidar_top"), lambda: f.lidar("lidar_front"), lambda: f.radar("radar_front")):
            with pytest.raises(CalibrationError):
                load()
        with pytest.raises(CalibrationError):
            f.lidar("lidar_front", missing="none")  # not an Unavailable: missing="none" does not hide it
        assert f.image("camera_ring_front").shape == (48, 64, 3)  # images need no calibration

# ---------------------------------------------------------------------------
# Ground labels
# ---------------------------------------------------------------------------


def _scene_with_labels(tmp_path: Path, offset: int, counts=(20, 23, 21, 25, 22, 24)) -> Path:
    """Sweeps with different point counts; label file i+offset holds the labels of sweep i."""
    scene_dir = make_scene(tmp_path / "s", n=len(counts), cameras=(), radars=())
    for i, n in enumerate(counts):
        write_lidar(scene_dir, "lidar_top", [i], n=n)
        if 0 <= i + offset:
            write_ground_labels(scene_dir, "lidar_top", i + offset, [j % 3 == 0 for j in range(n)])
    return scene_dir


class TestGroundMask:
    def test_aligned_labels(self, tmp_path: Path) -> None:
        scene = Scene(_scene_with_labels(tmp_path, 0), split="train")
        mask = scene.frame(3).ground_mask("lidar_top")
        assert len(mask) == 25 == len(scene.frame(3).lidar("lidar_top"))
        assert mask.tolist() == [j % 3 == 0 for j in range(25)]

    def test_shifted_labels_are_invalid(self, tmp_path: Path) -> None:
        """The release labels are aligned with their sweeps; a shifted file fits no sweep and says so."""
        scene = Scene(_scene_with_labels(tmp_path, 1), split="train")
        with pytest.raises(LabelsUnavailable) as err:
            scene.frame(1).ground_mask("lidar_top")
        assert err.value.state is Availability.INVALID and "20 labels for 23 points" in str(err.value)

    def test_unusable_labels(self, tmp_path: Path) -> None:
        scene_dir = _scene_with_labels(tmp_path, 0)
        for i in range(6):
            write_ground_labels(scene_dir, "lidar_top", i, [True] * 7)
        scene = Scene(scene_dir, split="train")
        with pytest.raises(LabelsUnavailable) as err:
            scene.frame(0).ground_mask("lidar_top")
        assert err.value.state is Availability.INVALID

    def test_one_bad_frame(self, tmp_path: Path) -> None:
        scene_dir = _scene_with_labels(tmp_path, 0)
        write_ground_labels(scene_dir, "lidar_top", 2, [True] * 5)
        scene = Scene(scene_dir, split="train")
        with pytest.raises(LabelsUnavailable) as err:
            scene.frame(2).ground_mask("lidar_top")
        assert err.value.state is Availability.INVALID and "5 labels for 21 points" in str(err.value)
        assert len(scene.frame(3).ground_mask("lidar_top")) == 25

    def test_a_file_without_the_label_column(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        scene_dir = _scene_with_labels(tmp_path, 0)
        labels = scene_dir / "processed" / "ground_segmentation" / "lidar_top" / "0000000001.parquet"
        pq.write_table(pa.Table.from_pydict({"x": [1, 2]}), labels)
        with pytest.raises(RuntimeError, match="isground"):
            Scene(scene_dir, split="train").frame(1).ground_mask("lidar_top")

    def test_states(self, tmp_path: Path) -> None:
        scene_dir = _scene_with_labels(tmp_path, 0)
        (scene_dir / "processed/ground_segmentation/lidar_top/0000000004.parquet").unlink()
        with pytest.raises(LabelsUnavailable) as err:
            Scene(scene_dir, split="train").frame(4).ground_mask("lidar_top")
        assert err.value.state is Availability.DROPPED
        with pytest.raises(LabelsUnavailable) as err:
            Scene(scene_dir, split="test").frame(0).ground_mask("lidar_top")
        assert err.value.state is Availability.WITHHELD_BY_SPLIT
        import shutil

        shutil.rmtree(scene_dir / "processed")
        with pytest.raises(LabelsUnavailable) as err:
            Scene(scene_dir, split="train").frame(0).ground_mask("lidar_top")
        assert err.value.state is Availability.NOT_DOWNLOADED


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------


class TestRadar:
    def test_doppler_and_positions_compensated(self, scene_dir: Path) -> None:
        # A static target straight ahead: the radar reads -10 m/s at 10 m/s ego speed.
        write_radar(scene_dir, "radar_front", [1], n=1, overrides={1: {
            "x": np.array([50.0], np.float32), "y": np.zeros(1, np.float32), "z": np.zeros(1, np.float32),
            "range": np.array([50.0], np.float32), "azimuth": np.zeros(1, np.float32),
            "elevation": np.zeros(1, np.float32), "range_rate": np.array([-10.0], np.float32),
        }})
        cloud = Scene(scene_dir).frame(1).radar("radar_front")
        assert cloud.frame == "base" and cloud.motion_compensated and cloud.doppler_compensated
        np.testing.assert_allclose(cloud["range_rate"], [0.0], atol=1e-3)
        # Captured 19 ms early: the target is 0.19 m closer at the frame time; radar at x = 3.5.
        np.testing.assert_allclose(cloud.xyz[0], [53.5 - 0.19, 0.0, 0.5], atol=1e-3)
        np.testing.assert_allclose(cloud["range"], [50.0 - 0.19], atol=1e-3)
        assert cloud.capture_t_ns == frame_ns(1) - 19_000_000

    def test_sensor_frame_and_disabled(self, scene: Scene) -> None:
        f = scene.frame(1)
        sensor = f.radar("radar_front", frame="sensor")
        base = f.radar("radar_front")
        np.testing.assert_allclose(sensor.xyz + [3.5, 0.0, 0.5], base.xyz, atol=1e-4)
        off = f.radar("radar_front", compensate=False)
        assert not off.motion_compensated and off.reason == "compensation disabled" and off.frame == "base"

    def test_without_poses_still_base_frame(self, scene_dir: Path) -> None:
        (scene_dir / "poses.txt").unlink()
        f = Scene(scene_dir).frame(1)
        cloud, raw = f.radar("radar_front"), f.radar("radar_front", frame="sensor")
        assert cloud.frame == "base" and not cloud.motion_compensated and cloud.reason == "no poses.txt"
        np.testing.assert_allclose(cloud.xyz, raw.xyz + [3.5, 0.0, 0.5], atol=1e-4)


# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------


def _load_lidar_sizes(frames, start: int, stop: int) -> list[int]:
    return [len(frames[i].lidar("lidar_top")) for i in range(start, stop)]


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_frames_load_in_worker_processes(scene_dir: Path, method: str) -> None:
    import multiprocessing as mp

    frames = kitscenes.open(scene_dir.parents[2]).frames()
    with mp.get_context(method).Pool(2) as pool:
        sizes = pool.starmap(_load_lidar_sizes, [(frames, 0, 2), (frames, 2, 3)])
    assert sizes == [[20, 20], [20]]


# ---------------------------------------------------------------------------
# Pickling and read-only caches
# ---------------------------------------------------------------------------


def test_errors_pickle() -> None:
    err = LabelsUnavailable("ground labels of lidar_top", 3, Availability.INVALID, "5 labels for 7 points")
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is LabelsUnavailable and clone.state is Availability.INVALID and str(clone) == str(err)
    m = pickle.loads(pickle.dumps(MapUnavailable("map", None, Availability.WITHHELD_BY_SPLIT, "", "no hint")))
    assert m.state is Availability.WITHHELD_BY_SPLIT and "withheld_by_split" in str(m) and "no hint" in str(m)
    assert isinstance(m, kitscenes.Unavailable) and not isinstance(m, SensorUnavailable)
    assert not isinstance(err, SensorUnavailable)  # labels and sensors are siblings under Unavailable


def test_cached_cloud_attributes_are_read_only(scene_dir: Path) -> None:
    f = kitscenes.open(scene_dir.parents[2], cache_mb=64)[0].frame(0)
    cloud = f.lidar("lidar_top")
    with pytest.raises(TypeError):
        cloud.attrs["timestamp"] = None
    with pytest.raises(AttributeError):
        cloud.attrs.pop("timestamp")
    assert "timestamp" in f.lidar("lidar_top").attrs
    assert len(cloud.select(cloud.measured)) == len(cloud)  # select() still works on the read-only mapping


def test_sensor_with_data_but_no_calibration_fails_the_rig(scene_dir: Path) -> None:
    """calib.json must have an entry for every sensor with data: no sensor-frame fallback."""
    calib = json.loads((scene_dir / "calibration" / "calib.json").read_text())
    del calib["lidar_top"], calib["radar_front"]
    (scene_dir / "calibration" / "calib.json").write_text(json.dumps(calib))
    scene = Scene(scene_dir)
    with pytest.raises(CalibrationError, match="no entry for lidar_top, radar_front.*scene_a"):
        scene.rig
    for load in (lambda f: f.lidar("lidar_top"), lambda f: f.radar("radar_front"), lambda f: f.fused_lidar()):
        with pytest.raises(CalibrationError):
            load(scene.frame(1))
    assert scene.availability("lidar_top", 1) is Availability.PRESENT  # the data is there


def test_sensor_downloaded_after_the_rig_loaded_needs_an_entry(scene_dir: Path) -> None:
    scene = Scene(scene_dir)
    assert "lidar_rear" not in scene.rig.lidars
    write_lidar(scene_dir, "lidar_rear", [0])
    with pytest.raises(CalibrationError, match="no entry for 'lidar_rear'"):
        scene.frame(0).lidar("lidar_rear")


def test_calib_json_missing(scene_dir: Path, tmp_path: Path) -> None:
    (scene_dir / "calibration" / "calib.json").unlink()
    with pytest.raises(CalibrationError, match="has no entry for camera_ring_front, lidar_top"):
        Scene(scene_dir).rig
    bare = make_scene(tmp_path / "bare", cameras=(), lidars=(), radars=())  # no sensor data: nothing to calibrate
    (bare / "calibration" / "calib.json").unlink()
    assert Scene(bare).rig.sensor_names == ()


def test_uncompensated_radar_keeps_unmeasured_rows_at_zero(scene_dir: Path) -> None:
    zeros = np.zeros(1, np.float32)
    write_radar(scene_dir, "radar_front", [0], n=1, overrides={0: {"x": zeros, "y": zeros, "z": zeros}})
    f = Scene(scene_dir).frame(0)
    for compensate in (True, False):
        cloud = f.radar("radar_front", compensate=compensate)
        assert not cloud.measured[0] and np.all(cloud.xyz[0] == 0)


def test_malformed_pose_lines_are_skipped(scene_dir: Path, caplog) -> None:
    text = (scene_dir / "poses.txt").read_text().splitlines()
    text.insert(1, "0.0960 1 2 three 0 0 0 1")
    text.insert(2, "abc 1 2 3 0 0 0 1")
    (scene_dir / "poses.txt").write_text("\n".join(text) + "\n")
    assert len(Scene(scene_dir).track) == 3
    assert caplog.text.count("Skipping malformed line") == 2


def test_shared_metadata_is_read_only(tmp_path) -> None:
    """An in-place edit of a cached scene's K, extrinsics or poses must not leak."""
    import kitscenes
    from release_fixture import make_scene

    make_scene(tmp_path / "data" / "train" / "s", n=4, cameras=("camera_ring_front",))
    scene = kitscenes.open(tmp_path)[0]
    camera = scene.rig.cameras["camera_ring_front"]
    for array in (camera.K, camera.T_base_sensor, scene.rig.lidars["lidar_top"].T_base_sensor,
                  scene._poses[0].xyz, scene._poses[0].quat, scene._trajectory[0].translations,
                  scene._trajectory[0].times, scene.frame(1).pose.xyz):
        with pytest.raises(ValueError, match="read-only"):
            array[0] += 1.0
    half = camera.scaled(0.5)  # derived cameras get their own (read-only) copy
    assert half.K[0, 0] == camera.K[0, 0] / 2 and not half.K.flags.writeable
    K = np.eye(3)
    from kitscenes.rig import Camera

    Camera("c", np.eye(4), K=K)
    K[0, 0] = 2.0  # the caller's array stays theirs


def test_pose_outside_the_poses_raises(tmp_path: Path) -> None:
    """Frame.pose raises PoseUnavailable (DROPPED) where the poses do not reach."""
    from kitscenes.samples import SampleSpec, make_sample

    scene_dir = make_scene(tmp_path / "data" / "train" / "s", n=12)
    lines = (scene_dir / "poses.txt").read_text().splitlines(keepends=True)
    (scene_dir / "poses.txt").write_text("".join(lines[:8]))
    scene = kitscenes.open(tmp_path)[0]
    assert scene.frame(8).pose is not None  # 0.1 s past the last pose: extrapolated
    with pytest.raises(PoseUnavailable, match="frame 11 is unavailable .dropped.") as err:
        scene.frame(11).pose
    assert err.value.state is Availability.DROPPED and scene.frame(11).availability("pose") is Availability.DROPPED
    assert scene.frame(11).lidar("lidar_top", frame="world", missing="none") is None
    with pytest.raises(PoseUnavailable):
        scene.ego(scene.frame(11).t_ns)
    assert np.isnan(make_sample(scene.frame(11), SampleSpec(frame="base"))["pose"]["T_world_base"]).all()
    with pytest.raises(PoseUnavailable):
        scene.accumulate([10, 11])

    from kitscenes.visualization.reprojection import render_frame

    class Layer:  # a map layer: labels around a pose
        hidden_ids = frozenset()

        def __call__(self, pose):
            assert pose is not None
            return None

    assert render_frame(scene.frame(11), map_layer=Layer(), lidar=False, radar=False, cameras=()) == {}


def test_split_rules_without_a_dataset(tmp_path: Path) -> None:
    """kitscenes.Scene(path) takes the split from its folder, or from the scene table."""
    scene_dir = make_scene(tmp_path / "data" / "test" / "s", n=2)
    scene = Scene(scene_dir)
    assert scene.split == "test" and scene.split_info.name == "test"
    assert scene.availability("ground_labels/lidar_top", 0) is Availability.WITHHELD_BY_SPLIT
    assert scene.availability("instances/camera_ring_front", 0) is Availability.WITHHELD_BY_SPLIT
    assert Scene(make_scene(tmp_path / "data" / "test_e2e" / "e", n=2)).split == "test-e2e"  # alias -> folder name

    listed = "00c8940e-5f49-aac3-2947-e481599bc970"  # a test-split scene of the v2.0 scene table
    for parent in ("unreleased_folder", None):
        path = tmp_path / parent / listed if parent else tmp_path / "loose" / listed
        scene = Scene(make_scene(path, n=2))
        assert scene.split is None and scene.split_info.name == "test"
        assert scene.availability("ground_labels/lidar_top", 0) is Availability.WITHHELD_BY_SPLIT
    with pytest.raises(kitscenes.DatasetNotFound, match="data/<split>/<scene_id>/"):
        kitscenes.open(tmp_path / "loose")  # scene folders without split folders are no dataset root

    unknown = Scene(make_scene(tmp_path / "unreleased_folder" / "u", n=2))
    assert unknown.split is None and unknown.split_info is None
    assert Scene(unknown.files.path, split="unreleased_folder").split_info is None


def test_one_error_model_for_missing_data(tmp_path: Path) -> None:
    """Every accessor raises an Unavailable with a state and takes missing="none"."""
    scene_dir = _scene_with_labels(tmp_path, 0)
    scene = Scene(scene_dir, split="test")
    f = scene.frame(0)
    for call in (lambda m: f.ground_mask("lidar_top", missing=m), lambda m: f.instances("camera_ring_front", missing=m),
                 lambda m: f.image("camera_ring_rear", missing=m), lambda m: f.lidar_deskew_inputs("lidar_rear", missing=m),
                 lambda m: f.fused_lidar(["lidar_rear"], missing=m), lambda m: f.aerial(missing=m),
                 lambda m: scene.messages("gnss", "navsatfix", missing=m), lambda m: scene.imu(missing=m)):
        with pytest.raises(kitscenes.Unavailable) as err:
            call("raise")
        assert err.value.state is not Availability.PRESENT and err.value.state.value in str(err.value)
        assert call("none") is None
    with pytest.raises(LabelsUnavailable, match="the test split ships no labels"):
        f.ground_mask("lidar_top")
    assert f.availability("ground_labels/lidar_top") is Availability.WITHHELD_BY_SPLIT
    assert f.availability("instances/camera_ring_rear") is Availability.WITHHELD_BY_SPLIT
    assert scene.availability("gnss/navsatfix") is Availability.NOT_DOWNLOADED  # the test split ships GNSS
    assert scene.availability("map") is Availability.WITHHELD_BY_SPLIT
    assert scene.availability("pose") is Availability.PRESENT and f.availability("pose") is Availability.PRESENT
    with pytest.raises(KeyError, match="ground_labels/<lidar>"):
        scene.availability("ground_labels/camera_ring_front")
    # missing="none" on lidars() leaves out what is missing and fuses the rest
    assert Scene(scene_dir, split="train").frame(0).fused_lidar(["lidar_top", "lidar_rear"], missing="none").sensor == ("lidar_top",)
    with pytest.raises(ValueError, match="missing"):
        f.ground_mask("lidar_top", missing="skip")


def test_lidar_and_radar_in_the_world_frame(scene: Scene) -> None:
    """frame.lidar(..., frame="world") puts the sweep through the frame's pose."""
    f = scene.frame(1)
    base, world = f.lidar("lidar_top"), f.lidar("lidar_top", frame="world")
    assert world.frame == "world" and world.motion_compensated == base.motion_compensated
    expected = kitscenes.transform_points(f.T_world_base, base.xyz[base.measured])
    np.testing.assert_allclose(world.xyz[world.measured], expected, atol=1e-3)
    assert not world.xyz[~world.measured].any()
    fused = f.fused_lidar(frame="world")
    assert fused.frame == "world" and len(fused) == len(f.fused_lidar())
    radar = f.radar("radar_front", frame="world")
    assert radar.frame == "world"
    np.testing.assert_allclose(radar.xyz[radar.measured], kitscenes.transform_points(
        f.T_world_base, f.radar("radar_front").xyz[radar.measured]), atol=1e-3)
