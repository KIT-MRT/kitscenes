"""GNSS/INS messages as tables (one Parquet file per topic) and the IMU accessor."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kitscenes
from kitscenes import Availability, Scene, SensorUnavailable
from kitscenes.errors import DataFormatError
from release_fixture import imu_message, make_scene, pack_scene, write_messages


def _scene(tmp_path: Path) -> Path:
    scene = make_scene(tmp_path / "data" / "train" / "s", n=3)
    # IMU at 50 Hz: messages 0, 5, 10 are the copies of frames 0, 1, 2.
    write_messages(scene, "gnss_ins", "imu", [imu_message(0.046049024 + 0.02 * k, -0.05 * k) for k in range(12)],
                   frames={0: 0, 5: 1, 10: 2})
    # 1 Hz status-like topic: one message, assigned to frame 0 only; nan and inf as the release has them.
    write_messages(scene, "gnss_ins", "insnavgeod", [
        {"header": {"stamp": 0.046 + 1.0 * k}, "latitude": 0.855, "heading_pitch_cov": "nan", "bad": [1.0, "inf"]}
        for k in range(2)], frames={0: 0}, t_sync_s=[0.0462, 1.0462])
    return scene


def test_native_table(tmp_path: Path) -> None:
    scene = Scene(_scene(tmp_path), split="train")
    assert scene.message_topics("gnss_ins") == ["imu", "insnavgeod"]
    assert scene.message_topics("gnss") == []
    table = scene.messages("gnss_ins", "imu")
    assert table.column_names[:5] == ["index", "t_ns", "t_sync_ns", "frame_index", "header.seq"]
    assert table.num_rows == 12 and table.column("index").to_pylist() == list(range(12))
    assert table.column("t_ns").to_pylist()[:2] == [46_049_024, 66_049_024]
    assert table.column("t_sync_ns").to_pylist()[:2] == [46_200_000, 66_200_000]
    assert table.column("frame_index").to_pylist()[:6] == [0, None, None, None, None, 1]
    assert "angular_velocity.z" in table.column_names and len(table.column("orientation_covariance")[0]) == 9
    assert scene.messages("gnss_ins", "imu") is table  # cached
    ins = scene.messages("gnss_ins", "insnavgeod")
    assert np.isnan(ins.column("heading_pitch_cov").to_numpy()).all()
    assert ins.column("bad")[0].as_py()[1] == math.inf


def test_frame_rate_is_the_rows_with_a_frame(tmp_path: Path) -> None:
    scene = Scene(_scene(tmp_path), split="train")
    native = scene.messages("gnss_ins", "imu")
    frame = scene.messages("gnss_ins", "imu", rate="frame")
    assert frame.column_names == native.column_names
    assert frame.column("index").to_pylist() == [0, 1, 2] == frame.column("frame_index").to_pylist()  # index = frame
    assert frame.column("t_ns").to_pylist() == [native.column("t_ns")[k].as_py() for k in (0, 5, 10)]
    assert frame.column("t_sync_ns").to_pylist() == [round(t * 1e9) for t in (0.0462, 0.1462, 0.2462)]
    np.testing.assert_array_equal(frame.column("angular_velocity.z").to_numpy(), [0.0, -0.25, -0.5])
    assert scene.messages("gnss_ins", "imu", rate="frame") is frame  # cached
    slow = scene.messages("gnss_ins", "insnavgeod", rate="frame")  # slower than the frames: fewer rows
    assert slow.column("index").to_pylist() == [0] and slow.column("t_ns").to_pylist() == [46_000_000]


def test_bad_arguments_and_missing_topics(tmp_path: Path) -> None:
    scene = Scene(_scene(tmp_path), split="train")
    with pytest.raises(ValueError, match="rate"):
        scene.messages("gnss_ins", "imu", rate="fast")
    with pytest.raises(KeyError, match="device"):
        scene.messages("gps", "imu")
    with pytest.raises(KeyError, match="'imuu'.*imu"):  # a topic the device does not have: a typo, not missing data
        scene.messages("gnss_ins", "imuu", missing="none")
    with pytest.raises(KeyError, match="imuu"):
        scene.availability("gnss_ins/imuu")
    assert scene.availability("gnss_ins/imu") is Availability.PRESENT


def test_not_a_topic_file(tmp_path: Path) -> None:
    scene_dir = _scene(tmp_path)
    pq.write_table(pa.table({"x": [1]}), scene_dir / "gnss_ins" / "pose.parquet")
    with pytest.raises(DataFormatError, match="not a GNSS topic file"):
        Scene(scene_dir, split="train").messages("gnss_ins", "pose")


def test_imu_frames(tmp_path: Path) -> None:
    scene = Scene(_scene(tmp_path), split="train")
    raw, base = scene.imu(), scene.imu(frame="base")
    assert raw["orientation"].shape == (12, 4) and "orientation" not in base
    np.testing.assert_allclose(raw["angular_velocity"][3], [0.1, 0.2, -0.15])
    np.testing.assert_allclose(base["angular_velocity"][3], [-0.2, -0.1, 0.15])  # x=-y_imu, y=-x_imu, z=-z_imu
    np.testing.assert_allclose(base["linear_acceleration"][:, 2], 9.81)  # specific force up at rest
    assert raw["t_ns"].dtype == np.int64 and raw["t_ns"][0] == 46_049_024


def test_availability_and_zip(tmp_path: Path) -> None:
    scene_dir = _scene(tmp_path)
    with pytest.raises(SensorUnavailable) as err:
        Scene(scene_dir, split="test-e2e").imu()
    assert err.value.state is Availability.WITHHELD_BY_SPLIT
    assert Scene(scene_dir, split="test-e2e").availability("gnss_ins/imu") is Availability.WITHHELD_BY_SPLIT
    with pytest.raises(SensorUnavailable) as err:
        Scene(scene_dir, split="train").messages("gnss", "navsatfix")
    assert err.value.state is Availability.NOT_DOWNLOADED and "--components gnss" in str(err.value)
    assert Scene(scene_dir, split="train").messages("gnss", "navsatfix", rate="frame", missing="none") is None
    folder = Scene(scene_dir, split="train")
    folder_imu, folder_frame = folder.imu(), folder.messages("gnss_ins", "insnavgeod", rate="frame")
    pack_scene(scene_dir)
    assert (scene_dir / "gnss_ins.zip").is_file() and not (scene_dir / "gnss_ins").exists()
    assert not (scene_dir / "async.zip").exists()
    zipped = kitscenes.open(tmp_path)[0]
    assert zipped.message_topics("gnss_ins") == ["imu", "insnavgeod"]
    np.testing.assert_array_equal(zipped.imu()["angular_velocity"], folder_imu["angular_velocity"])
    cols = ["index", "t_ns", "t_sync_ns", "frame_index", "latitude", "bad"]  # (NaN != NaN in Table.equals)
    assert zipped.messages("gnss_ins", "insnavgeod", rate="frame").select(cols).equals(folder_frame.select(cols))
    with pytest.raises(SensorUnavailable) as err:
        zipped.messages("gnss", "navsatfix")
    assert err.value.state is Availability.NOT_DOWNLOADED


def test_rows_without_index_or_sync_time(tmp_path: Path) -> None:
    """A few release rows have a null index or t_sync_ns; both rates read them."""
    scene_dir = make_scene(tmp_path / "data" / "train" / "r", n=3)
    write_messages(scene_dir, "gnss_ins", "imu", [imu_message(0.0462 + 0.1 * k, 0.0) for k in range(3)],
                   frames={0: 0, 1: 1, 2: 2}, t_sync_s=[0.0462, 0.1462, 0.2462], index=[0, None, 2])
    scene = Scene(scene_dir, split="train")
    imu = scene.messages("gnss_ins", "imu")
    assert imu.column("index").to_pylist() == [0, None, 2] and imu.column("frame_index").to_pylist() == [0, 1, 2]
    assert scene.messages("gnss_ins", "imu", rate="frame").column("index").to_pylist() == [0, 1, 2]


