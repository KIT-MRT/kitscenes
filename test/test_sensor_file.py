"""frame.sensor_file / scene.sensor_file: where a sensor frame's file is, in folders and in the release zips."""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import pytest

from kitscenes import Availability, Scene, SensorFile, SensorUnavailable
from kitscenes._storage import DirStorage
from release_fixture import make_scene, pack_scene

SENSORS = ("camera_ring_front", "lidar_top", "radar_front")


@pytest.fixture(params=["folder", "zip"])
def layout(request) -> str:
    return request.param


@pytest.fixture
def scene(tmp_path: Path, layout: str) -> Scene:
    path = make_scene(tmp_path / "data" / "train" / "scene_a", n=4, calib_only=("camera_ring_rear",))
    (path / "camera_ring_front" / "0000000002.jpg").unlink()  # a dropped frame
    if layout == "zip":
        pack_scene(path)
    return Scene(path)


def test_sensor_file_reads_what_the_frame_reads(scene: Scene, layout: str) -> None:
    f = scene.frame(1)
    for sensor in SENSORS:
        where = f.sensor_file(sensor)
        assert isinstance(where, SensorFile) and (where.sensor, where.index) == (sensor, 1)
        assert where.member == f"{sensor}/0000000001.{'jpg' if sensor.startswith('camera') else 'parquet'}"
        assert where.read() == scene.files.storage.read_bytes(where.member)
        if layout == "folder":
            assert not where.in_zip and where.zip_member is None and where.offset == 0
            assert where.path == scene.files.path / where.member and where.size == where.path.stat().st_size
        else:
            component = "radar" if sensor.startswith("radar") else sensor
            assert where.in_zip and where.path == scene.files.path / f"{component}.zip"
            assert where.zip_member == f"scene_a/{where.member}"
            with zipfile.ZipFile(where.path) as zf:  # a plain zip reader finds the same bytes by name
                assert zf.read(where.zip_member) == where.read()
            if sensor != "radar_front":
                assert where.offset % 4096 == 0  # camera and lidar members are aligned
    assert f.sensor_file("camera_ring_front").read() == f.image_bytes("camera_ring_front")
    same = scene.frame(-3).sensor_file("camera_ring_front")  # negative indices count from the end
    assert same == f.sensor_file("camera_ring_front")
    assert pickle.loads(pickle.dumps(same)) == same


def test_sensor_file_without_data(scene: Scene, layout: str) -> None:
    with pytest.raises(SensorUnavailable) as err:
        scene.frame(2).sensor_file("camera_ring_front")
    assert err.value.state is Availability.DROPPED
    assert scene.frame(2).sensor_file("camera_ring_front", missing="none") is None
    with pytest.raises(SensorUnavailable) as err:
        scene.frame(0).sensor_file("camera_ring_rear")
    # every scene has all sensors: one without data is not downloaded, also if index.parquet does not list it
    assert err.value.state is Availability.NOT_DOWNLOADED and "--components camera_ring_rear" in str(err.value)
    with pytest.raises(KeyError):
        scene.frame(0).sensor_file("camera_nowhere")
    with pytest.raises(IndexError):
        scene.frame(9).sensor_file("camera_ring_front")


def test_sensor_file_of_an_extracted_component(scene: Scene, layout: str) -> None:
    if layout == "folder":
        pytest.skip("only zip scenes have components to extract")
    from kitscenes.download.archives import extract_component

    extract_component(scene.files.path / "lidar_top.zip")
    where = scene.frame(0).sensor_file("lidar_top")
    assert not where.in_zip and where.path == scene.files.path / "lidar_top" / "0000000000.parquet"
    assert where.read() == scene.files.storage.read_bytes(where.member)


def test_sensor_file_deleted_since_the_scene_looked(scene: Scene, layout: str) -> None:
    f = scene.frame(0)
    assert f.sensor_file("camera_ring_front")
    gone = scene.files.path / ("camera_ring_front.zip" if layout == "zip" else "camera_ring_front/0000000000.jpg")
    gone.unlink()
    if layout == "folder":
        for other in (scene.files.path / "camera_ring_front").iterdir():
            other.unlink()
        (scene.files.path / "camera_ring_front").rmdir()
    with pytest.raises(SensorUnavailable) as err:
        f.sensor_file("camera_ring_front")
    assert err.value.state is Availability.NOT_DOWNLOADED


class _NoFiles:
    """A storage that serves bytes but has no files on disk (e.g. a remote one)."""

    def __init__(self, inner: DirStorage) -> None:
        self.inner = inner

    def exists(self, name):
        return self.inner.exists(name)

    def is_dir(self, name):
        return self.inner.is_dir(name)

    def list(self, prefix):
        return self.inner.list(prefix)

    def read_bytes(self, name):
        return self.inner.read_bytes(name)

    def describe(self, name=""):
        return self.inner.describe(name)


def test_sensor_file_needs_files_on_disk(tmp_path: Path) -> None:
    path = make_scene(tmp_path / "data" / "train" / "scene_a")
    scene = Scene(_NoFiles(DirStorage(path)), "scene_a", "train")
    assert scene.frame(0).image_bytes("camera_ring_front")
    with pytest.raises(TypeError, match="no files on disk"):
        scene.frame(0).sensor_file("camera_ring_front")
