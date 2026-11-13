"""The release archive format: ZipStorage reads per-component zips without extracting; kitscenes.pack writes them."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes import Availability, pack
from kitscenes._archive import COMPONENTS, INDEX_FILE, ArchiveIndex, component_of, write_index
from kitscenes._layout import sensor_member
from kitscenes._storage import DirStorage, ZipStorage, open_storage
from kitscenes.errors import ChecksumError, DataFormatError
from release_fixture import make_scene, pack_scene, write_ground_labels, write_messages


def _scene(tmp_path: Path, name: str = "scene_a") -> Path:
    scene = make_scene(
        tmp_path / "data" / "train" / name, n=4,
        cameras=("camera_ring_front", "camera_ring_rear"), lidars=("lidar_top",), radars=("radar_front", "radar_left"),
    )
    write_ground_labels(scene, "lidar_top", 0, [True] * 20)
    write_messages(scene, "gnss", "navsatfix", [{"header": {"stamp": 0.05}, "latitude": 49.0}])
    return scene


def test_component_of() -> None:
    assert component_of("camera_ring_front/0000000000.jpg") == "camera_ring_front"
    assert component_of("radar_left/0000000003.parquet") == "radar"
    assert component_of("processed/ground_segmentation/lidar_top/0000000000.parquet") == "processed_ground_segmentation"
    assert component_of("gnss/navsatfix.parquet") == "gnss" and component_of("gnss_ins/imu.parquet") == "gnss_ins"
    assert component_of("poses.txt") is None and component_of("processed/other/x") is None


def test_zip_reads_equal_folder_reads(tmp_path: Path) -> None:
    folder = _scene(tmp_path / "dir")
    packed = pack_scene(shutil.copytree(folder, tmp_path / "zip" / "data" / "train" / "scene_a"))
    assert isinstance(open_storage(packed), ZipStorage) and type(open_storage(folder)) is DirStorage
    assert not (packed / "camera_ring_front").exists() and (packed / "radar.zip").is_file()
    a, b = DirStorage(folder), ZipStorage(packed, verify=True)
    for prefix in ("camera_ring_front", "lidar_top", "radar_left", "processed/ground_segmentation/lidar_top", "gnss"):
        assert a.list(prefix) == b.list(prefix), prefix
        for member in a.list(prefix):
            assert b.read_bytes(member) == a.read_bytes(member)
    assert b.is_dir("radar_front") and b.is_dir("processed/ground_segmentation") and not b.is_dir("lidar_rear")
    assert b.read_bytes("poses.txt") == a.read_bytes("poses.txt")
    from kitscenes._archive import ALIGNED

    entries = [b.index.entry(m) for c in b.index.components for f in COMPONENTS[c] for m in a.list(f)]
    offsets = {(c in ALIGNED, o % 4096 == 0) for c, o, _, _ in entries}
    assert (True, False) not in offsets and (False, False) in offsets  # cameras/lidars aligned, small files not

    sa, sb = kitscenes.Scene(folder, split="train"), kitscenes.Scene(packed, split="train")
    assert sa.sensors == sb.sensors
    fa, fb = sa.frame(2), sb.frame(2)
    np.testing.assert_array_equal(fa.lidar("lidar_top").xyz, fb.lidar("lidar_top").xyz)
    np.testing.assert_array_equal(fa.image("camera_ring_rear"), fb.image("camera_ring_rear"))
    np.testing.assert_array_equal(sa.frame(0).ground_mask("lidar_top"), sb.frame(0).ground_mask("lidar_top"))


def test_partial_download_and_extracted_components(tmp_path: Path) -> None:
    scene = pack_scene(_scene(tmp_path))
    (scene / "camera_ring_rear.zip").unlink()  # not downloaded
    ds = kitscenes.open(tmp_path)
    f = ds[0].frame(1)
    assert f.availability("camera_ring_rear") is Availability.NOT_DOWNLOADED
    assert f.availability("camera_ring_front") is Availability.PRESENT
    with pytest.raises(kitscenes.SensorUnavailable):
        f.image("camera_ring_rear")
    # Extracting a component zip (into data/<split>/) works the same.
    import zipfile

    with zipfile.ZipFile(scene / "radar.zip") as zf:
        zf.extractall(scene.parent)
    (scene / "radar.zip").unlink()
    s = kitscenes.Scene(scene, split="train")
    assert s.availability("radar_left", 1) is Availability.PRESENT and len(s.frame(1).radar("radar_left")) == 5


def test_crc_check_and_newer_format(tmp_path: Path) -> None:
    scene = pack_scene(_scene(tmp_path))
    storage = ZipStorage(scene, verify=True)
    _, offset, _, _ = storage.index.entry("lidar_top/0000000001.parquet")
    with open(scene / "lidar_top.zip", "r+b") as fh:
        fh.seek(offset + 10)
        byte = fh.read(1)
        fh.seek(offset + 10)
        fh.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ChecksumError, match="CRC") as err:
        storage.read_bytes("lidar_top/0000000001.parquet")
    # corrupt data, not an I/O error: callers that retry OSErrors (a failing mount) must not wait for it
    assert isinstance(err.value, DataFormatError) and not isinstance(err.value, OSError)
    assert ZipStorage(scene).read_bytes("lidar_top/0000000001.parquet")  # verify is off by default

    import io

    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(write_index([("lidar_top", "lidar_top/x", 0, 1, 0)])))
    table = table.replace_schema_metadata({"kitscenes_archive_format": "99"})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    with pytest.raises(ValueError, match="format 99"):
        ArchiveIndex.from_bytes(buf.getvalue())


def test_index_is_read_per_component(tmp_path: Path) -> None:
    scene = pack_scene(_scene(tmp_path))
    index = ZipStorage(scene).index
    assert index._parts == {} and "radar" in index.components and len(index) > 0
    # Folders a component wholly covers are answered without indexing its members.
    assert index.folder_components("camera_ring_front") == {"camera_ring_front"}
    assert index.folder_components("processed") == {"processed_ground_segmentation"}
    assert index.folder_components("") == frozenset() and index.children("") == []
    assert index._parts == {}
    entry = index.entry(sensor_member("lidar_top", 1))
    assert entry is not None and entry[0] == "lidar_top" and set(index._parts) == {"lidar_top"}
    # radar holds three folders; this scene ships two of them.
    assert index.folder_components("radar_left") == {"radar"} and index.folder_components("radar_right") == frozenset()
    assert index.folder_components("gnss") == {"gnss"} and index.folder_components("gnss_ins") == frozenset()
    assert index.children("gnss") == ["gnss/navsatfix.parquet"]
    members = index.children("radar_left")
    assert members == sorted(members) and members and all(m.startswith("radar_left/") for m in members)
    assert index.entry("poses.txt") is None and index.entry("lidar_rear/0000000000.parquet") is None
    assert set(index._parts) == {"lidar_top", "radar", "gnss"}


def test_scene_cache_and_threaded_frames(tmp_path: Path) -> None:
    for name in ("scene_a", "scene_b", "scene_c"):
        pack_scene(_scene(tmp_path, name))
    ds = kitscenes.open(tmp_path, scene_cache=2)
    assert ds.scene_cache == 2 and pickle.loads(pickle.dumps(ds)).scene_cache == 2
    assert ds.filter(splits="train").scene_cache == 2
    for scene_id in ds.scene_ids:
        ds.scene(scene_id)
    assert len(ds._scenes) == 2
    with pytest.raises(ValueError, match="scene_cache"):
        kitscenes.open(tmp_path, scene_cache=0)
    serial = ds.frames(require="lidar_top", workers=1)
    assert len(serial) == 12 and (kitscenes.open(tmp_path).frames(require="lidar_top", workers=4).refs) == serial.refs


def _read_in_worker(storage: ZipStorage, member: str) -> int:
    return len(storage.read_bytes(member))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_worker_processes(tmp_path: Path, method: str) -> None:
    scene = pack_scene(_scene(tmp_path))
    storage = ZipStorage(scene)
    member = sensor_member("camera_ring_front", 0)
    expected = len(storage.read_bytes(member))  # opens a handle in the parent
    clone = pickle.loads(pickle.dumps(storage))
    assert clone == storage and clone._handles == {}
    with mp.get_context(method).Pool(2) as pool:
        assert pool.starmap(_read_in_worker, [(storage, member)] * 4) == [expected] * 4


def test_aligned_zip_is_deterministic_and_checked(tmp_path: Path) -> None:
    entries = [("s/camera_x/0000000000.jpg", b"a" * 5000), ("s/camera_x/0000000001.jpg", b"b" * 10)]
    m1 = pack.write_aligned_zip(entries, tmp_path / "a.zip")
    m2 = pack.write_aligned_zip(iter(entries), tmp_path / "b.zip")
    assert pack.sha256_file(tmp_path / "a.zip") == pack.sha256_file(tmp_path / "b.zip") and m1 == m2
    assert all(m.data_offset % 4096 == 0 for m in m1)
    with zipfile.ZipFile(tmp_path / "a.zip") as zf:
        assert zf.namelist() == ["s/", "s/camera_x/", *(n for n, _ in entries)] and zf.testzip() is None
    pack.check_zip(tmp_path / "a.zip", m1)
    with pytest.raises(ValueError, match="sorted"):
        pack.write_aligned_zip(entries[::-1], tmp_path / "c.zip")


def test_pack_scene_round_trip(tmp_path: Path) -> None:
    scene = make_scene(tmp_path / "src" / "train" / "s1", n=3, cameras=("camera_ring_front",))
    files = pack.pack_scene(scene, tmp_path / "out" / "data" / "train" / "s1")
    again = pack.pack_scene(scene, tmp_path / "out2" / "data" / "train" / "s1")
    assert [(f.path, f.sha256) for f in files] == [(f.path, f.sha256) for f in again]
    assert {f.path for f in files} >= {"camera_ring_front.zip", "lidar_top.zip", "radar.zip", "index.parquet", "poses.txt"}
    a, b = kitscenes.Scene(scene), kitscenes.Scene(tmp_path / "out" / "data" / "train" / "s1")
    assert a.frame(1).image_bytes("camera_ring_front") == b.frame(1).image_bytes("camera_ring_front")
    np.testing.assert_array_equal(a.frame(2).lidar("lidar_top").xyz, b.frame(2).lidar("lidar_top").xyz)
    (scene / "processed").mkdir()
    (scene / "processed" / "stray.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="outside the archive format"):
        pack.pack_scene(scene, tmp_path / "out3" / "data" / "train" / "s1")
