"""kitscenes.open / Dataset: discovery, splits, selection, pickling."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest

import kitscenes
from kitscenes.catalog import SCENE_CACHE_SIZE, Dataset
from kitscenes.constants import KITSCENES_ROOT_ENV_VAR
from kitscenes.errors import DatasetNotFound, SceneNotFound
from release_fixture import make_dataset, make_scene, write_timestamps

_RELEASE_SPLITS = (
    "train", "val", "test", "test-e2e", "overlap_train_val",
    "map_change_train", "map_change_val", "map_change_test", "map_change_test-e2e",
    "map_change_overlap_train_val",
)


def _light_scene(path: Path, n: int = 2) -> Path:
    """Only the timeline: enough for discovery tests."""
    path.mkdir(parents=True)
    write_timestamps(path, n)
    return path


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    root = tmp_path / "kitscenes"
    for i, split in enumerate(_RELEASE_SPLITS):
        _light_scene(root / "data" / split / f"scene_{i:02d}")
    return root


class TestDiscovery:
    def test_all_release_splits(self, release_root: Path) -> None:
        ds = kitscenes.open(release_root)
        assert len(ds) == 10 and ds.splits == list(_RELEASE_SPLITS)
        assert [ds.split_of(i) for i in ds.scene_ids] == list(_RELEASE_SPLITS)
        assert [s.split for s in ds] == list(_RELEASE_SPLITS)

    @pytest.mark.parametrize("split", _RELEASE_SPLITS)
    def test_each_split_by_name(self, release_root: Path, split: str) -> None:
        ds = kitscenes.open(release_root, splits=split)
        assert len(ds) == 1 and ds[0].split == split

    def test_aliases_and_lists(self, release_root: Path) -> None:
        assert kitscenes.open(release_root, "test_e2e").splits == ["test-e2e"]
        assert kitscenes.open(release_root, "validation").splits == ["val"]
        assert kitscenes.open(release_root, ["map_change_train", "train"]).splits == ["train", "map_change_train"]

    def test_unknown_or_missing_split(self, tmp_path: Path) -> None:
        root = tmp_path / "kitscenes"
        _light_scene(root / "data" / "train" / "a")
        with pytest.raises(ValueError, match="Unknown split"):
            kitscenes.open(root, "training")
        with pytest.raises(DatasetNotFound, match="test-e2e"):
            kitscenes.open(root, "test-e2e")

    def test_membership_is_the_folder(self, tmp_path: Path) -> None:
        root = tmp_path / "kitscenes"
        _light_scene(root / "data" / "val" / "not-in-any-list")
        assert kitscenes.open(root, "val").scene_ids == ["not-in-any-list"]

    def test_only_the_release_layout(self, tmp_path: Path, caplog) -> None:
        """Scenes are in data/<split>/<scene_id>/; any other layout names that one."""
        no_data = tmp_path / "no_data"
        _light_scene(no_data / "map_change_val" / "a")
        flat = tmp_path / "flat"
        _light_scene(flat / "b")
        no_split = tmp_path / "no_split"
        _light_scene(no_split / "data" / "c")
        for root in (no_data, flat, no_split):
            with pytest.raises(DatasetNotFound, match="data/<split>/<scene_id>/"):
                kitscenes.open(root)
        future = tmp_path / "future"
        _light_scene(future / "data" / "future_split" / "c")
        (future / "data" / "not_a_split").mkdir()
        (future / "data" / ".kitscenes_staging").mkdir()
        assert kitscenes.open(future).split_of("c") == "future_split"
        assert "Unknown split folder" in caplog.text

    def test_only_test_e2e_folder_gives_no_fake_scene(self, tmp_path: Path) -> None:
        root = tmp_path / "kitscenes"
        _light_scene(root / "data" / "test-e2e" / "a")
        assert kitscenes.open(root).scene_ids == ["a"]

    def test_errors(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv(KITSCENES_ROOT_ENV_VAR, raising=False)
        with pytest.raises(ValueError, match=KITSCENES_ROOT_ENV_VAR):
            kitscenes.open()
        with pytest.raises(DatasetNotFound):
            kitscenes.open(tmp_path / "missing")
        (tmp_path / "empty" / "data").mkdir(parents=True)
        with pytest.raises(DatasetNotFound, match="No scenes"):
            kitscenes.open(tmp_path / "empty")
        scene = _light_scene(tmp_path / "x" / "data" / "train" / "a")
        with pytest.raises(ValueError, match="scene folder"):
            kitscenes.open(scene)

    def test_env_root(self, release_root: Path, monkeypatch) -> None:
        monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, str(release_root))
        assert len(kitscenes.open()) == 10


class TestSelection:
    def test_index_slice_filter(self, release_root: Path) -> None:
        ds = kitscenes.open(release_root)
        assert ds["scene_03"].split == "test-e2e" and ds[-1].id == "scene_09"
        sub = ds[2:4]
        assert isinstance(sub, Dataset) and sub.scene_ids == ["scene_02", "scene_03"]
        assert ds.filter(splits=["test", "test_e2e"]).scene_ids == ["scene_02", "scene_03"]
        assert ds.filter(scene_ids=["scene_05", "scene_01"]).scene_ids == ["scene_05", "scene_01"]
        assert len(ds.filter(splits="train", scene_ids=["scene_05"])) == 0
        assert "scene_00" in ds and "nope" not in ds
        with pytest.raises(SceneNotFound, match="nope"):
            ds["nope"]
        with pytest.raises(KeyError):
            ds.filter(scene_ids=["nope"])

    def test_scene_objects_are_reused_but_bounded(self, tmp_path: Path) -> None:
        root = tmp_path / "kitscenes"
        for i in range(SCENE_CACHE_SIZE + 2):
            _light_scene(root / "data" / "train" / f"s{i:03d}")
        ds = kitscenes.open(root)
        assert ds[0] is ds[0]
        for scene in ds:
            scene.timestamps_ns
        assert len(ds._scenes) == SCENE_CACHE_SIZE
        assert "s000" not in ds._scenes  # evicted; rebuilt on access
        assert ds.scene("s000").num_frames == 2


class TestFrames:
    def test_frames_and_required_sensors(self, tmp_path: Path) -> None:
        root = make_dataset(tmp_path / "kitscenes", {"train": ["a"], "val": ["b"]}, n=4)
        (root / "data/train/a/lidar_top/0000000002.parquet").unlink()
        ds = kitscenes.open(root)
        assert len(ds.frames()) == 8
        frames = ds.frames(require=("lidar_top", "camera_ring_front"))
        assert [(f.scene.id, f.index) for f in frames] == [
            ("a", 0), ("a", 1), ("a", 3), ("b", 0), ("b", 1), ("b", 2), ("b", 3)]
        assert len(frames[1:3]) == 2

    def test_pickling(self, tmp_path: Path) -> None:
        root = make_dataset(tmp_path / "kitscenes", {"train": ["a"]})
        ds = kitscenes.open(root, cache_mb=16)
        frames = ds.frames()
        frames[0].lidar("lidar_top")
        clone = pickle.loads(pickle.dumps(frames))
        assert len(pickle.dumps(frames)) < 2000
        assert clone[1].t_ns == frames[1].t_ns
        assert clone.dataset.cache is not None and len(clone.dataset.cache) == 0
        scene = pickle.loads(pickle.dumps(ds[0]))
        assert scene.split == "train" and scene.num_frames == 3


def test_small_traps(tmp_path: Path) -> None:
    """One name as a string, city/lidar in any case and checked, scene[i], IndexError with context."""
    import kitscenes
    from release_fixture import make_scene

    listed = "00c8940e-5f49-aac3-2947-e481599bc970"  # a test-split scene of the v2.0 scene table
    make_scene(tmp_path / "data" / "test" / listed, n=3, cameras=(), radars=())
    ds = kitscenes.open(tmp_path)
    assert len(ds.frames(require="lidar_top")) == 3  # not 'l', 'i', 'd', ...
    city = kitscenes.scene_table()[listed].city
    assert len(ds.filter(city=city.lower())) == 1 and len(ds.filter(city=[city.upper()])) == 1
    lidar = kitscenes.scene_table()[listed].lidar
    assert len(ds.filter(lidar=lidar.upper())) == 1
    with pytest.raises(ValueError, match="Unknown city 'Karlsruh'.*Frankfurt"):
        ds.filter(city="Karlsruh")
    with pytest.raises(ValueError, match="Unknown lidar"):
        ds.filter(lidar="ouster")
    with pytest.raises(IndexError, match="out of range: the dataset has 1 scenes"):
        ds[10_000]
    scene = ds[0]
    assert scene[1].index == 1 and scene[-1].index == 2 and [f.index for f in scene[1:]] == [1, 2]
    with pytest.raises(KeyError, match=r"No camera 'camera_x' in this scene's calib.json; it has"):
        scene.rig.cameras["camera_x"]
