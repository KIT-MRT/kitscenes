"""Dataset.prepare_aerials / use_aerial / Frame.aerial on synthetic release-layout roots (test/aerial_fixture.py)."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes.aerial import AerialCache, AerialCrops, cache_ids
from kitscenes.aerial import prepare as prepare_module
from kitscenes.aerial._geo import latlon_to_epsg25832
from kitscenes.errors import AerialUnavailable, Availability

from aerial_fixture import FRANKFURT, KARLSRUHE, RAW_CACHE_ID, write_scene, write_tiles

# Small crops and tiles so the tests stay fast: 1 m/px, 6 x 4 m crops, 20 m tiles.
ARGS = dict(roi=(6.0, 4.0))
CACHE_ID = "dop20_res100_roi6x4_ego_v1"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Two train scenes in two areas with their tiles in <root>/aerial, and a test scene without a map origin."""
    root = tmp_path / "root"
    write_scene(root / "data" / "train" / "scene_frankfurt", FRANKFURT)
    write_scene(root / "data" / "train" / "scene_karlsruhe", KARLSRUHE)
    write_scene(root / "data" / "test" / "scene_test")  # the test splits ship no maps
    write_tiles(root)
    return root


def mod256_distance(a, b) -> int:
    d = int(a - b) % 256
    return min(d, 256 - d)


def metadata(root: Path, name: str) -> dict:
    return json.loads((root / "aerial" / "metadata" / name).read_text())


def pngs(root: Path) -> list[Path]:
    return sorted((root / "aerial" / "crop_cache").rglob("*.png"))


class TestPrepare:
    def test_raw_tiles_per_area_and_one_crop_per_frame(self, root: Path) -> None:
        ds = kitscenes.open(root)
        cache = ds.prepare_aerials(**ARGS, workers=1)

        assert cache.cache_id == CACHE_ID and cache.config.raw_cache_id == RAW_CACHE_ID
        frames = metadata(root, f"crop_manifest_{CACHE_ID}.json")["frames"]
        assert {s: set(f) for s, f in frames.items()} == {
            "scene_frankfurt": {"0", "1", "2"}, "scene_karlsruhe": {"0", "1", "2"},
        }
        assert len(pngs(root)) == 6
        entry = frames["scene_frankfurt"]["0"]["path"]
        assert not Path(entry).is_absolute()
        assert entry.startswith(f"crop_cache/{CACHE_ID}/train/area_")
        assert entry.endswith("/scene_frankfurt/0000000000.png")

    def test_crop_shape_and_attach(self, root: Path) -> None:
        ds = kitscenes.open(root, splits="train")
        cache = ds.prepare_aerials(**ARGS, workers=1)
        assert ds.aerial is cache
        crop = ds["scene_frankfurt"].frame(1).aerial()
        assert crop.shape == (4, 6, 3) and crop.dtype == np.uint8

    def test_crop_centre_is_the_frame_pose(self, root: Path) -> None:
        """End to end: origin.json + frame pose -> EPSG:25832 -> the crop's centre pixel."""
        ds = kitscenes.open(root, splits="train")
        ds.prepare_aerials(roi=(5.0, 3.0), workers=1)
        for scene_id, (lat, lon) in (("scene_frankfurt", FRANKFURT), ("scene_karlsruhe", KARLSRUHE)):
            e0, n0 = latlon_to_epsg25832(lat, lon)
            scene = ds[scene_id]
            for frame in scene:
                crop = frame.aerial()
                e = e0 + frame.pose.xyz[0]
                n = n0 + frame.pose.xyz[1]
                centre = crop[1, 2].astype(int)  # (3, 5) crop: the ego is pixel (1, 2)
                # bilinear between neighbouring 1 m pixels: within one of the floor value
                assert mod256_distance(centre[0], np.floor(e)) <= 1
                assert mod256_distance(centre[1], np.floor(n)) <= 1
                # heading east (yaw 0): forward = +column = growing easting
                assert (int(crop[1, 4, 0]) - int(crop[1, 0, 0])) % 256 == 4

    def test_heading_north_turns_the_crop(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        scene_dir = write_scene(root / "data" / "train" / "s", FRANKFURT, n=2)
        # one pose line per frame, both facing north (yaw 90 degrees)
        lines = scene_dir.joinpath("poses.txt").read_text().splitlines()
        q = f"0.0 0.0 {np.sin(np.pi / 4):.9f} {np.cos(np.pi / 4):.9f}"
        scene_dir.joinpath("poses.txt").write_text("".join(
            " ".join(line.split()[:3] + ["150.0", q]) + "\n" for line in lines
        ))
        write_tiles(root)
        ds = kitscenes.open(root)
        ds.prepare_aerials(roi=(5.0, 3.0), workers=1)
        crop = ds["s"].frame(0).aerial()
        # forward (+column) is north now: green (northing) grows to the right, red is constant along a row
        assert (int(crop[1, 4, 1]) - int(crop[1, 0, 1])) % 256 == 4
        assert crop[1, 0, 0] == crop[1, 4, 0]

    def test_rerun_writes_nothing(self, root: Path) -> None:
        ds = kitscenes.open(root)
        ds.prepare_aerials(**ARGS, workers=1)
        files = [p for p in (root / "aerial").rglob("*") if p.is_file()]
        before = {p: p.stat().st_mtime_ns for p in files}

        kitscenes.open(root).prepare_aerials(**ARGS, workers=1)

        assert {p: p.stat().st_mtime_ns for p in files} == before
        assert sorted(p for p in (root / "aerial").rglob("*") if p.is_file()) == sorted(files)

    def test_only_the_datasets_scenes(self, root: Path) -> None:
        ds = kitscenes.open(root).filter(scene_ids=["scene_karlsruhe"])
        ds.prepare_aerials(**ARGS, workers=1)
        assert set(metadata(root, f"crop_manifest_{CACHE_ID}.json")["frames"]) == {"scene_karlsruhe"}

    def test_parallel_matches_inline(self, tmp_path: Path) -> None:
        roots = []
        for name in ("a", "b"):
            r = tmp_path / name
            write_scene(r / "data" / "train" / "scene_frankfurt", FRANKFURT, yaw_rate=0.7)
            write_scene(r / "data" / "train" / "scene_karlsruhe", KARLSRUHE)
            write_tiles(r)
            roots.append(r)
        kitscenes.open(roots[0]).prepare_aerials(**ARGS, workers=1)
        kitscenes.open(roots[1]).prepare_aerials(**ARGS, workers=3)
        rel = [[p.relative_to(r) for p in pngs(r)] for r in roots]
        assert rel[0] == rel[1] and len(rel[0]) == 6
        for p in rel[0]:
            assert (roots[0] / p).read_bytes() == (roots[1] / p).read_bytes()
        manifests = [metadata(r, f"crop_manifest_{CACHE_ID}.json")["frames"] for r in roots]
        assert manifests[0] == manifests[1]

    def test_records_before_a_crash_survive(self, root: Path, monkeypatch) -> None:
        from kitscenes.aerial._crop import TileCoverageError

        from kitscenes.aerial import AerialCrops

        real = AerialCrops.crop
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 3:
                raise TileCoverageError("synthetic failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(AerialCrops, "crop", flaky)
        ds = kitscenes.open(root)
        with pytest.raises(TileCoverageError):
            ds.prepare_aerials(**ARGS, workers=1)  # one process: the counter is shared
        assert ds.aerial is None
        frames = metadata(root, f"crop_manifest_{CACHE_ID}.json")["frames"]
        assert sum(len(f) for f in frames.values()) == 3 and len(pngs(root)) == 3

        monkeypatch.setattr(AerialCrops, "crop", real)
        ds.prepare_aerials(**ARGS, workers=1)
        frames = metadata(root, f"crop_manifest_{CACHE_ID}.json")["frames"]
        assert sum(len(f) for f in frames.values()) == 6 and len(pngs(root)) == 6

    def test_relative_root(self, root: Path, monkeypatch) -> None:
        monkeypatch.chdir(root.parent)
        ds = kitscenes.open(root.name, splits="train")
        ds.prepare_aerials(**ARGS, workers=1)
        monkeypatch.chdir("/")
        assert ds["scene_karlsruhe"].frame(2).aerial().shape == (4, 6, 3)

class TestArguments:
    def test_without_tiles_nothing_is_written(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_scene(root / "data" / "train" / "scene_frankfurt", FRANKFURT)
        with pytest.raises(FileNotFoundError, match="No raw aerial tiles"):
            kitscenes.open(root).prepare_aerials(**ARGS, cache_dir=tmp_path / "cache")
        assert not (root / "aerial").exists() and not (tmp_path / "cache").exists()

    def test_no_georeferenced_scene(self, root: Path, tmp_path: Path) -> None:
        ds = kitscenes.open(root, splits="test")
        with pytest.raises(ValueError, match="map origin"):
            ds.prepare_aerials(**ARGS, cache_dir=tmp_path / "cache")
        assert not (tmp_path / "cache").exists()

    def test_tiles_of_some_scenes_only(self, tmp_path: Path) -> None:
        """Frames whose tiles are missing get no crop; the coverage names them."""
        root = tmp_path / "root"
        write_scene(root / "data" / "train" / "scene_frankfurt", FRANKFURT)
        write_scene(root / "data" / "train" / "scene_karlsruhe", KARLSRUHE)
        write_tiles(root, scenes=["scene_frankfurt"])
        cache = kitscenes.open(root).prepare_aerials(**ARGS, workers=1)
        assert not cache.coverage.complete and set(cache.coverage.incomplete()) == {"scene_karlsruhe"}
        assert cache.scene_ids == ["scene_frankfurt"]


class TestReading:
    def test_not_prepared(self, root: Path) -> None:
        frame = kitscenes.open(root)["scene_frankfurt"].frame(0)
        with pytest.raises(AerialUnavailable) as err:
            frame.aerial()
        assert err.value.state is Availability.NOT_PREPARED
        assert "use_aerial(roi=" in str(err.value)
        assert frame.aerial(missing="none") is None  # like every accessor
        assert frame.availability("aerial") is Availability.NOT_PREPARED

    def test_test_split_and_missing_frames(self, root: Path) -> None:
        ds = kitscenes.open(root)
        ds.prepare_aerials(**ARGS, workers=1)
        frame = ds["scene_test"].frame(0)
        with pytest.raises(AerialUnavailable) as err:
            frame.aerial()
        assert err.value.state is Availability.WITHHELD_BY_SPLIT
        assert frame.aerial(missing="none") is None

        write_scene(root / "data" / "train" / "scene_new", FRANKFURT)
        ds = kitscenes.open(root)
        ds.use_aerial()
        with pytest.raises(AerialUnavailable) as err:
            ds["scene_new"].frame(0).aerial()
        assert err.value.state is Availability.DROPPED
        assert isinstance(err.value, LookupError) and isinstance(err.value, kitscenes.Unavailable)

    def test_test_splits_placed_with_the_city_origin(self, tmp_path: Path) -> None:
        """test and map_change_test ship no maps/origin.json, but poses in the city's map frame: placed with the
        city's origin. test-e2e (poses relative to the first pose) stays withheld."""
        from kitscenes.aerial import placement_origin
        from kitscenes.map.origin import CITY_ORIGINS
        from kitscenes.splits import scene_table

        table = scene_table().values()
        test_id = next(i.scene_id for i in table if i.split == "test" and i.city == "Karlsruhe")
        e2e_id = next(i.scene_id for i in table if i.split == "test-e2e")
        root = tmp_path / "root"
        write_scene(root / "data" / "test" / test_id)
        write_scene(root / "data" / "test-e2e" / e2e_id)
        write_tiles(root)
        ds = kitscenes.open(root)
        assert placement_origin(ds[test_id]) == CITY_ORIGINS["Karlsruhe"]
        ds.prepare_aerials(roi=(5.0, 3.0), workers=1)

        e0, n0 = latlon_to_epsg25832(*dataclasses.astuple(CITY_ORIGINS["Karlsruhe"]))
        scene = ds[test_id]
        assert scene.availability("aerial", 0) is Availability.PRESENT
        for frame in scene:
            centre = frame.aerial()[1, 2].astype(int)
            assert mod256_distance(centre[0], np.floor(e0 + frame.pose.xyz[0])) <= 1
            assert mod256_distance(centre[1], np.floor(n0 + frame.pose.xyz[1])) <= 1

        frame = ds[e2e_id].frame(0)
        with pytest.raises(AerialUnavailable, match="first pose") as err:
            frame.aerial()
        assert err.value.state is Availability.WITHHELD_BY_SPLIT

    def test_city_origins_are_the_release_origins(self) -> None:
        from kitscenes.map.origin import CITY_ORIGINS
        from kitscenes.splits import scene_table

        assert set(CITY_ORIGINS) == {i.city for i in scene_table().values() if i.city is not None}

    def test_use_aerial_in_a_new_dataset(self, root: Path) -> None:
        kitscenes.open(root).prepare_aerials(**ARGS, workers=1)
        ds = kitscenes.open(root, splits="train")
        scene = ds["scene_frankfurt"]  # made before the cache is attached
        assert ds.use_aerial().cache_id == CACHE_ID
        assert scene.frame(0).aerial().shape == (4, 6, 3)
        assert ds.filter(scene_ids=["scene_karlsruhe"])["scene_karlsruhe"].frame(1).aerial() is not None
        assert ds[:1].aerial is ds.aerial

    def test_use_aerial_choices(self, root: Path) -> None:
        ds = kitscenes.open(root)
        with pytest.raises(AerialUnavailable):
            ds.use_aerial()
        ds.prepare_aerials(**ARGS, workers=1)
        ds.prepare_aerials(**{**ARGS, "roi": (4.0, 2.0)}, workers=1)
        assert cache_ids(root / "aerial") == ["dop20_res100_roi4x2_ego_v1", CACHE_ID]
        with pytest.raises(ValueError, match="several"):
            kitscenes.open(root).use_aerial()
        with pytest.raises(AerialUnavailable):
            kitscenes.open(root).use_aerial("dop20_res020_roi60x30_ego_v1")
        ds2 = kitscenes.open(root)
        ds2.use_aerial(CACHE_ID)
        assert ds2["scene_frankfurt"].frame(0).aerial().shape == (4, 6, 3)
        assert ds.aerial.cache_id == "dop20_res100_roi4x2_ego_v1"  # the last prepared is attached
        assert ds["scene_frankfurt"].frame(0).aerial().shape == (2, 4, 3)

    def test_crops_by_frame_and_the_shortcut(self, root: Path) -> None:
        """crops[frame] is the primary form; frame.aerial() reads what is attached; missing= works through both."""
        ds = kitscenes.open(root)
        wide = ds.use_aerial(roi=(6.0, 4.0))
        narrow = AerialCrops(wide.tiles, (4.0, 2.0))  # a second ROI on the same dataset, not attached
        frame = ds["scene_frankfurt"].frame(1)
        assert wide[frame].shape == (4, 6, 3) and narrow[frame].shape == (2, 4, 3)
        np.testing.assert_array_equal(frame.aerial(), wide[frame])
        np.testing.assert_array_equal(wide.at(frame), wide[frame])

        test_frame = ds["scene_test"].frame(0)
        with pytest.raises(AerialUnavailable) as err:
            wide[test_frame]
        assert err.value.state is Availability.WITHHELD_BY_SPLIT
        assert wide.at(test_frame, missing="none") is None and test_frame.aerial(missing="none") is None
        with pytest.raises(ValueError, match="missing"):
            wide.at(frame, missing="skip")

        cache = ds.prepare_aerials(**ARGS, workers=1)  # a prepared cache reads the same way
        np.testing.assert_array_equal(cache[frame], frame.aerial())
        write_scene(root / "data" / "train" / "scene_new", FRANKFURT)
        new = kitscenes.open(root)["scene_new"].frame(0)
        with pytest.raises(AerialUnavailable) as err:
            cache[new]
        assert err.value.state is Availability.DROPPED and cache.at(new, missing="none") is None

    def test_pickles_into_workers(self, root: Path) -> None:
        ds = kitscenes.open(root, splits="train")
        ds.prepare_aerials(**ARGS, workers=1)
        frame = ds["scene_karlsruhe"].frame(2)
        clone = pickle.loads(pickle.dumps(frame))
        np.testing.assert_array_equal(clone.aerial(), frame.aerial())
        frames = pickle.loads(pickle.dumps(ds.frames()))
        assert len(frames) == 6 and frames[0].aerial().shape == (4, 6, 3)

    def test_payload_cache(self, root: Path) -> None:
        ds = kitscenes.open(root, splits="train", cache_mb=10)
        ds.prepare_aerials(**ARGS, workers=1)
        frame = ds["scene_frankfurt"].frame(0)
        assert frame.aerial() is frame.aerial()


class TestCacheFiles:
    def test_manifest_with_absolute_paths(self, root: Path) -> None:
        """Crop paths are relative to the cache folder; an absolute one is refused."""
        kitscenes.open(root).prepare_aerials(**ARGS, workers=1)
        path = root / "aerial" / "metadata" / f"crop_manifest_{CACHE_ID}.json"
        data = json.loads(path.read_text())
        data["frames"]["scene_frankfurt"]["0"]["path"] = str(root / "aerial" / data["frames"]["scene_frankfurt"]["0"]["path"])
        path.write_text(json.dumps(data))

        cache = AerialCache.open(root / "aerial")
        with pytest.raises(ValueError, match="not relative"):
            cache.crop_path("scene_frankfurt", 0)

    def test_cache_listing(self, root: Path) -> None:
        cache = kitscenes.open(root).prepare_aerials(**ARGS, workers=1)
        assert cache.scene_ids == ["scene_frankfurt", "scene_karlsruhe"]
        np.testing.assert_array_equal(cache.frame_indices("scene_karlsruhe"), [0, 1, 2])
        assert cache.has_crop("scene_karlsruhe", 2) and not cache.has_crop("scene_test", 0)
        assert len(cache.frame_indices("scene_test")) == 0
        with pytest.raises(AerialUnavailable, match="scene_test"):
            cache.crop_path("scene_test", 0)

    def test_moved_root(self, root: Path, tmp_path: Path) -> None:
        kitscenes.open(root).prepare_aerials(**ARGS, workers=1)
        moved = tmp_path / "moved"
        os.rename(root, moved)
        ds = kitscenes.open(moved)
        ds.use_aerial()
        assert ds["scene_frankfurt"].frame(0).aerial().shape == (4, 6, 3)
