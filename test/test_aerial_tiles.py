"""Raw aerial tiles in every layout (AerialTiles), the tile store, and crops on the fly (AerialCrops)."""

from __future__ import annotations

import json
import math
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes.aerial import AerialCrops, AerialTiles, TileCoverageError, build_tile_store
from kitscenes.aerial import tile_store as tile_store_module
from kitscenes.aerial._tile_io import ZipMembers
from kitscenes.errors import AerialUnavailable, Availability
from aerial_fixture import FRANKFURT, KARLSRUHE, write_scene
from aerial_fixture import write_tiles as write_scene_tiles
from aerial_tiles_fixture import AREA, RAW_ID, SIZE, TILES, pixels, write_aligned_zip, write_tiles


def files_under(path: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in path.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Layouts


@pytest.mark.parametrize("fmt", ["png", "jpeg"])
@pytest.mark.parametrize("layout", ["dir", "zip", "zip-deflated"])
def test_every_layout_reads_the_same_pixels(tmp_path: Path, fmt: str, layout: str) -> None:
    aerial = write_tiles(tmp_path / "aerial", fmt=fmt, layout=layout.split("-")[0],
                         deflate="deflated" in layout)
    before = files_under(tmp_path)
    tiles = AerialTiles(tmp_path)  # a dataset root: its aerial/
    assert tiles.roots == [aerial.resolve()] and tiles.area_ids == [AREA] and tiles.num_tiles == 9
    assert tiles.formats == {fmt} and tiles.resolution_m == 1.0 and tiles.tile_size_px == SIZE
    for i, j in TILES:
        got = tiles.tile(AREA, i, j)
        assert got.shape == (SIZE, SIZE, 3) and got.dtype == np.uint8
        if fmt == "png":
            np.testing.assert_array_equal(got, pixels(i, j))
        else:
            assert np.abs(got.astype(int) - pixels(i, j)).max() <= 6
    assert files_under(tmp_path) == before  # read in place: nothing extracted or written


def test_jpeg_from_a_zip_equals_the_extracted_jpeg(tmp_path: Path) -> None:
    a = AerialTiles(write_tiles(tmp_path / "a", fmt="jpeg", layout="zip"))
    b = AerialTiles(write_tiles(tmp_path / "b", fmt="jpeg", layout="dir"))
    loc = a.location(AREA, 1, 2)
    assert loc.member == f"{AREA}/tile_1_2.jpg" and loc.path.endswith(f"{AREA}.zip")
    for i, j in TILES:
        np.testing.assert_array_equal(a.tile(AREA, i, j), b.tile(AREA, i, j))


def test_zip_members_by_offset(tmp_path: Path) -> None:
    path = tmp_path / "x.zip"
    write_aligned_zip(path, {"a/one.bin": b"1" * 5000, "a/two.bin": b"22"})
    members = ZipMembers(path)
    assert members.read("a/one.bin") == b"1" * 5000 and members.read("a/two.bin") == b"22"
    assert members._data_offset(members.infos["a/two.bin"]) % 4096 == 0
    members.close()


def test_a_zip_is_preferred_over_an_extracted_folder(tmp_path: Path) -> None:
    aerial = write_tiles(tmp_path / "aerial", layout="zip")
    write_tiles(tmp_path / "aerial", layout="dir", content=lambda i, j, o: np.zeros((SIZE, SIZE, 3), np.uint8))
    np.testing.assert_array_equal(AerialTiles(aerial).tile(AREA, 0, 0), pixels(0, 0))
    (aerial / "raw_tiles" / RAW_ID / f"{AREA}.zip").unlink()
    assert AerialTiles(aerial).tile(AREA, 0, 0).max() == 0


def test_no_tiles(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="unless --no-aerial"):
        AerialTiles(tmp_path)


# ---------------------------------------------------------------------------
# Several folders


def test_folders_are_merged(tmp_path: Path) -> None:
    west = write_tiles(tmp_path / "west", tiles=[(0, 0), (0, 1)], scenes={"s1": AREA})
    east = write_tiles(tmp_path / "east", fmt="jpeg", layout="zip", tiles=[(0, 0), (1, 0)], scenes={"s2": AREA})
    other = write_tiles(tmp_path / "other", area="area_9_9", origin=(9000.0, 9000.0), tiles=[(0, 0)],
                        scenes={"s3": "area_9_9"})
    tiles = AerialTiles([west, east, other])
    assert tiles.area_ids == [AREA, "area_9_9"] and tiles.tiles(AREA) == {(0, 0), (0, 1), (1, 0)}
    assert tiles.location(AREA, 0, 0).format == "png"  # the first folder that lists it
    assert tiles.location(AREA, 1, 0).format == "jpeg" and tiles.formats == {"png", "jpeg"}
    assert tiles.scene_to_area == {"s1": AREA, "s2": AREA, "s3": "area_9_9"}
    assert tiles.area_of("s3") == "area_9_9" and tiles.area_of("unknown") is None
    assert len(tiles.attribution) == 3

    elsewhere = write_tiles(tmp_path / "shifted", origin=(1000.0, 2001.0), tiles=[(0, 0)])
    with pytest.raises(ValueError, match="another tile grid"):
        AerialTiles([west, elsewhere])
    moved = write_tiles(tmp_path / "moved", tiles=[(0, 0)], scenes={"s1": "area_9_9"})
    with pytest.raises(ValueError, match="Scene s1"):
        AerialTiles([west, moved])


def test_several_tile_sets_need_a_choice(tmp_path: Path) -> None:
    aerial = write_tiles(tmp_path / "aerial")
    manifest = aerial / "metadata" / f"raw_tile_manifest_{RAW_ID}.json"
    (aerial / "metadata" / "raw_tile_manifest_dop20_res020_tile2048.json").write_text(manifest.read_text())
    with pytest.raises(ValueError, match="raw_cache_id"):
        AerialTiles(aerial)
    assert AerialTiles(aerial, raw_cache_id=RAW_ID).raw_cache_id == RAW_ID


# ---------------------------------------------------------------------------
# Crops


POSES = [((1048.0, 2048.0), 0.0), ((1031.3, 2052.7), 0.7), ((1050.2, 2041.9), -2.3), ((1047.5, 2047.5), math.pi / 2)]


@pytest.mark.parametrize("xy, yaw", POSES)
def test_crops_from_every_tile_source_agree(tmp_path: Path, xy, yaw) -> None:
    """PNG tiles with scipy are the reference (its geometry: test_aerial_crop.py)."""
    aerial = write_tiles(tmp_path / "aerial")
    exact = AerialCrops(aerial, (20.0, 10.0), backend="scipy")
    assert exact.cache_id == "dop20_res100_roi20x10_ego_v1" and exact.shape == (10, 20, 3)
    want = exact.crop(xy, yaw)
    assert want.shape == exact.shape and want.dtype == np.uint8
    fast = AerialCrops(aerial, (20.0, 10.0), backend="cv2").crop(xy, yaw)
    assert np.abs(fast.astype(int) - want).max() <= 1

    store = build_tile_store(tmp_path / "store", aerial, jobs=1)
    np.testing.assert_array_equal(AerialCrops(store, (20.0, 10.0), backend="scipy").crop(xy, yaw), want)
    jpeg = AerialCrops(write_tiles(tmp_path / "jpeg", fmt="jpeg", layout="zip"), (20.0, 10.0)).crop(xy, yaw)
    assert np.abs(jpeg.astype(int) - want).mean() < 2


def test_coverage_of_a_crop(tmp_path: Path) -> None:
    crops = AerialCrops(write_tiles(tmp_path / "aerial"), (20.0, 10.0))
    assert crops.covers((1048.0, 2048.0)) and not crops.covers((1090.0, 2048.0))
    with pytest.raises(TileCoverageError):
        crops.crop((1090.0, 2048.0), 0.0)
    with pytest.raises(ValueError, match="backend"):
        AerialCrops(crops.tiles, (20.0, 10.0), backend="torch")


def test_the_decoded_tile_cache_is_bounded(tmp_path: Path) -> None:
    aerial = write_tiles(tmp_path / "aerial", fmt="jpeg", layout="zip")
    tile_bytes = SIZE * SIZE * 3
    tiles = AerialTiles(aerial, cache_mb=2.5 * tile_bytes / (1 << 20))
    for i, j in TILES:
        tiles.tile(AREA, i, j)
    decoded = tiles._reader.decoded
    assert len(decoded) == 2 and decoded.nbytes == 2 * tile_bytes and decoded.misses == 9
    assert tiles.tile(AREA, *TILES[-1]) is tiles.tile(AREA, *TILES[-1])
    assert decoded.hits == 2

    uncached = AerialTiles(aerial, cache_mb=0)
    assert uncached.tile(AREA, 0, 0) is not uncached.tile(AREA, 0, 0)
    assert len(uncached._reader.decoded) == 0


def _crop_in_child(args):
    crops, xy, yaw = args
    return crops.crop(xy, yaw)


def test_pickles_small_and_forks_clean(tmp_path: Path) -> None:
    crops = AerialCrops(write_tiles(tmp_path / "aerial", fmt="jpeg", layout="zip"), (20.0, 10.0))
    want = [crops.crop(xy, yaw) for xy, yaw in POSES]  # the parent has open zips and decoded tiles now
    assert len(crops.tiles._reader.decoded) > 0
    blob = pickle.dumps(crops)
    assert len(blob) < 2000  # folders and parameters, no pixels or handles
    clone = pickle.loads(blob)
    assert len(clone.tiles._reader.decoded) == 0
    np.testing.assert_array_equal(clone.crop(*POSES[1]), want[1])

    with multiprocessing.get_context("fork").Pool(2) as pool:  # DataLoader workers fork the dataset as it is
        results = pool.map(_crop_in_child, [(crops, xy, yaw) for xy, yaw in POSES])
    for got, expected in zip(results, want):  # each child opens its own zips
        np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Tile store


def test_tile_store_from_release_zips(tmp_path: Path) -> None:
    aerial = write_tiles(tmp_path / "aerial", fmt="jpeg", layout="zip", scenes={"s1": AREA})
    source = AerialTiles(aerial)
    store_dir = tmp_path / "store"
    store = build_tile_store(store_dir, aerial, jobs=2)
    assert store.formats == {"npy"} and store.num_tiles == 9 and store.scene_to_area == {"s1": AREA}
    assert (store_dir / "ATTRIBUTION.md").read_text() == "test imagery\n"
    manifest = json.loads((store_dir / "metadata" / f"raw_tile_manifest_{RAW_ID}.json").read_text())
    assert manifest["tile_format"]["format"] == "npy" and "Pillow" in manifest["tile_format"]["decoder"]
    entry = manifest["areas"][AREA]["tiles"][0]
    assert entry["path"] == f"raw_tiles/{RAW_ID}/{AREA}/tile_0_0.npy" and len(entry["source"]) == 2
    for i, j in TILES:
        np.testing.assert_array_equal(store.tile(AREA, i, j), source.tile(AREA, i, j))
        assert isinstance(store.tile(AREA, i, j), np.memmap)

    # idempotent: a rerun converts nothing and touches no tile
    before = {p: t for p, t in files_under(store_dir).items() if p.suffix == ".npy"}
    build_tile_store(store_dir, aerial, jobs=1)
    assert {p: t for p, t in files_under(store_dir).items() if p.suffix == ".npy"} == before

    # a missing or truncated tile, and a tile whose source changed, are converted again
    tile_dir = store_dir / "raw_tiles" / RAW_ID / AREA
    (tile_dir / "tile_0_0.npy").unlink()
    with open(tile_dir / "tile_1_1.npy", "r+b") as fh:
        fh.truncate(100)
    write_tiles(tmp_path / "aerial", fmt="jpeg", layout="zip", scenes={"s1": AREA},
                content=lambda i, j, o: pixels(i, j, o) // 2 if (i, j) == (2, 2) else pixels(i, j, o))
    build_tile_store(store_dir, aerial, jobs=1)
    after = {p: t for p, t in files_under(store_dir).items() if p.suffix == ".npy"}
    changed = sorted(p.name for p in after if after[p] != before.get(p))
    assert changed == ["tile_0_0.npy", "tile_1_1.npy", "tile_2_2.npy"]
    np.testing.assert_array_equal(AerialTiles(store_dir).tile(AREA, 2, 2), AerialTiles(aerial).tile(AREA, 2, 2))


def test_tile_store_resumes_after_a_failure(tmp_path: Path, monkeypatch) -> None:
    aerial = write_tiles(tmp_path / "aerial")
    real = tile_store_module._convert
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        return "synthetic failure" if calls["n"] == 5 else real(*args)

    monkeypatch.setattr(tile_store_module, "_convert", flaky)
    with pytest.raises(ValueError, match="1 tile.*synthetic failure"):
        build_tile_store(tmp_path / "store", aerial, jobs=1)
    assert AerialTiles(tmp_path / "store").num_tiles == 8  # the others are recorded
    killed = tmp_path / "store" / "raw_tiles" / RAW_ID / AREA / "tile_0_0.npy.partial123"
    killed.write_bytes(b"half")  # a worker killed while writing
    monkeypatch.setattr(tile_store_module, "_convert", real)
    calls["n"] = 0
    monkeypatch.setattr(tile_store_module, "_convert", lambda *a: (calls.__setitem__("n", calls["n"] + 1), real(*a))[1])
    assert build_tile_store(tmp_path / "store", aerial, jobs=1).num_tiles == 9
    assert calls["n"] == 1 and not killed.exists()


def test_tile_store_checks(tmp_path: Path) -> None:
    aerial = write_tiles(tmp_path / "aerial")
    with pytest.raises(ValueError, match="cannot be one of its sources"):
        build_tile_store(aerial, aerial)
    with pytest.raises(ValueError, match="Unknown area"):
        build_tile_store(tmp_path / "store", aerial, areas=["area_x"])
    bad = write_tiles(tmp_path / "bad", content=lambda i, j, o: np.zeros((8, 8, 3), np.uint8), tiles=[(0, 0)])
    with pytest.raises(ValueError, match=r"\(8, 8, 3\)"):
        build_tile_store(tmp_path / "store2", bad, jobs=1)
    store = build_tile_store(tmp_path / "store3", [aerial, write_tiles(
        tmp_path / "far", area="area_9_9", origin=(9000.0, 9000.0), tiles=[(0, 0)])], areas=["area_9_9"], jobs=1)
    assert store.area_ids == ["area_9_9"]


# ---------------------------------------------------------------------------
# Frames: on the fly == prepared, attach, coverage


def test_on_the_fly_crops_are_the_prepared_crops(tmp_path: Path) -> None:
    roi = (6.0, 4.0)
    root = tmp_path / "root"
    write_scene(root / "data" / "train" / "f", FRANKFURT, yaw_rate=0.7)
    write_scene(root / "data" / "train" / "k", KARLSRUHE)
    write_scene(root / "data" / "test" / "t")
    tiles_dir = write_scene_tiles(root, tmp_path / "tiles")  # the root stays untouched
    tiles_before = files_under(tiles_dir)
    ds = kitscenes.open(root)
    cache = ds.prepare_aerials(roi, tiles=tiles_dir, cache_dir=tmp_path / "cache", backend="scipy", workers=2)
    assert not (root / "aerial").exists() and files_under(tiles_dir) == tiles_before  # tiles are only read
    assert cache.root == (tmp_path / "cache").resolve()
    assert sorted(p.name for p in (tmp_path / "cache" / "metadata").iterdir()) == [
        f"coverage_{cache.cache_id}.json", f"crop_manifest_{cache.cache_id}.json"]
    assert cache.coverage.complete and cache.coverage.num_covered == 6
    prepared = {(f.scene.id, f.index): f.aerial() for f in ds.frames() if f.scene.split == "train"}

    store = build_tile_store(tmp_path / "store", tiles_dir, jobs=1)
    for tiles in (tiles_dir, store):
        fresh = kitscenes.open(root)
        crops = fresh.use_aerial(roi=roi, tiles=tiles, backend="scipy")
        assert isinstance(crops, AerialCrops) and crops.cache_id == cache.cache_id
        for key, want in prepared.items():
            np.testing.assert_array_equal(crops[fresh[key[0]].frame(key[1])], want)
            np.testing.assert_array_equal(fresh[key[0]].frame(key[1]).aerial(), want)
        frame = fresh["t"].frame(0)
        assert frame.availability("aerial") is Availability.WITHHELD_BY_SPLIT
        clone = pickle.loads(pickle.dumps(fresh["f"].frame(2)))
        np.testing.assert_array_equal(clone.aerial(), prepared[("f", 2)])

    # tiles for Frankfurt only: Karlsruhe frames are DROPPED, and the coverage says so
    only_f = AerialTiles(tiles_dir)
    area_f = only_f.area_of("f")
    store_f = build_tile_store(tmp_path / "store_f", tiles_dir, areas=[area_f], jobs=1)
    fresh = kitscenes.open(root)
    crops = fresh.use_aerial(roi=roi, tiles=store_f)
    with pytest.raises(AerialUnavailable) as err:
        fresh["k"].frame(0).aerial()
    assert err.value.state is Availability.DROPPED and "no crop for this frame" in str(err.value)
    report = crops.coverage(fresh)
    assert not report.complete and report.num_covered == 3 and report.num_uncovered == 3
    assert set(report.incomplete()) == {"k"} and report.scenes["t"].reason.startswith("no map origin")
    assert json.loads(report.to_json())["scenes"]["k"]["uncovered_frames"] == [0, 1, 2]

    with pytest.raises(ValueError, match="either"):
        fresh.use_aerial(cache.cache_id, roi=(6.0, 4.0))
    with pytest.raises(ValueError, match="tiles= goes with roi="):
        fresh.use_aerial(tiles=store_f)
