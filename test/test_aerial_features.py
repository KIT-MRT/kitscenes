"""DINOv3 feature tiles of the aerial imagery (AerialFeatures) and ego-aligned feature crops (AerialFeatureCrops)."""

from __future__ import annotations

import math
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest

from kitscenes.aerial import AerialFeatureCrops, AerialFeatures, TileCoverageError
from kitscenes.aerial.features import bf16_to_float32, feature_set_ids, warp_tokens_ego_aligned

from aerial_features_fixture import (AREA, CHANNELS, LAYERS, ORIGIN, SET, TOKEN_M, TOKENS, float32_to_bf16, token_field,
                                     write_features)
from aerial_fixture import FRANKFURT, write_scene


def test_bf16_conversion_is_exact() -> None:
    values = np.array([0.0, 1.0, -2.5, 3.140625, 1e30, -1e-30], dtype=np.float32)
    assert np.array_equal(bf16_to_float32(float32_to_bf16(values)), float32_to_bf16(values).astype(np.uint32)
                          .__lshift__(16).view(np.float32))
    assert bf16_to_float32(float32_to_bf16(np.float32([1.0, -2.5])))[1] == -2.5


@pytest.mark.parametrize("layout", ["dir", "zip"])
def test_tiles_and_layers(tmp_path: Path, layout: str) -> None:
    feats = AerialFeatures(write_features(tmp_path, layout=layout))
    assert feats.layers == LAYERS and feats.channels == CHANNELS and feats.tokens == TOKENS
    assert feats.token_m == TOKEN_M and feats.num_tiles == 9 and feats.area_ids == [AREA]
    assert np.array_equal(feats.tile(AREA, 1, 2), token_field(1, 2))
    assert np.array_equal(feats.tile(AREA, 1, 2, layers=[11]), token_field(1, 2)[1:])
    assert np.allclose(feats.cls(AREA, 0, 0), token_field(0, 0).mean(axis=(2, 3)), atol=0.05)
    assert feats.tile_metadata(AREA, 1, 2) == {"i": 1, "j": 2}
    assert feats.terms is not None and feats.tile_info(AREA, 0, 0) == {"missing_neighbors": []}
    with pytest.raises(ValueError, match="not stored"):
        feats.tile(AREA, 0, 0, layers=[23])
    with pytest.raises(KeyError):
        feats.tile(AREA, 5, 5)


def test_zip_is_preferred_and_found_from_any_level(tmp_path: Path) -> None:
    write_features(tmp_path, layout="zip")
    assert AerialFeatures(tmp_path).location(AREA, 0, 0).member == f"{AREA}/tile_0_0.safetensors"
    assert feature_set_ids(tmp_path) == [SET]
    for path in (tmp_path / "aerial", tmp_path / "aerial" / "features", tmp_path / "aerial" / "features" / SET):
        assert AerialFeatures(path).feature_set == SET
    with pytest.raises(FileNotFoundError, match="with --aerial-features"):
        AerialFeatures(tmp_path / "nothing")


def test_window_spans_tiles(tmp_path: Path) -> None:
    feats = AerialFeatures(write_features(tmp_path))
    # 6 x 4 tokens across tiles (0..1, 1..2): easting 1012..1024, northing 2028..2036
    bbox = (ORIGIN[0] + 12.0, ORIGIN[1] + 28.0, ORIGIN[0] + 24.0, ORIGIN[1] + 36.0)
    win = feats.window(AREA, bbox, layers=[4])
    assert win.shape == (1, CHANNELS, 4, 6)
    assert np.array_equal(win[0, 0, 0], 12.0 + 1.0 + 2.0 * np.arange(6))  # token centres, west to east
    assert np.array_equal(win[0, 1, :, 0], 36.0 - 1.0 - 2.0 * np.arange(4))  # north to south
    with pytest.raises(TileCoverageError):
        feats.window(AREA, (ORIGIN[0] - 4.0, ORIGIN[1], ORIGIN[0] + 4.0, ORIGIN[1] + 4.0))


@pytest.mark.parametrize("yaw", [0.0, 0.7, math.pi / 2, -2.4])
def test_crop_geometry(tmp_path: Path, yaw: float) -> None:
    """Bilinear sampling reproduces a linear field: every crop cell holds its own world position."""
    crops = AerialFeatureCrops(write_features(tmp_path), roi=(10.0, 6.0), layers=LAYERS, resolution_m=1.0,
                               orientation="ego")
    ego = (ORIGIN[0] + 23.3, ORIGIN[1] + 24.9)
    crop = crops.crop(ego, yaw)
    assert crop.shape == crops.shape == (2, CHANNELS, 6, 10)
    rows, cols = np.meshgrid(np.arange(6.0), np.arange(10.0), indexing="ij")
    fwd, left = cols - 4.5, -(rows - 2.5)  # forward +column, left -row, ego at the centre
    e = ego[0] + math.cos(yaw) * fwd - math.sin(yaw) * left - ORIGIN[0]
    n = ego[1] + math.sin(yaw) * fwd + math.cos(yaw) * left - ORIGIN[1]
    assert np.allclose(crop[0, 0], e, atol=1e-3) and np.allclose(crop[0, 1], n, atol=1e-3)
    assert np.allclose(crop[1, 2], 1.0)


def test_default_layer_and_cache_id(tmp_path: Path) -> None:
    crops = AerialFeatureCrops(write_features(tmp_path), roi=(10.0, 6.0), orientation="ego")
    assert crops.layers == (11,) and crops.resolution_m == TOKEN_M and crops.shape == (1, CHANNELS, 3, 5)
    assert crops.cache_id == f"{SET}_roi10x6_ego_l11_r200"
    north = AerialFeatureCrops(crops.features, roi=(10.0, 6.0))  # the default orientation
    assert north.orientation == "north" and north.cache_id == f"{SET}_roi10x6_north_l11_r200"
    with pytest.raises(ValueError, match="not stored"):
        AerialFeatureCrops(crops.features, roi=(10.0, 6.0), layers=[1])


def test_coverage(tmp_path: Path) -> None:
    crops = AerialFeatureCrops(write_features(tmp_path), roi=(10.0, 6.0), orientation="ego")
    assert crops.covers((ORIGIN[0] + 24.0, ORIGIN[1] + 24.0))
    assert not crops.covers((ORIGIN[0] + 2.0, ORIGIN[1] + 24.0))  # reaches west of tile column 0
    with pytest.raises(TileCoverageError):
        crops.crop((ORIGIN[0] + 2.0, ORIGIN[1] + 24.0), 0.0)


def test_warp_matches_identity_at_token_centres() -> None:
    source = np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7)
    bbox = (0.0, 0.0, 7.0, 5.0)  # 1 m tokens
    out = warp_tokens_ego_aligned(source, bbox, 1.0, (3.5, 2.5), 0.0, (3.0, 3.0), 1.0)
    assert np.array_equal(out, source[:, 1:4, 2:5])


def _read_in_child(feats, conn) -> None:
    conn.send(float(feats.tile(AREA, 2, 2)[0, 0, 0, 0]))
    conn.close()


def test_pickles_small_and_forks_clean(tmp_path: Path) -> None:
    feats = AerialFeatures(write_features(tmp_path, layout="zip"))
    first = feats.tile(AREA, 2, 2)  # opens the zip and a memory map in this process
    blob = pickle.dumps(feats)
    assert len(blob) < 1000
    assert np.array_equal(pickle.loads(blob).tile(AREA, 2, 2), first)
    crops = pickle.loads(pickle.dumps(AerialFeatureCrops(feats, roi=(10.0, 6.0), layers=[4], orientation="ego")))
    assert crops.layers == (4,) and crops.crop((ORIGIN[0] + 24.0, ORIGIN[1] + 24.0), 0.3).shape == (1, 3, 3, 5)
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_read_in_child, args=(feats, child))
    proc.start()
    assert parent.recv() == float(first[0, 0, 0, 0])
    proc.join(10)
    assert proc.exitcode == 0


def test_frames_read_feature_crops(tmp_path: Path) -> None:
    import kitscenes
    from kitscenes.aerial import scene_poses
    from kitscenes.aerial._areas import bucket_area_id
    from kitscenes.errors import AerialUnavailable, Availability
    root = tmp_path / "root"
    write_scene(root / "data" / "train" / "f", FRANKFURT, yaw_rate=0.7)
    write_scene(root / "data" / "test" / "t")
    ds = kitscenes.open(root)
    poses = scene_poses(ds["f"])  # the tiles go around the frames (the poses start away from the map origin)
    e, n = poses.xy.mean(axis=0)
    area = bucket_area_id(*FRANKFURT)
    origin = (math.floor(e / TOKEN_M) * TOKEN_M - 24.0, math.floor(n / TOKEN_M) * TOKEN_M - 24.0)
    write_features(root, area=area, origin=origin)

    frame = ds["f"].frame(1)
    with pytest.raises(AerialUnavailable) as err:
        frame.aerial_features()
    assert err.value.state is Availability.NOT_PREPARED and "use_aerial_features" in str(err.value)

    crops = ds.use_aerial_features(roi=(10.0, 6.0), layers=LAYERS, resolution_m=1.0, orientation="ego")
    for index in range(3):
        k = poses.position(index)
        want = crops.crop(tuple(poses.xy[k]), float(poses.yaw[k]))
        got = ds["f"].frame(index).aerial_features()
        assert got.shape == (2, CHANNELS, 6, 10) and np.array_equal(got, want)
        assert np.array_equal(crops[ds["f"].frame(index)], want)  # the explicit form
        assert np.allclose(got[0, 0, 3, 5] + origin[0], poses.xy[k, 0], atol=1.0)  # near the ego
    assert ds["t"].frame(0).availability("aerial_features") is Availability.WITHHELD_BY_SPLIT
    assert ds["t"].frame(0).aerial_features(missing="none") is None
    assert crops.at(ds["t"].frame(0), missing="none") is None
    with pytest.raises(AerialUnavailable) as err:
        crops[ds["t"].frame(0)]
    assert err.value.what == "aerial_features" and err.value.state is Availability.WITHHELD_BY_SPLIT
    clone = pickle.loads(pickle.dumps(ds["f"].frame(2)))
    assert np.array_equal(clone.aerial_features(), ds["f"].frame(2).aerial_features())
    sub = ds.filter(splits="train")
    assert sub.aerial_features is crops
    with pytest.raises(ValueError, match="either"):
        ds.use_aerial_features()


@pytest.mark.parametrize("xy", [(23.3, 24.9), (25.9, 20.1), (24.0, 24.0)])
def test_north_up_window(tmp_path: Path, xy) -> None:
    from kitscenes.aerial import NorthUpFeatures

    feats = AerialFeatures(write_features(tmp_path))
    crops = AerialFeatureCrops(feats, roi=(10.0, 6.0), layers=LAYERS, orientation="north")
    ego, yaw = (ORIGIN[0] + xy[0], ORIGIN[1] + xy[1]), 0.7
    win = crops.crop(ego, yaw)
    assert isinstance(win, NorthUpFeatures) and win.shape == crops.shape == (2, CHANNELS, 9, 9)
    assert np.array_equal(np.asarray(win), feats.window(AREA, win.bbox_epsg25832, LAYERS))  # stored tokens, unchanged
    r, c = win.ego_rc
    assert 4.0 <= r <= 5.0 and 4.0 <= c <= 5.0  # the ego lies in the centre token (or on its edge)
    # every token centre in the ego frame: from the encoded world position, and from the helper
    centres = win.token_centers_ego()
    de, dn = win.tokens[0, 0] + ORIGIN[0] - ego[0], win.tokens[0, 1] + ORIGIN[1] - ego[1]
    assert np.allclose(centres[..., 0], math.cos(yaw) * de + math.sin(yaw) * dn, atol=1e-3)
    assert np.allclose(centres[..., 1], -math.sin(yaw) * de + math.cos(yaw) * dn, atol=1e-3)
    rc = win.ego_to_tokens(centres.reshape(-1, 2)).reshape(9, 9, 2)
    assert np.allclose(rc[..., 0], np.arange(9)[:, None] + 0.5, atol=1e-4)
    assert np.allclose(rc[..., 1], np.arange(9)[None, :] + 0.5, atol=1e-4)
    # the window holds the ROI at any heading
    assert np.hypot(*centres.reshape(-1, 2).T).max() >= 0.5 * math.hypot(10.0, 6.0) + TOKEN_M / 2


def test_north_up_grid_sample_gives_the_ego_crop(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    feats = AerialFeatures(write_features(tmp_path))
    ego, yaw = (ORIGIN[0] + 23.3, ORIGIN[1] + 24.9), -2.1
    win = AerialFeatureCrops(feats, roi=(10.0, 6.0), layers=LAYERS, orientation="north").crop(ego, yaw)
    ego_crop = AerialFeatureCrops(feats, roi=(10.0, 6.0), layers=LAYERS, resolution_m=1.0,
                                  orientation="ego").crop(ego, yaw)
    grid = torch.from_numpy(win.sampling_grid((10.0, 6.0), 1.0))[None]
    t = torch.from_numpy(win.tokens).reshape(1, -1, *win.tokens.shape[-2:])
    out = torch.nn.functional.grid_sample(t, grid, mode="bilinear", align_corners=False, padding_mode="border")
    assert np.allclose(out.reshape(ego_crop.shape).numpy(), ego_crop, atol=1e-3)


def test_north_up_frames_and_options(tmp_path: Path) -> None:
    import kitscenes
    from kitscenes.aerial import NorthUpFeatures, scene_poses
    from kitscenes.aerial._areas import bucket_area_id
    root = tmp_path / "root"
    write_scene(root / "data" / "train" / "f", FRANKFURT, yaw_rate=0.7)
    ds = kitscenes.open(root)
    poses = scene_poses(ds["f"])
    e, n = poses.xy.mean(axis=0)
    write_features(root, area=bucket_area_id(*FRANKFURT),
                   origin=(math.floor(e / TOKEN_M) * TOKEN_M - 24.0, math.floor(n / TOKEN_M) * TOKEN_M - 24.0))
    crops = ds.use_aerial_features(roi=(10.0, 6.0))
    assert "_north_" in crops.cache_id and crops.orientation == "north"
    win = ds["f"].frame(1).aerial_features()
    k = poses.position(1)
    assert isinstance(win, NorthUpFeatures) and win.yaw == pytest.approx(float(poses.yaw[k]))
    assert win.ego_xy == pytest.approx(tuple(poses.xy[k]))
    clone = pickle.loads(pickle.dumps(crops))
    assert clone.orientation == "north" and np.array_equal(np.asarray(clone.crop(win.ego_xy, win.yaw)), win.tokens)
    with pytest.raises(ValueError, match="resolution_m"):
        AerialFeatureCrops(crops.features, roi=(10.0, 6.0), orientation="north", resolution_m=1.0)
    with pytest.raises(ValueError, match="orientation"):
        AerialFeatureCrops(crops.features, roi=(10.0, 6.0), orientation="up")
