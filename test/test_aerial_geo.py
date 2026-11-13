"""kitscenes.aerial._geo: scene frame -> EPSG:25832 and the heading."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kitscenes.aerial._geo import heading_from_rotation, latlon_to_epsg25832, scene_xy_to_epsg25832
from kitscenes.aerial.config import AerialCacheConfig
from kitscenes.errors import AerialUnavailable, Availability
from kitscenes.map import MapOrigin

FRANKFURT = MapOrigin(50.110423, 8.682138)


def test_heading_of_a_pitched_and_rolled_pose_is_where_the_forward_axis_points() -> None:
    """An extrinsic 'zyx' Euler angle would be off here (0.57 instead of 0.6)."""
    rot = Rotation.from_euler("ZYX", [0.6, 0.3, 0.2])
    forward = rot.apply([1.0, 0.0, 0.0])
    assert float(heading_from_rotation(rot)) == pytest.approx(math.atan2(forward[1], forward[0]))
    assert float(heading_from_rotation(rot)) == pytest.approx(0.6)
    np.testing.assert_allclose(heading_from_rotation(Rotation.from_euler("z", [[0.1], [-2.0], [3.0]])), [0.1, -2.0, 3.0])


def test_heading_identity_and_quarter_turn() -> None:
    assert float(heading_from_rotation(Rotation.identity())) == pytest.approx(0.0)
    quat = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    assert float(heading_from_rotation(Rotation.from_quat(quat))) == pytest.approx(math.pi / 2)


def test_latlon_to_epsg25832_known_point() -> None:
    easting, northing = latlon_to_epsg25832(50.110423, 8.682138)
    assert easting == pytest.approx(477272.0, abs=1.0)
    assert northing == pytest.approx(5550957.0, abs=1.0)


def test_scene_frame_is_a_pure_translation() -> None:
    world = scene_xy_to_epsg25832(FRANKFURT, np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [0.0, 10.0, 5.0]]))
    assert world.shape == (3, 2)
    np.testing.assert_allclose(world[0], latlon_to_epsg25832(FRANKFURT.lat, FRANKFURT.lon), atol=1e-9)
    np.testing.assert_allclose(world[1] - world[0], [10.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(world[2] - world[0], [0.0, 10.0], atol=1e-6)


def test_epsg25832_matches_the_map_frames_utm_zone() -> None:
    """The map frame is UTM 32N (WGS 84); the imagery is EPSG:25832 (ETRS89): the same within a millimetre."""
    for origin in (FRANKFURT, MapOrigin(49.006889, 8.403653), MapOrigin(48.7, 9.0)):
        assert origin.epsg == 32632
        xy = np.array([[0.0, 0.0], [1500.0, -800.0]])
        np.testing.assert_allclose(scene_xy_to_epsg25832(origin, xy), origin.to_utm(xy), atol=1e-3)


def test_config_is_frozen() -> None:
    import dataclasses

    config = AerialCacheConfig(
        provider="dop20", resolution_m=0.20, roi_m=(60.0, 30.0), tile_size_px=2048, orientation="ego",
        version="v1", cache_id="dop20_res020_roi60x30_ego_v1", raw_cache_id="dop20_res020_tile2048",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.cache_id = "other"


def test_error_pickles() -> None:
    original = AerialUnavailable("aerial", 3, Availability.DROPPED, "no crop in cache x")
    err = pickle.loads(pickle.dumps(original))
    assert err.state is Availability.DROPPED and err.index == 3 and str(err) == str(original)
    assert "frame 3" in str(err) and "dropped" in str(err)
    assert isinstance(err, LookupError)
