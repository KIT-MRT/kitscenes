"""kitscenes.sdmap on a real dataset copy (read-only); run with ``KITSCENES_ROOT=<root> pytest``.

The tests need the SD maps of Sindelfingen and its scenes, so they run on the
full release only. The parsed maps are cached in a temporary folder.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import kitscenes
from kitscenes.map import Region
from kitscenes.navigation import KITSCENES, NAVSIM, driving_commands

from kitscenes.sdmap._files import PAPER_SNAPSHOT, sd_map_files

pytestmark = pytest.mark.real_data

#: Sindelfingen train scenes that lie (partly) outside the 2026-05-03 snapshot's download box.
#: The weekly snapshots cover them (3 km around every scene).
OUTSIDE = ("a18b7d47", "dbdaea96", "b3d0df28")


@pytest.fixture(scope="module")
def ds(real_root, tmp_path_factory):
    if PAPER_SNAPSHOT not in sd_map_files(real_root / "openstreetmap_sd_maps").get("Sindelfingen", {}):
        pytest.skip("the copy has no SD maps of Sindelfingen")
    mp = pytest.MonkeyPatch()
    if not os.environ.get("KITSCENES_CACHE"):
        mp.setenv("KITSCENES_CACHE", str(tmp_path_factory.mktemp("kitscenes_cache")))
    found = kitscenes.open(real_root, splits=["train", "val", "test", "test-e2e"]).filter(city="Sindelfingen")
    if not all(any(s.startswith(p) for s in found.scene_ids) for p in OUTSIDE):
        mp.undo()
        pytest.skip("the copy does not hold the Sindelfingen scenes")
    yield found
    mp.undo()


def test_val_scene_routes_and_commands(ds):
    scene = next(ds[s] for s in ds.scene_ids if ds.split_of(s) == "val")
    route = scene.sd_route
    assert route is not None and route.engine == "matched"
    xy = scene.track.xyz[:, :2]
    assert max(route.distance_to(p) for p in xy[::10]) < 15.0
    direct = scene.sd_map.route(scene.track.pose(0), scene.track.pose(-1))
    assert direct is not None
    commands = driving_commands(scene, KITSCENES)
    assert len(commands) == scene.num_frames and set(np.unique(commands)) <= {0, 1, 2, 3}
    road = scene.sd_map.attributes_at(scene.frame(0).pose)
    assert road is not None and road.speed_limit_mps is not None


def test_test_split_is_placed_with_the_city_origin(ds):
    scene = next(ds[s] for s in ds.scene_ids if ds.split_of(s) == "test")
    assert scene.sd_route is not None
    assert driving_commands(scene, NAVSIM, source="sd_route").shape == (scene.num_frames,)


def test_scenes_outside_the_snapshot_have_empty_sd_maps(ds):
    smap = ds.sd_map("Sindelfingen", snapshot=PAPER_SNAPSHOT)
    for prefix in OUTSIDE:
        scene = ds[next(s for s in ds.scene_ids if s.startswith(prefix))]
        xy = scene.track.xyz[:, :2]
        assert not smap.covers(xy, 50.0).all()
    scene = ds[next(s for s in ds.scene_ids if s.startswith("a18b7d47"))]
    assert smap.elements(Region().around(scene.frame(0).pose)).empty


def test_weekly_snapshot_covers_the_scenes_outside_the_old_one(ds):
    for prefix in OUTSIDE:
        scene = ds[next(s for s in ds.scene_ids if s.startswith(prefix))]
        smap = scene.sd_map
        assert smap.path.name == "Sindelfingen_2026-W08.osm.pbf" == scene.info.sd_snapshot
        assert smap.snapshot == "2026-02-16T00:00:00Z" and not smap.encoding_repaired
        assert smap.covers(scene.track.xyz[:, :2], 1000.0).all()
        assert not any(smap.elements(Region(30, 30, 15, 15).around(scene.frame(i).pose)).empty
                       for i in range(0, scene.num_frames, 10))
        assert scene.sd_route is not None
