"""The shipped scene table (split 2.0): split lists, city, lidar generation, recording metadata, filters."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest

import kitscenes
from kitscenes.splits import (
    FILTER_COLUMNS, SPLITS, _SCENE_COLUMNS, find_scenes, scene_info, scene_table, withheld_columns,
)
from release_fixture import write_timestamps

REPO = Path(__file__).resolve().parents[1]
E2E = ("test-e2e", "map_change_test-e2e")
#: The columns withheld in the test-e2e splits: recording order and road types.
E2E_WITHHELD = {"recording_order", "drive", "order_in_drive", "road_motorway_share", "road_arterial_share",
                "road_local_share", "road_service_share", "road_unmatched_share", "road_dominant",
                "speed_limit_kph_median"}
ROAD_SHARES = ("road_motorway_share", "road_arterial_share", "road_local_share", "road_service_share",
               "road_unmatched_share")


def test_table_covers_the_release_splits() -> None:
    table = scene_table()
    assert len(table) == 2079  # the release's scenes
    counts = Counter(info.split for info in table.values())
    assert set(counts) == set(SPLITS)  # only released splits
    assert counts["train"] == 1209 and counts["test-e2e"] == 107 and counts["map_change_overlap_train_val"] == 2
    assert counts["map_change_test"] == 17 and counts["map_change_test-e2e"] == 5
    assert {info.city for info in table.values()} == {"Karlsruhe", "Frankfurt", "Sindelfingen"}
    assert {info.city for info in table.values() if info.split in E2E} == {"Karlsruhe"}  # the city is kept
    assert {info.lidar for info in table.values()} == {"velodyne", "hesai"}
    info = scene_info("01883cf7-b15d-3336-a2dd-443a96e31d2c")
    assert (info.split, info.city, info.lidar, info.num_frames) == ("train", "Karlsruhe", "hesai", 190)
    assert scene_info("not-a-scene") is None


def test_columns_are_typed() -> None:
    text = (REPO / "kitscenes/_release_tables/v2.0.0/scenes.csv").read_text()
    assert tuple(next(csv.reader(text.splitlines()))) == _SCENE_COLUMNS
    table = list(scene_table().values())
    for info in table:
        assert isinstance(info.num_frames, int) and info.num_frames > 0
        assert isinstance(info.sun_below_horizon, bool) and isinstance(info.dark, bool)
        assert all(isinstance(getattr(info, c), float) for c in ("duration_s", "distance_m", "mean_speed_mps",
                                                                 "max_speed_mps"))
        assert info.lidar == ("velodyne" if info.rig == "velodyne_2025" else "hesai")
        assert info.sd_snapshot == f"{info.city}_{info.recording_week}.osm.pbf"
        assert not info.dark or info.sun_below_horizon
        if info.road_dominant is not None:
            shares = [getattr(info, c) for c in ROAD_SHARES]
            assert abs(sum(shares) - 1.0) <= 0.03
            assert getattr(info, f"road_{info.road_dominant}_share") == max(shares)  # ties after rounding
    limits = Counter(i.speed_limit_kph_median for i in table)
    assert math.inf in limits and None in limits and isinstance(next(iter(k for k in limits if k)), float)
    assert {i.weather for i in table} <= {"clear", "partly_cloudy", "overcast", "rain", "fog", "snow", "unknown"}
    assert Counter(i.recording_week.split("-")[0] for i in table) == Counter(
        {"2025": sum(i.lidar == "velodyne" for i in table), "2026": sum(i.lidar == "hesai" for i in table)})


def test_withheld_columns_are_empty() -> None:
    """The setting withholds the recording order and road types of test-e2e; nothing else is withheld."""
    setting = json.loads((REPO / "kitscenes/_release_tables/v2.0.0/withheld_columns.json").read_text())["by_split"]
    assert set(setting) == set(E2E)
    for split in SPLITS:
        expected = E2E_WITHHELD if split in E2E else set()
        assert set(withheld_columns(split)) == expected
    assert withheld_columns("test_e2e") == withheld_columns("test-e2e")  # aliases
    table = list(scene_table().values())
    for info in table:
        for column in withheld_columns(info.split):
            assert getattr(info, column) is None, (info.scene_id, column)
    e2e = [i for i in table if i.split in E2E]
    kept = set(_SCENE_COLUMNS) - E2E_WITHHELD - {"speed_limit_kph_median"}
    for column in kept:  # every other column is filled for test-e2e
        assert all(getattr(i, column) is not None for i in e2e), column


def test_recording_order_has_no_gaps() -> None:
    """recording_order and order_in_drive are numbered over the public scenes only: no gap shows where a
    test-e2e scene sat, and the test-e2e rows have no order values (taken as given, never renumbered)."""
    table = list(scene_table().values())
    public = [i for i in table if i.split not in E2E]
    assert sorted(i.recording_order for i in public) == list(range(1, len(public) + 1))
    by_drive = defaultdict(list)
    for info in public:
        by_drive[info.drive].append(info.order_in_drive)
    for numbers in by_drive.values():
        assert sorted(numbers) == list(range(1, len(numbers) + 1))
    assert sorted(by_drive) == list(range(1, len(by_drive) + 1))
    assert all(i.recording_order is None and i.drive is None and i.order_in_drive is None
               for i in table if i.split in E2E)
    # the order follows the drives
    ordered = sorted(public, key=lambda i: i.recording_order)
    assert all(a.drive <= b.drive for a, b in zip(ordered, ordered[1:]))


def test_find_scenes() -> None:
    table = list(scene_table().values())
    rain = find_scenes(weather="Rain")
    assert rain and all(i.weather == "rain" for i in rain) and len(rain) == sum(i.weather == "rain" for i in table)
    assert find_scenes(time_of_day=["evening", "night"]) == [i for i in table if i.time_of_day in ("evening", "night")]
    slow = find_scenes(max_speed_mps=(None, 15))
    assert slow == [i for i in table if i.max_speed_mps <= 15]
    motorway = find_scenes(road_motorway_share=(0.5, None), city="Karlsruhe")
    assert motorway and all(i.road_motorway_share >= 0.5 and i.city == "Karlsruhe" for i in motorway)
    assert not any(i.split in E2E for i in find_scenes(road_dominant=["local", "arterial", "motorway", "service"]))
    assert find_scenes(dark=True) == [i for i in table if i.dark]
    assert find_scenes(drive=[1, 2]) == [i for i in table if i.drive in (1, 2)]
    assert find_scenes(recording_order=1)[0].recording_order == 1
    assert find_scenes(speed_limit_kph_median=(130, None)) == [i for i in table if i.speed_limit_kph_median == math.inf]
    assert find_scenes(city=None, weather=None) == table
    assert find_scenes(sd_snapshot="karlsruhe_2026-w06.osm.pbf")[0].sd_snapshot == "Karlsruhe_2026-W06.osm.pbf"
    assert set(FILTER_COLUMNS["bool"]) == {"dark", "sun_below_horizon"}
    with pytest.raises(TypeError, match="Unknown scene table column"):
        find_scenes(colour="red")
    with pytest.raises(ValueError, match="Unknown weather 'hail'"):
        find_scenes(weather="hail")
    with pytest.raises(ValueError, match="Unknown recording_week"):
        find_scenes(recording_week="2024-W01")
    with pytest.raises(ValueError, match="range"):
        find_scenes(max_speed_mps=(1, 2, 3))
    with pytest.raises(TypeError):
        find_scenes(dark="yes")
    with pytest.raises(TypeError):
        find_scenes(max_speed_mps="fast")


def test_filter_and_mismatches(tmp_path: Path) -> None:
    table = scene_table()
    frankfurt_val = next(i for i in table.values() if i.split == "val" and i.city == "Frankfurt")
    karlsruhe_hesai = next(i for i in table.values() if i.split == "train" and i.city == "Karlsruhe" and i.lidar == "hesai")
    moved = next(i for i in table.values() if i.split == "train" and i.city == "Sindelfingen")
    for split, info in (("val", frankfurt_val), ("train", karlsruhe_hesai), ("test-e2e", moved)):
        scene = tmp_path / "data" / split / info.scene_id
        scene.mkdir(parents=True)
        write_timestamps(scene, 2)
    (tmp_path / "data" / "train" / "unlisted").mkdir()
    write_timestamps(tmp_path / "data" / "train" / "unlisted", 2)

    ds = kitscenes.open(tmp_path)
    assert ds.filter(city="Frankfurt").scene_ids == [frankfurt_val.scene_id]
    assert ds.filter(lidar="hesai", splits="train").scene_ids == [karlsruhe_hesai.scene_id]
    assert ds.filter(rig=karlsruhe_hesai.rig, recording_week=karlsruhe_hesai.recording_week,
                     splits="train").scene_ids == [karlsruhe_hesai.scene_id]
    assert ds.filter(num_frames=(frankfurt_val.num_frames, frankfurt_val.num_frames),
                     city=["Frankfurt", "Karlsruhe"]).scene_ids == [frankfurt_val.scene_id]
    assert ds.filter(drive=frankfurt_val.drive, scene_ids=[frankfurt_val.scene_id]).scene_ids == [frankfurt_val.scene_id]
    assert "unlisted" not in ds.filter(dark=False).scene_ids + ds.filter(dark=True).scene_ids
    with pytest.raises(TypeError, match="Unknown scene table column"):
        ds.filter(colour="red")
    assert ds[frankfurt_val.scene_id].info == frankfurt_val
