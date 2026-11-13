"""The SD-map snapshots of a release root: weekly files, the paper's 2026-05-03 files; PBF and encoding."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

import kitscenes
from kitscenes.errors import Availability, MapUnavailable
from kitscenes.map.origin import CITY_ORIGINS
from kitscenes.sdmap import SDMap, _parse
from kitscenes.sdmap._files import PAPER_SNAPSHOT, pick_snapshot, sd_map_files, snapshot_name
from kitscenes.splits import scene_table

from release_fixture import make_scene, write_poses
from sdmap_fixture import MOJIBAKE_NAME, ORIGIN, OSM_BASE, write_osm

NAME = MOJIBAKE_NAME.encode("cp1252").decode("utf-8")  # "Kaiserstraße"


def _scene_of(week: str, split: str = "train"):
    return next(i for i in scene_table().values() if i.split == split and i.sd_snapshot == f"Karlsruhe_{week}.osm.pbf")


def _write_scene(root: Path, info) -> None:
    scene = make_scene(root / "data" / info.split / info.scene_id, n=3, cameras=(), lidars=(), radars=(), poses=False)
    write_poses(scene, 3, origin=(0.0, 0.0, 0.0))
    if info.split != "test-e2e":
        (scene / "maps").mkdir()
        o = CITY_ORIGINS["Karlsruhe"]
        (scene / "maps" / "origin.json").write_text(json.dumps({"latitude": o.lat, "longitude": o.lon}))


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KITSCENES_CACHE", str(tmp_path / "cache"))


def _weekly_root(root: Path) -> Path:
    """Weekly snapshots (XML here) of 2026-W06 and 2026-W12, and the paper's file."""
    sd = root / "openstreetmap_sd_maps"
    write_osm(sd / "Karlsruhe_2026-W06.osm", correct_utf8=True, osm_base=None, extra_nodes={60: (0, 0, {"w": "06"})})
    write_osm(sd / "Karlsruhe_2026-W12.osm", correct_utf8=True, osm_base=None, extra_nodes={60: (0, 0, {"w": "12"})})
    write_osm(sd / PAPER_SNAPSHOT / "Karlsruhe.osm")
    (sd / "ATTRIBUTION.md").write_text("(c) OpenStreetMap contributors, ODbL 1.0\n")
    for week in ("2026-W06", "2026-W12"):
        _write_scene(root, _scene_of(week))
    return root


def test_files_and_snapshot_names(tmp_path):
    sd = tmp_path / "openstreetmap_sd_maps"
    for name in ("Karlsruhe_2026-W12.osm.pbf", "Karlsruhe_2025-W33.osm.pbf", "Frankfurt_2025-W34.osm.pbf",
                 "Karlsruhe.osm", "paper_2026-05-03/Karlsruhe.osm", "paper_2026-05-03/Frankfurt.osm",
                 "Sindelfingen.osm", "ATTRIBUTION.md", ".hidden/Karlsruhe.osm", "notes/readme.txt"):
        (sd / name).parent.mkdir(parents=True, exist_ok=True)
        (sd / name).write_text("x")
    files = sd_map_files(sd)
    assert list(files["Karlsruhe"]) == ["2025-W33", "2026-W12", PAPER_SNAPSHOT]
    assert files["Karlsruhe"][PAPER_SNAPSHOT] == sd / PAPER_SNAPSHOT / "Karlsruhe.osm"
    assert set(files) == {"Frankfurt", "Karlsruhe"}  # a <City>.osm at the top level is not a snapshot
    assert sd_map_files(tmp_path / "nothing") == {}

    assert snapshot_name("2026-w06") == "2026-W06"
    assert snapshot_name("Karlsruhe_2026-W06.osm.pbf", "karlsruhe") == "2026-W06"
    assert snapshot_name("Karlsruhe_2026-W06") == "2026-W06"
    assert snapshot_name("paper") == PAPER_SNAPSHOT and snapshot_name("latest") == "latest"
    with pytest.raises(ValueError, match="not of Frankfurt"):
        snapshot_name("Karlsruhe_2026-W06.osm.pbf", "Frankfurt")
    ka = files["Karlsruhe"]
    assert pick_snapshot(ka, None, "Karlsruhe")[0] == "2026-W12"  # the latest week
    assert pick_snapshot(ka, "latest", "Karlsruhe")[0] == "2026-W12"
    assert pick_snapshot(ka, "paper", "Karlsruhe")[0] == PAPER_SNAPSHOT
    assert pick_snapshot(ka, "2026-W06", "Karlsruhe") is None
    assert pick_snapshot({PAPER_SNAPSHOT: sd / PAPER_SNAPSHOT / "Frankfurt.osm"}, None, "Frankfurt") is None  # no week


def test_scene_uses_its_recording_week(tmp_path, cache):
    root = _weekly_root(tmp_path)
    ds = kitscenes.open(root)
    w06, w12 = _scene_of("2026-W06"), _scene_of("2026-W12")
    sd = root / "openstreetmap_sd_maps"
    assert ds.sd_maps() == {"Karlsruhe": {"2026-W06": sd / "Karlsruhe_2026-W06.osm",
                                          "2026-W12": sd / "Karlsruhe_2026-W12.osm",
                                          PAPER_SNAPSHOT: sd / PAPER_SNAPSHOT / "Karlsruhe.osm"}}
    a, b = ds[w06.scene_id].sd_map, ds[w12.scene_id].sd_map
    assert a.path.name == "Karlsruhe_2026-W06.osm" and b.path.name == "Karlsruhe_2026-W12.osm"
    assert a is ds.sd_map("Karlsruhe", snapshot="2026-W06") is ds.sd_map("karlsruhe", snapshot=w06.sd_snapshot)
    assert b is ds.sd_map("Karlsruhe")  # default: the latest week
    assert ds.sd_map("Karlsruhe", snapshot="paper").path == sd / PAPER_SNAPSHOT / "Karlsruhe.osm"
    assert a.elements(kitscenes.map.Box((0.0, 0.0), 0.0, 5, 5, 5, 5)).node_tags[0]["w"] == "06"  # really that file
    assert ds[w06.scene_id].availability("sd_map") is Availability.PRESENT
    # the weekly files are correct UTF-8 and stay as they are; the paper's are repaired
    assert not a.encoding_repaired and a.road(next(r.id for r in a.roads() if r.way_id == 10)).name == NAME
    paper = ds.sd_map("Karlsruhe", snapshot=PAPER_SNAPSHOT)
    assert paper.encoding_repaired and paper.snapshot == OSM_BASE and a.snapshot is None
    with pytest.raises(MapUnavailable, match="2026-W06, 2026-W12, paper_2026-05-03") as err:
        ds.sd_map("Karlsruhe", snapshot="2025-W33")
    assert err.value.state is Availability.NOT_DOWNLOADED
    with pytest.raises(ValueError, match="not of Karlsruhe"):
        ds.sd_map("Karlsruhe", snapshot="Frankfurt_2025-W34.osm.pbf")


def test_choose_the_paper_snapshot(tmp_path, cache):
    root = _weekly_root(tmp_path)
    w06 = _scene_of("2026-W06")
    ds = kitscenes.open(root, sd_snapshot="paper_2026-05-03")
    smap = ds[w06.scene_id].sd_map
    assert smap.path == root / "openstreetmap_sd_maps" / PAPER_SNAPSHOT / "Karlsruhe.osm"
    assert smap is ds.sd_map("Karlsruhe", snapshot="paper")
    for again in (ds.filter(splits="train"), ds[:1], pickle.loads(pickle.dumps(ds))):  # kept by sub-datasets
        assert again.sd_snapshot == "paper_2026-05-03"
    scene = pickle.loads(pickle.dumps(ds[w06.scene_id]))
    assert scene.sd_snapshot == "paper_2026-05-03"
    assert kitscenes.open(root, sd_snapshot="latest")[w06.scene_id].sd_map.path.name == "Karlsruhe_2026-W12.osm"
    missing = kitscenes.open(root, sd_snapshot="2025-W33")[w06.scene_id]
    assert missing.availability("sd_map") is Availability.NOT_DOWNLOADED
    with pytest.raises(ValueError):
        kitscenes.open(root, sd_snapshot="")


def test_e2e_stays_withheld(tmp_path, cache):
    root = _weekly_root(tmp_path)
    e2e = next(i for i in scene_table().values() if i.split == "test-e2e")
    _write_scene(root, e2e)
    scene = kitscenes.open(root)[e2e.scene_id]
    assert scene.info.city == "Karlsruhe" and scene.info.sd_snapshot  # the table knows, the poses cannot be placed
    assert scene.availability("sd_map") is Availability.WITHHELD_BY_SPLIT
    with pytest.raises(MapUnavailable, match="first pose"):
        scene.sd_map


def test_mojibake_repair():
    assert _parse.repair_mojibake(MOJIBAKE_NAME) == NAME
    lost = "Ёжик".encode("utf-8").decode("cp1252", errors="replace")  # 0x81 (in Ё) is not cp1252: U+FFFD
    assert "\ufffd" in lost and _parse.repair_mojibake(lost).endswith("жик")
    correct = ["Kaiserstraße", "Böblingen", "Café", "Øresund", "€uro", "東京", "Москва", "ÄÖÜ äöü ß", "naïve – “quoted”",
               "Ã", "plain"]
    assert [_parse.repair_mojibake(s) for s in correct] == correct
    assert not _parse.looks_double_encoded(correct)
    broken = [s.encode("utf-8").decode("cp1252", errors="replace") for s in correct[:5]]
    assert _parse.looks_double_encoded(broken + correct[-1:])
    assert not _parse.looks_double_encoded(["ascii only"])


def test_repair_encoding_modes(tmp_path):
    good = write_osm(tmp_path / "good.osm", correct_utf8=True)
    bad = write_osm(tmp_path / "bad.osm")

    def name(path, mode):
        smap = SDMap(path, ORIGIN, cache_dir=tmp_path / "cache", repair_encoding=mode)
        return smap.road(next(r.id for r in smap.roads() if r.way_id == 10)).name

    assert name(good, "auto") == NAME and name(bad, "auto") == NAME
    assert name(bad, False) == MOJIBAKE_NAME and name(good, True) == NAME
    with pytest.raises(ValueError):
        SDMap(good, ORIGIN, repair_encoding="yes")


# ----------------------------------------------------------------------
# PBF
# ----------------------------------------------------------------------

def _write_pbf(xml: Path, out: Path, timestamp: str = "2026-02-02T00:00:00Z") -> Path:
    import osmium

    header = osmium.io.Header()
    header.set("osmosis_replication_timestamp", timestamp)
    header.set("timestamp", timestamp)
    writer = osmium.SimpleWriter(str(out), 4096 * 1024, header)
    try:
        for obj in osmium.FileProcessor(str(xml)):
            writer.add(obj)
    finally:
        writer.close()
    return out


def test_pbf_snapshot(tmp_path, cache):
    root = _weekly_root(tmp_path)
    sd = root / "openstreetmap_sd_maps"
    _write_pbf(sd / "Karlsruhe_2026-W06.osm", sd / "Karlsruhe_2026-W06.osm.pbf")
    (sd / "Karlsruhe_2026-W06.osm").unlink()
    xml, pbf = _parse.parse(sd / "Karlsruhe_2026-W12.osm"), _parse.parse(sd / "Karlsruhe_2026-W06.osm.pbf")
    assert pbf.osm_base == "2026-02-02T00:00:00Z"
    assert pbf.num_nodes == xml.num_nodes and pbf.num_ways == xml.num_ways and pbf.num_relations == xml.num_relations
    assert (pbf.way_nodes == xml.way_nodes).all() and pbf.tags("way", 0) == xml.tags("way", 0)
    w06 = _scene_of("2026-W06")
    ds = kitscenes.open(root)
    smap = ds[w06.scene_id].sd_map
    assert smap.path.name == "Karlsruhe_2026-W06.osm.pbf" and smap.snapshot == "2026-02-02T00:00:00Z"
    assert not smap.encoding_repaired and smap.road(next(r.id for r in smap.roads() if r.way_id == 10)).name == NAME
    assert ds.sd_map("Karlsruhe", snapshot=PAPER_SNAPSHOT).path == sd / PAPER_SNAPSHOT / "Karlsruhe.osm"
