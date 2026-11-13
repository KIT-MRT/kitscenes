"""The kitscenes CLI and the component downloader, against a local stand-in for the hub."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import pytest

import kitscenes
from kitscenes import Availability
from kitscenes.cli import build_parser, cmd_download, main
from kitscenes.download import archives
from aerial_features_fixture import AREA, SET, token_field, write_features
from aerial_tiles_fixture import write_tiles
from hub_fixture import FakeHub, fetcher, hub_download, make_hub, run_download, versioned_hub
from release_fixture import make_scene, pack_scene


def test_partial_download_then_more(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "train", "--components", "front-camera", "lidar-top"], calls) == 0
    scene = root / "data" / "train" / "scene_a"
    assert sorted(p.name for p in scene.glob("*.zip")) == ["camera_ring_front.zip", "lidar_top.zip", "radar.zip"]  # radar always
    assert (scene / "index.parquet").is_file() and (scene / "calibration" / "calib.json").is_file()
    assert not (root / "data" / "val").exists()
    f = kitscenes.open(root)[0].frame(1)
    assert f.availability("camera_ring_front") is Availability.PRESENT
    assert f.availability("camera_ring_rear") is Availability.NOT_DOWNLOADED
    assert len(f.lidar("lidar_top")) == 20

    calls.clear()
    assert run_download(hub, root, ["--splits", "train", "--components", "cameras"], calls) == 0
    assert calls == ["data/train/scene_a/camera_ring_rear.zip"]  # the rest is in place
    assert kitscenes.open(root)[0].frame(1).image("camera_ring_rear").shape == (48, 64, 3)


def test_extract_and_checksum_failure(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "radar", "--extract"], calls) == 0
    scene = root / "data" / "val" / "scene_b"
    assert not (scene / "radar.zip").exists() and (scene / "radar_front" / "0000000000.parquet").is_file()
    assert len(kitscenes.open(root)[0].frame(0).radar("radar_front")) == 5

    rows = list(csv.DictReader(open(hub / "data" / "sequence_archives.csv")))
    for row in rows:
        if row["component"] == "lidar_top":
            row["archive_sha256"] = "0" * 64
    with open(hub / "data" / "sequence_archives.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # The default check is the size (fast); --verify checks SHA-256 and catches the wrong manifest hash.
    assert run_download(hub, tmp_path / "other", ["--scenes", "scene_c", "--components", "lidar-top"], calls) == 0
    assert run_download(hub, tmp_path / "other2", ["--scenes", "scene_c", "--components", "lidar-top", "--verify"], calls) == 1
    assert not (tmp_path / "other2" / "data" / "test-e2e" / "scene_c" / "lidar_top.zip").exists()


def test_selection_and_validation(tmp_path: Path, capsys) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "test_e2e", "--dry-run"], calls) == 0
    assert "1 scene(s)" in capsys.readouterr().out and calls == []
    files = archives.parse_manifest(csv.DictReader(open(hub / "data" / "sequence_archives.csv")))
    assert {f.scene_id for f in archives.select_files(files, max_gb=1e-9, aerial_features=False)} == set()
    with pytest.raises(ValueError, match="Unknown split"):
        archives.select_files(files, splits=["training"], aerial_features=False)
    with pytest.raises(ValueError, match="Unknown component"):
        archives.resolve_components(["lidar_roof"])
    bad = {"sequence_id": "scene_a", "split": "train", "component": "lidar_top",
           "archive_path": "data/train/scene_a/../../../escape.zip", "archive_sha256": "", "archive_size_bytes": "1"}
    with pytest.raises(ValueError, match="Unsafe"):
        archives.parse_manifest([bad])
    with pytest.raises(ValueError, match="Unsafe"):
        archives.parse_manifest([{**bad, "split": "..", "archive_path": "data/../scene_a/x.zip"}])


def test_extract_refuses_escaping_members(tmp_path: Path) -> None:
    import zipfile

    scene = tmp_path / "data" / "train" / "scene_a"
    scene.mkdir(parents=True)
    with zipfile.ZipFile(scene / "lidar_top.zip", "w") as zf:
        zf.writestr("scene_a/lidar_top/0.parquet", b"x")
        zf.writestr("other/evil.txt", b"x")
    with pytest.raises(ValueError, match="Refusing"):
        archives.extract_component(scene / "lidar_top.zip")
    assert (scene / "lidar_top.zip").exists() and not (tmp_path / "data" / "train" / "other").exists()


def test_info_and_extract_commands(tmp_path: Path, capsys) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    run_download(hub, root, ["--splits", "train", "--components", "front-camera"], calls)
    assert main(["info", str(root), "-v"]) == 0
    out = capsys.readouterr().out
    assert "train: 1 scene(s)" in out and "2/4 components" in out  # front camera + radar (always)
    assert main(["extract", str(root), "--components", "front-camera"]) == 0
    assert (root / "data/train/scene_a/camera_ring_front/0000000000.jpg").is_file()
    # A bad selection fails with exit code 1 (local manifest: no network).
    assert main(["download", str(root), "--splits", "train", "--components", "nope"]) == 1
    assert "Unknown component" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Extract, checksums, staging, versions, city maps
# ---------------------------------------------------------------------------


def test_extracted_components_are_not_downloaded_again(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "radar", "lidar-top", "--extract"], calls) == 0
    scene = root / "data" / "val" / "scene_b"
    assert not list(scene.glob("*.zip")) and (scene / ".extracted" / "radar.zip.sha256").is_file()
    calls.clear()
    assert run_download(hub, root, ["--splits", "val", "--components", "radar", "lidar-top"], calls) == 0
    assert calls == []  # extracted is complete
    shutil.rmtree(scene / "lidar_top")  # a deleted folder is not
    assert run_download(hub, root, ["--splits", "val", "--components", "radar", "lidar-top"], calls) == 0
    assert calls == ["data/val/scene_b/lidar_top.zip"]


def test_a_same_size_corruption_is_downloaded_again(tmp_path: Path) -> None:
    import os
    import time

    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "train", "--components", "lidar-top"], calls) == 0
    path = root / "data/train/scene_a/lidar_top.zip"
    data = bytearray(path.read_bytes())
    data[100] ^= 0xFF  # same size, other bytes
    time.sleep(0.01)
    path.write_bytes(bytes(data))
    os.utime(path, ns=(time.time_ns(), time.time_ns()))
    calls.clear()
    assert run_download(hub, root, ["--splits", "train", "--components", "lidar-top"], calls) == 0
    assert calls == []  # the fast default checks sizes: a same-size corruption needs --verify
    assert run_download(hub, root, ["--splits", "train", "--components", "lidar-top", "--verify"], calls) == 0
    assert calls == ["data/train/scene_a/lidar_top.zip"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256((hub / path.relative_to(root)).read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes()[:-10])  # a truncated file: the size catches it
    calls.clear()
    assert run_download(hub, root, ["--splits", "train", "--components", "lidar-top"], calls) == 0
    assert calls == ["data/train/scene_a/lidar_top.zip"]


def test_a_failed_extraction_leaves_the_scene_as_it_was(tmp_path: Path) -> None:
    import zipfile

    scene = tmp_path / "data" / "train" / "scene_a"
    (scene / "lidar_top").mkdir(parents=True)
    (scene / "lidar_top" / "old.parquet").write_bytes(b"old")
    with zipfile.ZipFile(scene / "lidar_top.zip", "w") as zf:
        zf.writestr("scene_a/lidar_top/0000000000.parquet", b"new")
        zf.writestr("scene_a/radar_front/0000000000.parquet", b"not this component")
    with pytest.raises(ValueError, match="outside the component"):
        archives.extract_component(scene / "lidar_top.zip")
    assert (scene / "lidar_top" / "old.parquet").read_bytes() == b"old"  # untouched
    assert (scene / "lidar_top.zip").exists() and not list(scene.parent.glob(".partial-*"))
    assert not (scene / ".extracted").exists()


def test_a_root_holds_one_version(tmp_path: Path, capsys) -> None:
    import json

    hub, root = versioned_hub(tmp_path), tmp_path / "local"
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top"]) == 0
    info = json.loads((root / "dataset_info.json").read_text())
    assert info["revision"] == "rev-a" and "lidar_top" in info["components_in_release"]
    assert info["manifest_sha256"] == hashlib.sha256((root / "data/sequence_archives.csv").read_bytes()).hexdigest()

    hub.main = "rev-b"  # the hub moved on; the root stays on its revision
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top"]) == 0
    assert json.loads((root / "dataset_info.json").read_text())["revision"] == "rev-a"
    assert hub_download(hub, root, ["--revision", "rev-b", "--splits", "train"]) == 2
    assert "--update" in capsys.readouterr().err

    assert hub_download(hub, root, ["--update", "--splits", "train", "--components", "lidar-top"]) == 0
    assert json.loads((root / "dataset_info.json").read_text())["revision"] == "rev-b"
    local = root / "data/train/scene_a/lidar_top.zip"
    assert local.read_bytes() == (hub.revisions["rev-b"] / "data/train/scene_a/lidar_top.zip").read_bytes()

    (root / "data/sequence_archives.csv").write_text("sequence_id,split,component,archive_path,archive_sha256,archive_size_bytes\n")
    before = hub.manifests
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top", "--dry-run"]) == 0
    assert hub.manifests == before + 1  # a manifest that does not match its record is fetched again


def test_city_maps_are_loose_files(tmp_path: Path) -> None:
    hub = make_hub(tmp_path)
    (hub / "lanelet2_hd_maps").mkdir()
    (hub / "lanelet2_hd_maps" / "Karlsruhe.osm").write_text("<osm/>")
    with open(hub / "data" / "sequence_archives.csv", "a", newline="") as fh:
        csv.writer(fh).writerow(["", "", "city_map", "lanelet2_hd_maps/Karlsruhe.osm",
                                 hashlib.sha256(b"<osm/>").hexdigest(), 6])
    root, calls = tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert "lanelet2_hd_maps/Karlsruhe.osm" in calls  # city maps always come along
    assert {c.rsplit("/", 1)[-1] for c in calls if c.endswith(".zip")} == {"radar.zip"}  # no cameras or lidars
    assert kitscenes.open(root).city_maps() == {"Karlsruhe": root / "lanelet2_hd_maps" / "Karlsruhe.osm"}
    for path in ("lanelet2_hd_maps/../escape.osm", "lanelet2_hd_maps/sub/Karlsruhe.osm", "maps/Karlsruhe.osm",
                 "data/train/x.osm"):
        with pytest.raises(ValueError, match="Unsafe"):
            archives.parse_manifest([{"sequence_id": "", "split": "", "component": "city_map", "archive_path": path,
                                      "archive_sha256": "", "archive_size_bytes": "1"}])


def test_hd_and_sd_city_maps_in_their_folders(tmp_path: Path) -> None:
    """The release layout: lanelet2_hd_maps/ and openstreetmap_sd_maps/ (always downloaded)."""
    hub = make_hub(tmp_path)
    files = {"lanelet2_hd_maps/Karlsruhe.osm": ("city_map", b"<osm/>"),
             "openstreetmap_sd_maps/paper_2026-05-03/Karlsruhe.osm": ("osm_map", b"<osm version='0.6'/>"),
             "openstreetmap_sd_maps/ATTRIBUTION.md": ("osm_metadata", b"(c) OpenStreetMap contributors, ODbL")}
    with open(hub / "data" / "sequence_archives.csv", "a", newline="") as fh:
        for path, (component, data) in files.items():
            (hub / path).parent.mkdir(parents=True, exist_ok=True)
            (hub / path).write_bytes(data)
            csv.writer(fh).writerow(["", "", component, path, hashlib.sha256(data).hexdigest(), len(data)])
    root, calls = tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert set(files) <= set(calls)
    ds = kitscenes.open(root)
    assert ds.city_maps() == {"Karlsruhe": root / "lanelet2_hd_maps" / "Karlsruhe.osm"}
    assert ds.sd_maps() == {"Karlsruhe": {"paper_2026-05-03": root / "openstreetmap_sd_maps" / "paper_2026-05-03" / "Karlsruhe.osm"}}
    for component, path in (("city_map", "openstreetmap_sd_maps/Karlsruhe.osm"), ("osm_map", "lanelet2_hd_maps/Karlsruhe.osm"),
                            ("osm_map", "openstreetmap_sd_maps/../x.osm"), ("osm_map", "openstreetmap_sd_maps/ATTRIBUTION.md"),
                            ("osm_map", "openstreetmap_sd_maps/Karlsruhe_2026-W06.pbf"),
                            ("osm_map", "openstreetmap_sd_maps/a/b/Karlsruhe.osm"),
                            ("osm_map", "openstreetmap_sd_maps/.hidden/Karlsruhe.osm"),
                            ("osm_map", "openstreetmap_sd_maps//Karlsruhe.osm"),
                            ("osm_map", "/openstreetmap_sd_maps/Karlsruhe.osm"),
                            ("osm_map", "openstreetmap_sd_maps/paper\\Karlsruhe.osm"),
                            ("osm_metadata", "openstreetmap_sd_maps/sub/../../x.md")):
        with pytest.raises(ValueError, match="Unsafe"):
            archives.parse_manifest([{"sequence_id": "", "split": "", "component": component, "archive_path": path,
                                      "archive_sha256": "", "archive_size_bytes": "1"}])


def test_weekly_sd_snapshots_in_the_manifest(tmp_path: Path) -> None:
    """The weekly snapshots <City>_<YYYY>-W<ww>.osm.pbf and the paper's files in paper_2026-05-03/ are
    always downloaded; there is no component name for them."""
    hub = make_hub(tmp_path)
    files = {"openstreetmap_sd_maps/Karlsruhe_2026-W06.osm.pbf": ("osm_map", b"pbf-1"),
             "openstreetmap_sd_maps/Karlsruhe_2026-W12.osm.pbf": ("osm_map", b"pbf-2"),
             "openstreetmap_sd_maps/paper_2026-05-03/Karlsruhe.osm": ("osm_map", b"<osm version='0.6'/>"),
             "openstreetmap_sd_maps/paper_2026-05-03/README.md": ("osm_metadata", b"the paper's snapshots"),
             "openstreetmap_sd_maps/ATTRIBUTION.md": ("osm_metadata", b"(c) OpenStreetMap contributors, ODbL")}
    with open(hub / "data" / "sequence_archives.csv", "a", newline="") as fh:
        for path, (component, data) in files.items():
            (hub / path).parent.mkdir(parents=True, exist_ok=True)
            (hub / path).write_bytes(data)
            csv.writer(fh).writerow(["", "", component, path, hashlib.sha256(data).hexdigest(), len(data)])
    root, calls = tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert all((root / p).read_bytes() == d for p, (_, d) in files.items())
    folder = root / "openstreetmap_sd_maps"
    with pytest.raises(ValueError, match="sd-maps-paper"):
        archives.resolve_components(["sd-maps-paper"])
    assert kitscenes.open(root).sd_maps() == {"Karlsruhe": {
        "2026-W06": folder / "Karlsruhe_2026-W06.osm.pbf", "2026-W12": folder / "Karlsruhe_2026-W12.osm.pbf",
        "paper_2026-05-03": folder / "paper_2026-05-03" / "Karlsruhe.osm"}}


def test_partial_copies_name_the_command(tmp_path: Path) -> None:
    """A sensor without data in this copy is NOT_DOWNLOADED: every scene has all 19 sensors."""
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    # scene_d: no lidar_left anywhere (not even in the release manifest): still not downloaded, not "not in release"
    pack_scene(make_scene(hub / "data" / "train" / "scene_d", n=3, cameras=("camera_ring_front",), lidars=("lidar_top",),
                          radars=(), calib_only=("lidar_left",)))
    rows = list(csv.DictReader(open(hub / "data" / "sequence_archives.csv")))
    for path in sorted((hub / "data" / "train" / "scene_d").rglob("*")):
        if path.is_file():
            rel = path.relative_to(hub).as_posix()
            rows.append({"sequence_id": "scene_d", "split": "train", "component": path.stem if path.suffix == ".zip" else "metadata",
                         "archive_path": rel, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "archive_size_bytes": path.stat().st_size})
    with open(hub / "data" / "sequence_archives.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert run_download(hub, root, ["--splits", "train", "--components", "front-camera"], calls) == 0
    ds = kitscenes.open(root)
    a, d = ds.scene("scene_a"), ds.scene("scene_d")
    assert a.availability("camera_ring_rear") is Availability.NOT_DOWNLOADED  # in the release
    assert d.availability("lidar_left") is Availability.NOT_DOWNLOADED
    assert a.availability("camera_base_front_center") is Availability.NOT_DOWNLOADED  # not even in calib.json
    with pytest.raises(kitscenes.SensorUnavailable,
                       match=f"kitscenes download {root} --scenes scene_d --components lidar_left"):
        d.frame(0).lidar("lidar_left")
    with pytest.raises(kitscenes.SensorUnavailable, match="--scenes scene_a --components camera_base_front_center"):
        a.frame(0).image("camera_base_front_center")
    with pytest.raises(kitscenes.SensorUnavailable,
                       match=f"kitscenes download {root} --scenes scene_a --components camera_ring_rear"):
        a.frame(0).image("camera_ring_rear")
    with pytest.raises(kitscenes.SensorUnavailable, match="--components lidar_top"):
        a.frame(0).lidar("lidar_top")
    assert len(a.frame(0).radar("radar_front")) == 5  # radar always comes along
    with pytest.raises(kitscenes.SensorUnavailable, match="not downloaded in 2 scene"):
        ds.frames(require=["lidar_top"])

    # Downloaded while the dataset is open: the same Scene sees it.
    assert run_download(hub, root, ["--splits", "train", "--components", "ring-cameras", "lidar-top"], calls) == 0
    assert a.availability("camera_ring_rear") is Availability.PRESENT
    assert len(ds.frames(require=["lidar_top"])) == 6


def test_a_1x_release_is_refused(tmp_path: Path, capsys) -> None:
    import json

    hub = versioned_hub(tmp_path)
    old = tmp_path / "old"  # dataset 1.x: one tar per scene, a manifest without a component column
    (old / "data").mkdir(parents=True)
    (old / "data" / "sequence_archives.csv").write_text(
        "sequence_id,split,archive_path,archive_sha256,archive_size_bytes\nscene_a,train,data/train/scene_a.tar,00,1\n")
    hub.revisions["v1.0.1"] = old
    root = tmp_path / "local"
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top"]) == 0
    before = (root / "data/sequence_archives.csv").read_bytes()
    assert hub_download(hub, root, ["--update", "--revision", "v1.0.1", "--dry-run"]) == 2
    assert "legacy kitscenes 0.1.0" in capsys.readouterr().err
    assert (root / "data/sequence_archives.csv").read_bytes() == before  # the root still holds its release
    assert json.loads((root / "dataset_info.json").read_text())["revision"] == "rev-a"

    new_root = tmp_path / "new"
    assert hub_download(hub, new_root, ["--revision", "v1.0.1"]) == 2
    assert not (new_root / "data/sequence_archives.csv").exists() and not (new_root / "dataset_info.json").exists()


# ---------------------------------------------------------------------------
# Updates, extraction checks, markers
# ---------------------------------------------------------------------------


def _patch_member(zip_path: Path, name: str, data: bytes) -> None:
    """Replace a stored member's bytes in place (same size, CRCs fixed): the archive keeps its size."""
    import struct
    import zipfile
    import zlib

    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(name)
    assert len(data) == info.file_size
    raw = bytearray(zip_path.read_bytes())
    name_len, extra_len = struct.unpack_from("<HH", raw, info.header_offset + 26)
    start = info.header_offset + 30 + name_len + extra_len
    raw[start:start + len(data)] = data
    crc = zlib.crc32(data)
    struct.pack_into("<I", raw, info.header_offset + 14, crc)
    pos = raw.find(b"PK\x01\x02")
    while pos >= 0:  # the central directory entry of the member
        n = struct.unpack_from("<H", raw, pos + 28)[0]
        if raw[pos + 46:pos + 46 + n] == name.encode():
            struct.pack_into("<I", raw, pos + 16, crc)
        pos = raw.find(b"PK\x01\x02", pos + 4)
    zip_path.write_bytes(bytes(raw))


CHANGED = "data/train/scene_a/lidar_top.zip"
MEMBER = "scene_a/lidar_top/0000000001.parquet"


def _same_size_hub(tmp_path: Path) -> FakeHub:
    """rev-b changes one member of train/scene_a/lidar_top.zip; the archive keeps its size (as aligned zips do)."""
    import zipfile

    a = make_hub(tmp_path / "a")
    b = tmp_path / "b" / "hub"
    shutil.copytree(a, b)
    with zipfile.ZipFile(b / CHANGED) as zf:
        old = zf.read(MEMBER)
    _patch_member(b / CHANGED, MEMBER, bytes(reversed(old)))
    assert (b / CHANGED).stat().st_size == (a / CHANGED).stat().st_size
    rows = list(csv.DictReader(open(b / "data/sequence_archives.csv")))
    for row in rows:
        if row["archive_path"] == CHANGED:
            row["archive_sha256"] = hashlib.sha256((b / CHANGED).read_bytes()).hexdigest()
    with open(b / "data/sequence_archives.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return FakeHub({"rev-a": a, "rev-b": b}, main="rev-a")


def _member(hub: FakeHub, revision: str) -> bytes:
    import zipfile

    with zipfile.ZipFile(hub.revisions[revision] / CHANGED) as zf:
        return zf.read(MEMBER)


def test_extraction_checks_the_zip_against_the_manifest(tmp_path: Path, capsys) -> None:
    """The marker holds the zip's real SHA-256; a zip that does not match the manifest is not extracted."""
    hub, root = _same_size_hub(tmp_path), tmp_path / "local"
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top"]) == 0
    shutil.copyfile(hub.revisions["rev-b"] / CHANGED, root / CHANGED)  # same size, other bytes than the manifest says
    marker = root / "data/train/scene_a/.extracted/lidar_top.zip.sha256"
    assert main(["extract", str(root), "--splits", "train"]) == 1
    assert "does not match the manifest" in capsys.readouterr().err
    assert (root / CHANGED).is_file() and not marker.exists()  # nothing stamped, the zip kept
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top", "--extract"]) == 1
    assert not marker.exists()
    # --verify fetches the right zip, which then extracts; the marker holds its hash
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top", "--verify", "--extract"]) == 0
    assert marker.read_text().split()[0] == hashlib.sha256((hub.revisions["rev-a"] / CHANGED).read_bytes()).hexdigest()
    assert (root / "data/train/scene_a/lidar_top/0000000001.parquet").read_bytes() == _member(hub, "rev-a")
    assert hub_download(hub, root, ["--splits", "train", "--components", "lidar-top", "--verify"]) == 0


def test_a_broken_marker_means_not_extracted(tmp_path: Path) -> None:
    """An empty marker (a crash while writing it) counts as not extracted: the zip is fetched again."""
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none", "--extract"], calls) == 0
    marker = root / "data/val/scene_b/.extracted/radar.zip.sha256"
    marker.write_text("")
    calls.clear()
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert calls == ["data/val/scene_b/radar.zip"] and not marker.exists()


def test_update_fetches_a_file_that_changed_at_the_same_size(tmp_path: Path, capsys) -> None:
    """--update re-fetches what the new manifest changed; --update --dry-run changes nothing."""
    import json

    from kitscenes._storage import ZipStorage

    hub, root = _same_size_hub(tmp_path), tmp_path / "local"
    sel = ["--splits", "train", "--components", "lidar-top"]
    assert hub_download(hub, root, sel) == 0
    manifest, info = (root / "data/sequence_archives.csv").read_bytes(), (root / "dataset_info.json").read_bytes()
    hub.main = "rev-b"
    capsys.readouterr()
    assert hub_download(hub, root, ["--update", "--dry-run", *sel]) == 0
    out = capsys.readouterr().out
    assert "1 file(s) changed" in out and f"  {CHANGED}" in out
    assert (root / "data/sequence_archives.csv").read_bytes() == manifest  # a dry run moves nothing
    assert (root / "dataset_info.json").read_bytes() == info
    assert (root / CHANGED).read_bytes() == (hub.revisions["rev-a"] / CHANGED).read_bytes()

    assert hub_download(hub, root, ["--update", *sel]) == 0
    assert json.loads((root / "dataset_info.json").read_text())["revision"] == "rev-b"
    assert (root / CHANGED).read_bytes() == (hub.revisions["rev-b"] / CHANGED).read_bytes()
    assert ZipStorage(root / "data/train/scene_a").read_bytes("lidar_top/0000000001.parquet") == _member(hub, "rev-b")


def test_update_re_extracts_a_changed_component(tmp_path: Path) -> None:
    hub, root = _same_size_hub(tmp_path), tmp_path / "local"
    sel = ["--splits", "train", "--components", "lidar-top"]
    assert hub_download(hub, root, [*sel, "--extract"]) == 0
    member = root / "data/train/scene_a/lidar_top/0000000001.parquet"
    marker = root / "data/train/scene_a/.extracted/lidar_top.zip.sha256"
    assert member.read_bytes() == _member(hub, "rev-a")
    other = tmp_path / "other"
    shutil.copytree(root, other)

    hub.main = "rev-b"
    assert hub_download(hub, root, ["--update", *sel]) == 0  # extracted before, so extracted again
    assert member.read_bytes() == _member(hub, "rev-b") and not (root / CHANGED).exists()
    assert marker.read_text().split()[0] == hashlib.sha256((hub.revisions["rev-b"] / CHANGED).read_bytes()).hexdigest()

    # Not selected: the stale extraction is removed (it would be read as data of the new revision).
    assert hub_download(hub, other, ["--update", "--splits", "val", "--components", "none"]) == 0
    scene = other / "data/train/scene_a"
    assert not (scene / "lidar_top").exists() and not (scene / ".extracted/lidar_top.zip.sha256").exists()
    assert (scene / ".extracted/radar.zip.sha256").is_file()  # unchanged components stay
    assert kitscenes.open(other).scene("scene_a").availability("lidar_top") is Availability.NOT_DOWNLOADED
    assert hub_download(hub, other, sel) == 0
    assert (scene / "lidar_top.zip").read_bytes() == (hub.revisions["rev-b"] / CHANGED).read_bytes()


def test_an_unpinned_root_needs_update_every_time(tmp_path: Path, capsys) -> None:
    """A root without a pinned revision asks for --update on every run; a refused format leaves it intact."""
    import json

    hub, root = versioned_hub(tmp_path), tmp_path / "local"
    (root / "data").mkdir(parents=True)
    shutil.copyfile(hub.revisions["rev-a"] / "data/sequence_archives.csv", root / "data/sequence_archives.csv")
    sel = ["--splits", "train", "--components", "lidar-top"]
    assert hub_download(hub, root, sel) == 0  # a local copy keeps its manifest, unpinned
    assert json.loads((root / "dataset_info.json").read_text())["revision"] is None
    for _ in range(2):
        assert hub_download(hub, root, ["--revision", "rev-b", *sel, "--dry-run"]) == 2
        assert "unpinned" in capsys.readouterr().err

    old = tmp_path / "old"  # a dataset 1.x manifest
    (old / "data").mkdir(parents=True)
    (old / "data" / "sequence_archives.csv").write_text(
        "sequence_id,split,archive_path,archive_sha256,archive_size_bytes\nscene_a,train,data/train/scene_a.tar,00,1\n")
    hub.revisions["v1.0.1"] = old
    before = (root / "data/sequence_archives.csv").read_bytes()
    assert hub_download(hub, root, ["--update", "--revision", "v1.0.1"]) == 2
    assert (root / "data/sequence_archives.csv").read_bytes() == before
    assert hub_download(hub, root, sel) == 0  # later runs still work

    assert hub_download(hub, root, ["--update", *sel]) == 0
    assert json.loads((root / "dataset_info.json").read_text())["revision"] == "rev-a"


def test_a_dry_run_on_a_new_root_writes_nothing(tmp_path: Path) -> None:
    hub, root = versioned_hub(tmp_path), tmp_path / "local"
    assert hub_download(hub, root, ["--splits", "train", "--dry-run"]) == 0
    assert not root.exists()


def test_verify_hashes_each_file_once(tmp_path: Path, monkeypatch) -> None:
    """--verify hashes each file once."""
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    sel = ["--splits", "train", "--components", "lidar-top"]
    assert run_download(hub, root, sel, calls) == 0
    n = len([p for p in (root / "data/train").rglob("*") if p.is_file()])
    hashed = []
    original = archives.sha256_file
    monkeypatch.setattr(archives, "sha256_file", lambda path: hashed.append(Path(path).name) or original(path))
    assert run_download(hub, root, [*sel, "--verify"], calls) == 0
    assert len([h for h in hashed if h != "sequence_archives.csv"]) == n


def test_one_writer_per_root(tmp_path: Path, capsys) -> None:
    """A second writer on the same root is refused; a dry run only reads and is allowed."""
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    with archives.root_lock(root):
        assert main(["download", str(root), "--splits", "val", "--components", "none"]) == 1
        assert "another kitscenes" in capsys.readouterr().err
        assert main(["extract", str(root)]) == 1
        assert run_download(hub, root, ["--splits", "val", "--dry-run"], calls) == 0  # a dry run only reads
    assert main(["extract", str(root)]) == 0
    assert main(["extract", str(tmp_path / "typo")]) == 1 and not (tmp_path / "typo").exists()


def test_extract_takes_split_aliases_and_refuses_unknown_names(tmp_path: Path, capsys) -> None:
    """'extract --splits' takes the split aliases; an unknown split or scene is an error."""
    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--components", "none"], calls) == 0
    assert main(["extract", str(root), "--splits", "test_e2e"]) == 0
    assert not (root / "data/test-e2e/scene_c/radar.zip").exists() and (root / "data/train/scene_a/radar.zip").exists()
    assert "1 archive(s) extracted" in capsys.readouterr().out
    assert main(["extract", str(root), "--splits", "training"]) == 1
    assert "No split(s) ['training']" in capsys.readouterr().err
    assert main(["extract", str(root), "--scenes", "scene_x"]) == 1
    assert main(["extract", str(root), "--scenes", "scene_a"]) == 0
    assert not (root / "data/train/scene_a/radar.zip").exists()


def test_select_by_city_and_lidar(tmp_path: Path, capsys) -> None:
    """Download selection by city and lidar generation, from the shipped scene table."""
    from kitscenes.splits import scene_table

    table = scene_table()
    pick = {}
    for info in table.values():
        pick.setdefault(("e2e" in info.split, info.city, info.lidar), info)
    ka_hesai, fra_velo = pick[(False, "Karlsruhe", "hesai")], pick[(False, "Frankfurt", "velodyne")]
    e2e = pick[(True, "Karlsruhe", "hesai")]  # test-e2e scenes have a city too
    files = [archives.ArchiveFile(i.scene_id, i.split, "lidar_top", f"data/{i.split}/{i.scene_id}/lidar_top.zip", "0", 1)
             for i in (ka_hesai, fra_velo, e2e)]
    ids = lambda **kw: {f.scene_id for f in archives.select_files(files, aerial_features=False, **kw)}  # noqa: E731
    assert ids(cities=["karlsruhe"]) == {ka_hesai.scene_id, e2e.scene_id}  # names are case-insensitive
    assert ids(cities=["Karlsruhe", "Frankfurt"]) == {ka_hesai.scene_id, fra_velo.scene_id, e2e.scene_id}
    assert ids(lidar="Hesai") == {ka_hesai.scene_id, e2e.scene_id}
    assert ids(cities=["Frankfurt"], lidar="velodyne") == {fra_velo.scene_id}
    with pytest.raises(ValueError, match="Unknown city"):
        ids(cities=["Paris"])
    with pytest.raises(ValueError, match="Unknown lidar"):
        ids(lidar="ouster")
    with pytest.raises(ValueError, match="Frankfurt.*hesai"):
        ids(cities=["Frankfurt"], lidar="hesai")

    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert main(["download", str(root), "--cities", "paris", "--dry-run"]) == 1
    assert "Unknown city" in capsys.readouterr().err
    assert main(["extract", str(root), "--lidar", "ouster"]) == 1


def test_select_by_scene_table_columns(tmp_path: Path, capsys) -> None:
    """--where: any column of the scene table; withheld values (test-e2e) never match."""
    from kitscenes.cli import build_parser, where_criterion
    from kitscenes.splits import scene_table

    infos = list(scene_table().values())
    rain = next(i for i in infos if i.weather == "rain" and "e2e" not in i.split)
    fast = next(i for i in infos if i.max_speed_mps > 20 and i.weather != "rain" and "e2e" not in i.split)
    e2e = next(i for i in infos if i.split == "test-e2e" and i.weather != "rain")
    files = [archives.ArchiveFile(i.scene_id, i.split, "lidar_top", f"data/{i.split}/{i.scene_id}/lidar_top.zip", "0", 1)
             for i in (rain, fast, e2e)]
    ids = lambda **kw: {f.scene_id for f in archives.select_files(files, aerial_features=False, **kw)}  # noqa: E731
    assert ids(where={"weather": ["Rain"]}) == {rain.scene_id}
    assert ids(where={"max_speed_mps": (20, None)}) == {fast.scene_id}
    assert ids(where={"drive": (1, None)}) == {rain.scene_id, fast.scene_id}  # withheld for test-e2e
    assert ids(where={"weather": ["rain"]}, cities=[rain.city]) == {rain.scene_id}
    with pytest.raises(ValueError, match="Unknown weather"):
        ids(where={"weather": "hail"})
    with pytest.raises(ValueError, match="No selected scene.*weather"):
        ids(where={"weather": "rain"}, scene_ids=[fast.scene_id])

    assert where_criterion("weather=rain,overcast") == ("weather", ["rain", "overcast"])
    assert where_criterion("dark=True") == ("dark", True)
    assert where_criterion("max-speed-mps=..15") == ("max_speed_mps", (None, 15.0))
    assert where_criterion("road_motorway_share=0.5..") == ("road_motorway_share", (0.5, None))
    assert where_criterion("drive=3,4") == ("drive", [3.0, 4.0])
    assert where_criterion("speed_limit_kph_median=inf") == ("speed_limit_kph_median", float("inf"))
    args = build_parser().parse_args(["download", "R", "--where", "weather=rain", "--where", "dark=false"])
    assert args.where == [("weather", ["rain"]), ("dark", False)]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["download", "R", "--where", "colour=red"])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        build_parser().parse_args(["download", "R", "--where", "dark=maybe"])
    assert "true or false" in capsys.readouterr().err

    hub, root, calls = make_hub(tmp_path), tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert main(["download", str(root), "--where", "weather=hail", "--dry-run"]) == 1
    assert "Unknown weather" in capsys.readouterr().err
    assert main(["download", str(root), "--where", "dark=true", "--where", "dark=false", "--dry-run"]) == 1
    assert "given twice" in capsys.readouterr().err


def test_an_interrupt_starts_no_more_downloads(tmp_path: Path) -> None:
    hub, root = make_hub(tmp_path), tmp_path / "local"
    files = archives.select_files(archives.parse_manifest(csv.DictReader(open(hub / "data/sequence_archives.csv"))),
                                  aerial_features=True)
    calls = []

    def download_one(f, staging):
        calls.append(f.path)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        archives.fetch(root, files, download_one, jobs=1)
    assert len(calls) == 1 < len(files)


def test_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    import kitscenes.cli as cli

    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_extract", interrupted)
    assert main(["extract", str(tmp_path)]) == 130
    assert "interrupted" in capsys.readouterr().err
    assert main(["info", str(tmp_path / "nothing")]) == 1
    with pytest.raises(SystemExit) as err:
        main(["extract", str(tmp_path), "--jobs", "4"])  # a download flag
    assert err.value.code == 2


def _add_aerial(hub: Path) -> None:
    """The aerial rows of a release: one tile zip and one feature zip per area, loose metadata (scene_a in area_1,
    scene_b in area_2; scene_c, of test-e2e, in area_1 too)."""
    import json
    import zipfile

    rows = []
    for area, scene in (("area_1", "scene_a"), ("area_2", "scene_b")):
        src = write_tiles(hub / "_src" / area, fmt="jpeg", layout="zip", area=area, tiles=[(0, 0)],
                          scenes={scene: area})
        for path in src.rglob("*.zip"):
            target = hub / "aerial" / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            assert zipfile.is_zipfile(target)
    features = hub / "aerial" / "features" / SET
    areas = {}
    for area in ("area_1", "area_2"):
        write_features(hub, layout="zip", area=area, tiles=[(0, 0)])
        areas.update(json.loads((features / "manifest.json").read_text())["areas"])
    manifest = json.loads((features / "manifest.json").read_text())
    (features / "manifest.json").write_text(json.dumps({**manifest, "areas": areas}))
    meta = hub / "aerial" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "area_index.json").write_text(json.dumps({
        "bucket_size_m": 10000.0, "strategy": "epsg25832_bucket", "areas": {},
        "scene_to_area": {"scene_a": "area_1", "scene_b": "area_2", "scene_c": "area_1"}}))
    manifest = json.loads((hub / "_src" / "area_1" / "metadata" / "raw_tile_manifest_dop20_res100_tile32.json").read_text())
    manifest["areas"].update(json.loads((hub / "_src" / "area_2" / "metadata" /
                                         "raw_tile_manifest_dop20_res100_tile32.json").read_text())["areas"])
    (meta / "raw_tile_manifest_dop20_res100_tile32.json").write_text(json.dumps(manifest))
    (meta / "provider_metadata.json").write_text("{}")
    (hub / "aerial" / "ATTRIBUTION.md").write_text("licences\n")
    shutil.rmtree(hub / "_src")
    for path in sorted((hub / "aerial").rglob("*")):
        if path.is_file():
            rel = path.relative_to(hub).as_posix()
            if rel.startswith("aerial/features/"):
                component = "aerial_features" if path.suffix == ".zip" else "aerial_features_metadata"
            else:
                component = "aerial_tiles" if path.suffix == ".zip" else "aerial_metadata"
            rows.append(["", "", component, rel, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size])
    with open(hub / "data" / "sequence_archives.csv", "a", newline="") as fh:
        csv.writer(fh).writerows(rows)


_AERIAL_METADATA = ["aerial/ATTRIBUTION.md", "aerial/metadata/area_index.json",
                    "aerial/metadata/provider_metadata.json",
                    "aerial/metadata/raw_tile_manifest_dop20_res100_tile32.json"]
_FEATURES = "aerial/features/dinov3_test_tile8"


def _aerial_calls(calls: list) -> list:
    return sorted({c for c in calls if c.startswith("aerial/")})


_TILES = "aerial/raw_tiles/dop20_res100_tile32"
_FEATURE_METADATA = [f"{_FEATURES}/TERMS.md", f"{_FEATURES}/manifest.json"]


def test_aerial_by_area(tmp_path: Path, capsys) -> None:
    from kitscenes.aerial import AerialFeatures, AerialTiles

    hub = make_hub(tmp_path)
    _add_aerial(hub)
    root, calls = tmp_path / "local", []
    # a filtered download: the tiles of the selected scenes' areas, also with no cameras or lidars; no features
    assert run_download(hub, root, ["--splits", "val", "--components", "none"], calls) == 0
    assert _aerial_calls(calls) == sorted(_AERIAL_METADATA + [f"{_TILES}/area_2.zip"])
    assert "--aerial-features adds them (1 zip(s)" in capsys.readouterr().out
    assert AerialTiles(root).location("area_2", 0, 0).member == "area_2/tile_0_0.jpg"  # read from the zip, in place

    calls.clear()  # --aerial-features: the features of the same areas
    assert run_download(hub, root, ["--splits", "val", "--components", "none", "--aerial-features"], calls) == 0
    assert _aerial_calls(calls) == sorted(_FEATURE_METADATA + [f"{_FEATURES}/area_2.zip"])
    assert "--aerial-features adds" not in capsys.readouterr().out
    assert AerialFeatures(root).location("area_2", 0, 0).member == "area_2/tile_0_0.safetensors"

    calls.clear()  # a full download (--components filters no scene): every area, the features too
    assert run_download(hub, root, ["--components", "none"], calls) == 0
    assert _aerial_calls(calls) == [f"{_FEATURES}/area_1.zip", f"{_TILES}/area_1.zip"]

    # extract: the same rules; the area folder replaces the zip, the next download does not fetch it again
    assert main(["extract", str(root), "--splits", "val"]) == 0  # filtered: the tiles of area_2
    assert (root / _TILES / "area_2" / "tile_0_0.jpg").is_file() and not (root / _TILES / "area_2.zip").exists()
    assert (root / _TILES / "area_1.zip").is_file()  # scene_a is not in val
    assert (root / _FEATURES / "area_2.zip").is_file()
    assert main(["extract", str(root), "--splits", "val", "--aerial-features"]) == 0
    assert (root / _FEATURES / "area_2").is_dir() and not (root / _FEATURES / "area_2.zip").exists()
    assert main(["extract", str(root), "--no-aerial"]) == 0  # full, but without any aerial zip
    assert (root / _TILES / "area_1.zip").is_file() and (root / _FEATURES / "area_1.zip").is_file()
    assert main(["extract", str(root), "--no-aerial-features"]) == 0
    assert (root / _TILES / "area_1").is_dir() and (root / _FEATURES / "area_1.zip").is_file()
    assert main(["extract", str(root), "--components", "none"]) == 0  # full: the features too
    assert (root / _FEATURES / "area_1").is_dir() and not (root / _FEATURES / "area_1.zip").exists()
    assert AerialTiles(root).location("area_2", 0, 0).member is None
    calls.clear()
    assert run_download(hub, root, ["--components", "none", "--verify"], calls) == 0
    assert calls == []

    # an update that changes the zip removes the extraction (the API would read it as the new version)
    files = archives.parse_manifest(csv.DictReader(open(hub / "data/sequence_archives.csv")))
    area_2 = next(f for f in files if f.path == f"{_TILES}/area_2.zip")
    assert archives.remove_stale(root, [area_2]) == [area_2]
    assert not (root / _TILES / "area_2").exists() and not archives.is_complete(root, area_2)

    # a dry run that needs the area index fetches it elsewhere, not into the root; it names the features left out
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "data").mkdir()
    shutil.copyfile(hub / "data" / "sequence_archives.csv", fresh / "data" / "sequence_archives.csv")
    args = build_parser().parse_args(["download", str(fresh), "--splits", "train", "--dry-run"])
    capsys.readouterr()
    assert cmd_download(args, download_one=fetcher(hub, [])) == 0
    out = capsys.readouterr().out
    assert f"{_TILES}/area_1.zip" in out and "area_2.zip" not in out and not (fresh / "aerial").exists()
    assert f"{_FEATURES}/area_1.zip" not in out and "--aerial-features adds them (1 zip(s)" in out
    selected = [f for f in files if f.split == "train"]
    assert f"1 scene(s), {len(selected) + 5} file(s)" in out  # + the aerial metadata and the tiles of area_1


_ALL = _AERIAL_METADATA + _FEATURE_METADATA + [f"{d}/{a}.zip" for d in (_TILES, _FEATURES) for a in ("area_1", "area_2")]


@pytest.mark.parametrize("flags, expected", [
    ([], _ALL),  # everything
    (["--components", "front-camera"], _ALL),  # not a scene filter: a full download
    (["--images", "half", "--transcode-backend", "cpu"], _ALL),  # neither
    (["--no-aerial-features"], _AERIAL_METADATA + [f"{_TILES}/area_1.zip", f"{_TILES}/area_2.zip"]),
    (["--no-aerial"], []),
    (["--no-aerial", "--no-aerial-features"], []),
    (["--splits", "val"], _AERIAL_METADATA + [f"{_TILES}/area_2.zip"]),
    (["--splits", "val", "--aerial-features"],
     _AERIAL_METADATA + _FEATURE_METADATA + [f"{_TILES}/area_2.zip", f"{_FEATURES}/area_2.zip"]),
    (["--splits", "val", "--no-aerial-features"], _AERIAL_METADATA + [f"{_TILES}/area_2.zip"]),
    (["--splits", "val", "--no-aerial"], []),
    (["--scenes", "scene_a", "--aerial-features", "--components", "none"],
     _AERIAL_METADATA + _FEATURE_METADATA + [f"{_TILES}/area_1.zip", f"{_FEATURES}/area_1.zip"]),
])
def test_aerial_cases(tmp_path: Path, flags: list, expected: list) -> None:
    hub = make_hub(tmp_path)
    _add_aerial(hub)
    root, calls = tmp_path / "local", []
    assert run_download(hub, root, flags, calls) == 0
    assert sorted(c for c in _aerial_calls(calls) if (root / c).is_file()) == sorted(expected)  # (not the index read)
    if not expected:
        assert not (root / "aerial").exists()
    assert main(["info", str(root), "-v"]) == 0  # a root without (some of) KITScenes Aerial is a normal root


@pytest.mark.parametrize("command", ["download", "extract"])
@pytest.mark.parametrize("off", [["--no-aerial-features"], ["--no-aerial"], ["--no-aerial", "--no-aerial-features"]])
def test_aerial_features_flags_conflict(tmp_path: Path, capsys, command: str, off: list) -> None:
    hub = make_hub(tmp_path)
    root = tmp_path / "local"
    with pytest.raises(SystemExit) as exit_:
        main([command, str(root), "--splits", "val", "--aerial-features", *off])
    assert exit_.value.code == 2
    assert "--aerial-features asks for the aerial features" in capsys.readouterr().err
    if command == "download":  # also without main()
        with pytest.raises(ValueError, match="give one"):
            run_download(hub, root, ["--aerial-features", *off], [])


def test_test_e2e_scenes_pull_no_aerial(tmp_path: Path) -> None:
    hub = make_hub(tmp_path)
    _add_aerial(hub)
    root, calls = tmp_path / "local", []
    assert run_download(hub, root, ["--splits", "test-e2e", "--aerial-features"], calls) == 0  # scene_c: area_1, no crops
    assert "data/test-e2e/scene_c/camera_ring_front.zip" in calls
    assert _aerial_calls(calls) == ["aerial/metadata/area_index.json"]  # read to select, not stored
    assert not (root / "aerial").exists()
    calls.clear()  # with a scene that needs the area, it comes
    assert run_download(hub, root, ["--splits", "train", "test-e2e"], calls) == 0
    assert _aerial_calls(calls) == sorted(_AERIAL_METADATA + [f"{_TILES}/area_1.zip"])


def test_components_refuses_aerial_names(tmp_path: Path, capsys) -> None:
    hub = make_hub(tmp_path)
    _add_aerial(hub)
    for name in ("aerial-tiles", "aerial-features", "aerial_tiles", "aerial_features", "aerial_metadata"):
        with pytest.raises(ValueError) as err:  # main() prints it, exit code 1
            run_download(hub, tmp_path / "local", ["--components", "all", name], [])
        text = str(err.value)
        assert f"Unknown component or preset {name!r}" in text
        assert "not a component" in text and "--no-aerial-features" in text
        with pytest.raises(ValueError, match="--aerial-features"):
            archives.resolve_components([name])
    with pytest.raises(ValueError) as err:
        archives.resolve_components(["lidar_roof"])
    assert "aerial" not in str(err.value)  # the hint only for aerial names
    assert main(["extract", str(tmp_path / "local"), "--components", "aerial-tiles"]) == 1
    assert "--no-aerial" in capsys.readouterr().err


def test_aerial_alongside_downscaled_images(tmp_path: Path) -> None:
    from kitscenes.transcode import ImageRecipe, read_variant

    hub = make_hub(tmp_path)
    _add_aerial(hub)
    root, calls = tmp_path / "local", []
    args = ["--splits", "val", "--components", "front-camera", "--images", "half", "--transcode-backend", "cpu",
            "--aerial-features"]
    assert run_download(hub, root, args, calls) == 0
    assert f"{_FEATURES}/area_2.zip" in calls and f"{_TILES}/area_2.zip" in calls
    for zip_path in (f"{_FEATURES}/area_2.zip", f"{_TILES}/area_2.zip"):  # as released
        assert (root / zip_path).read_bytes() == (hub / zip_path).read_bytes()
    assert read_variant(root / "data/val/scene_b", "camera_ring_front").recipe == ImageRecipe.parse("half")


def test_max_gb_counts_aerial_by_area(tmp_path: Path, capsys) -> None:
    """--max-gb bounds the total: an area's aerial zips count with its first kept scene, tiles before features
    (only when asked for: --max-gb filters scenes), each only if it still fits; the scene is kept either way."""
    gb = 1000 ** 3
    files = [archives.ArchiveFile(s, "train", "metadata", f"data/train/{s}/poses.txt", "0", 1 * gb)
             for s in ("s1", "s2", "s3")]
    files.append(archives.ArchiveFile("e1", "test-e2e", "metadata", "data/test-e2e/e1/poses.txt", "0", 1 * gb))
    files += [archives.ArchiveFile("", "", "aerial_tiles", f"aerial/raw_tiles/x/{a}.zip", "0", gb // 2)
              for a in ("area_a", "area_b")]
    files += [archives.ArchiveFile("", "", "aerial_features", f"aerial/features/f/{a}.zip", "0", 3 * gb)
              for a in ("area_a", "area_b")]
    files.append(archives.ArchiveFile("", "", "aerial_metadata", archives.AREA_INDEX_PATH, "0", 1))
    files.append(archives.ArchiveFile("", "", "aerial_features_metadata", "aerial/features/f/TERMS.md", "0", 1))
    index = {"scene_to_area": {"s1": "area_a", "s2": "area_a", "s3": "area_b", "e1": "area_b"}}
    pick = lambda **kw: [f.path for f in archives.select_files(  # noqa: E731
        files, area_index=index, **{"aerial_features": False, **kw})]
    # 5.6 GB with the features: s1 (1) + tiles a (0.5) + features a (3), s2 (1); s3 does not fit
    assert pick(max_gb=5.6, aerial_features=True) == [
        archives.AREA_INDEX_PATH, "aerial/features/f/TERMS.md", "aerial/raw_tiles/x/area_a.zip",
        "aerial/features/f/area_a.zip", "data/train/s1/poses.txt", "data/train/s2/poses.txt"]
    # without them: every scene, with the tiles of both areas
    assert pick(max_gb=5.6) == [archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_a.zip",
                                "aerial/raw_tiles/x/area_b.zip", "data/train/s1/poses.txt",
                                "data/train/s2/poses.txt", "data/train/s3/poses.txt", "data/test-e2e/e1/poses.txt"]
    # 4.1 GB: the features of area_a do not fit; the scenes do (s1, s2, s3 with the tiles of area_b)
    assert pick(max_gb=4.1, aerial_features=True) == pick(max_gb=4.1) == pick(max_gb=4.1, aerial_features=False)
    assert pick(max_gb=4.1) == [archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_a.zip",
                                "aerial/raw_tiles/x/area_b.zip", "data/train/s1/poses.txt",
                                "data/train/s2/poses.txt", "data/train/s3/poses.txt"]
    left = archives.aerial_left_out(
        files, archives.select_files(files, area_index=index, max_gb=4.1, aerial_features=True), index)
    assert sorted(f.path for f in left) == ["aerial/features/f/area_a.zip", "aerial/features/f/area_b.zip"]
    assert pick(max_gb=3, aerial_tiles=False) == [
        "data/train/s1/poses.txt", "data/train/s2/poses.txt", "data/train/s3/poses.txt"]
    assert pick(max_gb=1.2) == ["data/train/s1/poses.txt"]  # nothing aerial fits
    assert pick(splits=["test-e2e"], max_gb=100, aerial_features=True) == ["data/test-e2e/e1/poses.txt"]
    with pytest.raises(ValueError, match="area index"):
        archives.select_files(files, max_gb=4, aerial_features=False)
    assert archives.select_files(files, max_gb=4, aerial_tiles=False, aerial_features=False)  # no index needed
    # no scene filter: everything, no index needed
    assert len(archives.select_files(files, aerial_features=True)) == len(files)


def test_max_gb_note(tmp_path: Path, capsys) -> None:
    hub = make_hub(tmp_path)
    _add_aerial(hub)
    rows = list(csv.DictReader(open(hub / "data" / "sequence_archives.csv")))
    scene = sum(int(r["archive_size_bytes"]) for r in rows if r["sequence_id"] == "scene_b")
    tiles = next(int(r["archive_size_bytes"]) for r in rows if r["archive_path"] == f"{_TILES}/area_2.zip")
    meta = sum(int(r["archive_size_bytes"]) for r in rows if r["component"] == "aerial_metadata")
    budget = f"{(scene + tiles + meta + 0.5) / 1e9:.10f}"  # half a byte of room for the float
    calls = []
    assert run_download(hub, tmp_path / "local", ["--splits", "val", "--max-gb", budget, "--aerial-features"], calls) == 0
    out = capsys.readouterr().out
    assert "KITScenes Aerial of the selected scenes' areas that does not fit is left out: 1 zip(s)" in out
    assert "(aerial_features)" in out and "--aerial-features adds" not in out
    assert f"{_TILES}/area_2.zip" in calls and f"{_FEATURES}/area_2.zip" not in calls
    # without --aerial-features the features are not asked for: no --max-gb note, the hint instead
    assert run_download(hub, tmp_path / "other", ["--splits", "val", "--max-gb", budget], []) == 0
    out = capsys.readouterr().out
    assert "does not fit" not in out and "--aerial-features adds them (1 zip(s)" in out


def test_aerial_rows_select_and_validate(tmp_path: Path) -> None:
    from kitscenes.splits import scene_table

    infos = {}
    for info in scene_table().values():
        infos.setdefault(info.city, info)
    ka, fra = infos["Karlsruhe"], infos["Frankfurt"]
    files = [archives.ArchiveFile(i.scene_id, i.split, "metadata", f"data/{i.split}/{i.scene_id}/poses.txt", "0", 1)
             for i in (ka, fra)]
    files += [archives.ArchiveFile("", "", "aerial_tiles", f"aerial/raw_tiles/x/{a}.zip", "0", 1)
              for a in ("area_ka", "area_fra")]
    files.append(archives.ArchiveFile("", "", "aerial_metadata", archives.AREA_INDEX_PATH, "0", 1))
    index = {"scene_to_area": {ka.scene_id: "area_ka", fra.scene_id: "area_fra"}}
    pick = lambda **kw: sorted(f.path for f in archives.select_files(  # noqa: E731
        files, area_index=index, **{"aerial_features": False, **kw}) if f.path.startswith("aerial/"))
    assert pick(cities=["Karlsruhe"]) == [archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_ka.zip"]
    assert pick(cities=["Karlsruhe"], components=["none"]) == pick(cities=["Karlsruhe"])
    assert pick() == [archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_fra.zip", "aerial/raw_tiles/x/area_ka.zip"]
    assert pick(cities=["Karlsruhe"], aerial_tiles=False) == []
    with pytest.raises(ValueError, match="area index"):
        archives.select_files(files, cities=["Karlsruhe"], aerial_features=False)
    assert archives.needs_area_index(files, aerial_features=False, cities=["Karlsruhe"])
    assert not archives.needs_area_index(files, aerial_features=True)
    assert not archives.needs_area_index(files, aerial_tiles=False, aerial_features=False, cities=["Karlsruhe"])
    for component, path in (("aerial_tiles", "aerial/raw_tiles/x/../a.zip"), ("aerial_tiles", "aerial/a.zip"),
                            ("aerial_metadata", "data/x.json"), ("aerial_metadata", "aerial/.hidden"),
                            ("aerial_tiles", "/aerial/raw_tiles/x/a.zip")):
        with pytest.raises(ValueError, match="Unsafe"):
            archives.parse_manifest([{"sequence_id": "", "split": "", "component": component, "archive_path": path,
                                      "archive_sha256": "", "archive_size_bytes": "1"}])


def test_aerial_feature_rows_select_and_validate(tmp_path: Path) -> None:
    from kitscenes.splits import scene_table

    infos = {}
    for info in scene_table().values():
        infos.setdefault(info.city, info)
    ka, fra = infos["Karlsruhe"], infos["Frankfurt"]
    files = [archives.ArchiveFile(i.scene_id, i.split, "metadata", f"data/{i.split}/{i.scene_id}/poses.txt", "0", 1)
             for i in (ka, fra)]
    files += [archives.ArchiveFile("", "", "aerial_tiles", f"aerial/raw_tiles/x/{a}.zip", "0", 1)
              for a in ("area_ka", "area_fra")]
    files += [archives.ArchiveFile("", "", "aerial_features", f"aerial/features/s/{a}.zip", "0", 1)
              for a in ("area_ka", "area_fra")]
    files.append(archives.ArchiveFile("", "", "aerial_metadata", archives.AREA_INDEX_PATH, "0", 1))
    files += [archives.ArchiveFile("", "", "aerial_features_metadata", f"aerial/features/s/{n}", "0", 1)
              for n in ("manifest.json", "TERMS.md")]
    index = {"scene_to_area": {ka.scene_id: "area_ka", fra.scene_id: "area_fra"}}
    pick = lambda **kw: sorted(f.path for f in archives.select_files(  # noqa: E731
        files, area_index=index, **{"aerial_features": False, **kw}) if f.path.startswith("aerial/"))
    assert pick(cities=["Karlsruhe"], aerial_tiles=False) == []  # no features unless asked for
    assert pick(cities=["Karlsruhe"], aerial_tiles=False, aerial_features=True) == [
        "aerial/features/s/TERMS.md", "aerial/features/s/area_ka.zip", "aerial/features/s/manifest.json",
        archives.AREA_INDEX_PATH]
    assert pick(aerial_features=False) == [
        archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_fra.zip", "aerial/raw_tiles/x/area_ka.zip"]
    assert pick(cities=["Frankfurt"], components=["none"]) == [archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_fra.zip"]
    assert len(pick(aerial_features=True)) == 7  # tiles and features of every area
    assert len(pick(components=["none"], aerial_features=True)) == 7
    both = pick(cities=["Frankfurt"], components=["none"], aerial_features=True)
    assert both == ["aerial/features/s/TERMS.md", "aerial/features/s/area_fra.zip", "aerial/features/s/manifest.json",
                    archives.AREA_INDEX_PATH, "aerial/raw_tiles/x/area_fra.zip"]
    assert pick(cities=["Karlsruhe"], aerial_tiles=False, aerial_features=False) == []
    assert archives.aerial_kinds() == ("aerial_tiles", "aerial_features")
    assert archives.aerial_kinds(aerial_features=False) == ("aerial_tiles",)
    with pytest.raises(ValueError, match="area index"):
        archives.select_files(files, cities=["Karlsruhe"], aerial_tiles=False, aerial_features=True)
    for component, path in (("aerial_features", "aerial/features/s/../a.zip"), ("aerial_features", "aerial/s/a.zip"),
                            ("aerial_features", "aerial/raw_tiles/s/a.zip"),
                            ("aerial_features_metadata", "aerial/features/s/a.zip"),
                            ("aerial_features_metadata", "aerial/features/manifest.json")):
        with pytest.raises(ValueError, match="Unsafe"):
            archives.parse_manifest([{"sequence_id": "", "split": "", "component": component, "archive_path": path,
                                      "archive_sha256": "", "archive_size_bytes": "1"}])


def test_feature_zips_extract_in_place(tmp_path: Path) -> None:
    import numpy as np

    from kitscenes.aerial import AerialFeatures
    write_features(tmp_path, layout="zip")
    zip_path = tmp_path / "aerial" / "features" / SET / f"{AREA}.zip"
    f = archives.ArchiveFile("", "", "aerial_features", zip_path.relative_to(tmp_path).as_posix(),
                             archives.sha256_file(zip_path), zip_path.stat().st_size)
    assert AerialFeatures(tmp_path).location(AREA, 1, 1).member is not None
    archives.extract_archive(tmp_path, f)
    assert not zip_path.exists() and archives.is_complete(tmp_path, f)
    feats = AerialFeatures(tmp_path)
    assert feats.location(AREA, 1, 1).member is None and np.array_equal(feats.tile(AREA, 1, 1), token_field(1, 1))


def test_pack_takes_the_scene_selection(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    for name in ("s1", "s2"):
        make_scene(src / "data" / "train" / name, n=2, cameras=("camera_ring_front",), lidars=(), radars=())
    assert main(["pack", str(src), "--out", str(out), "--scenes", "s2"]) == 0
    assert [p.name for p in (out / "data" / "train").iterdir()] == ["s2"]
    assert (out / "data/train/s2/camera_ring_front.zip").is_file()
    with pytest.raises(SystemExit):  # the cameras and lidars are chosen when downloading or extracting
        main(["pack", str(src), "--out", str(out), "--components", "front-camera"])


def _commands(parser: argparse.ArgumentParser, prefix: tuple = ()) -> list[tuple]:
    """Every command and subcommand of *parser*, as argument tuples."""
    found = [prefix]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                found += _commands(child, (*prefix, name))
    return found


@pytest.mark.parametrize("command", _commands(build_parser()) + [("aerial", c) for c in ("list", "check", "store", "prepare")],
                         ids=lambda c: " ".join(c) or "kitscenes")
def test_help_of_every_command(command: tuple, capsys) -> None:
    """``kitscenes --help`` and the help of every subcommand print a usage text and exit with 0."""
    if command[:1] == ("view",):
        pytest.importorskip("kitscenes_viewer")
    with pytest.raises(SystemExit) as done:
        main([*command, "--help"])
    assert done.value.code == 0 and "usage:" in capsys.readouterr().out
