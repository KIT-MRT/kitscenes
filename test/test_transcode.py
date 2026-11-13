"""Downscaled camera images (kitscenes.transcode)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import kitscenes
from kitscenes.cli import main
from kitscenes.errors import ImageSizeError
from kitscenes.instances import InstanceMasks
from kitscenes.pack import read_members
from kitscenes.transcode import (ImageRecipe, Transcoder, estimate_size, read_policy, read_variant, transcode,
                                 transcode_jpeg)
from hub_fixture import FakeHub, hub_download, make_hub, run_download
from release_fixture import instances_file, make_scene, pack_scene as fixture_pack_scene

CAMS = ("camera_ring_front", "camera_ring_rear")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def cpu() -> Transcoder:
    with Transcoder("cpu", jobs=2) as t:
        yield t


def _root(tmp_path: Path, *, packed: bool = True, n: int = 3) -> Path:
    root = tmp_path / "root"
    scene = make_scene(root / "data" / "train" / "s1", n=n, cameras=CAMS)
    if packed:
        fixture_pack_scene(scene)
    return root


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


def test_recipes() -> None:
    assert ImageRecipe.parse("half") == ImageRecipe(2, 90, "4:2:0")
    assert ImageRecipe.parse("quarter,quality=95") == ImageRecipe(4, 95, "4:2:0")
    assert ImageRecipe.parse("scale=1/8,subsampling=4:4:4") == ImageRecipe(8, 90, "4:4:4")
    assert ImageRecipe.parse("scale=1,quality=85") == ImageRecipe(1, 85, "4:2:0")
    assert str(ImageRecipe.parse("half")) == "1/2, q90, 4:2:0"
    for bad in ("third", "scale=1/3", "quality=101", "subsampling=4:1:1", "scale=2", "foo=1", ""):
        with pytest.raises(ValueError):
            ImageRecipe.parse(bad)
    assert ImageRecipe.from_dict(ImageRecipe(4, 85, "4:4:4").to_dict()) == ImageRecipe(4, 85, "4:4:4")
    assert estimate_size(6_300_000, ImageRecipe.parse("half")) == 1_000_000
    assert estimate_size(10**6, ImageRecipe.parse("half,quality=95")) > estimate_size(10**6, ImageRecipe.parse("half"))


def test_transcode_one_jpeg() -> None:
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :32] = 200
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=98)
    out = Image.open(io.BytesIO(transcode_jpeg(buf.getvalue(), "half")))
    assert out.size == (32, 24) and out.format == "JPEG"
    assert abs(int(np.asarray(out)[12, 4, 0]) - 200) <= 3 and int(np.asarray(out)[12, 28, 0]) <= 3
    with pytest.raises(ValueError, match="does not divide"):
        odd = io.BytesIO()
        Image.fromarray(img[:47]).save(odd, "JPEG")
        transcode_jpeg(odd.getvalue(), "half")


# ---------------------------------------------------------------------------
# Transcoding a root
# ---------------------------------------------------------------------------


def _check_half(scene: kitscenes.Scene, camera: str = "camera_ring_front") -> None:
    cam, orig = scene.rig.cameras[camera], scene.files.rig_original.cameras[camera]
    assert cam.image_size == (32, 24) and orig.image_size == (64, 48)
    assert scene.frame(1).image(camera).shape == (24, 32, 3)
    assert scene.frame(1).image(camera, reduce=2).shape == (12, 16, 3)
    xyz = np.array([[10.0, 0.5, 1.2], [8.0, -1.0, 2.0], [20.0, 2.0, 1.0]])
    uv, _, _ = cam.project(xyz)
    uv0, _, _ = orig.project(xyz)
    np.testing.assert_allclose(uv, (uv0 + 0.5) / 2 - 0.5)
    img = scene.frame(1).image(camera).astype(int)  # the fixture's white top-left block survives
    assert img[2, 2].min() > 240 and img[20, 28].max() < 200


def test_transcode_zip_in_place(tmp_path: Path, cpu: Transcoder) -> None:
    root = _root(tmp_path)
    scene_dir = root / "data/train/s1"
    lidar_sha, index_sha = sha(scene_dir / "lidar_top.zip"), sha(scene_dir / "index.parquet")
    result = transcode(root, "half", transcoder=cpu)
    assert len(result.done) == 2 and not result.failed
    assert sha(scene_dir / "lidar_top.zip") == lidar_sha and sha(scene_dir / "index.parquet") == index_sha
    variant = read_variant(scene_dir, "camera_ring_front")
    assert variant.layout == "zip" and variant.sha256 == sha(scene_dir / "camera_ring_front.zip")
    assert variant.original_size == (64, 48) and variant.image_size == (32, 24) and variant.images == 3
    assert all(m.data_offset % 4096 == 0 for m in read_members(scene_dir / "camera_ring_front.zip"))
    scene = kitscenes.open(root).scene("s1")
    _check_half(scene)
    assert scene.files.image_variants["camera_ring_front"] == variant and "lidar_top" not in scene.files.image_variants
    where = scene.frame(0).sensor_file("camera_ring_front")
    with open(where.path, "rb") as fh:
        fh.seek(where.offset)
        assert Image.open(io.BytesIO(fh.read(where.size))).size == (32, 24)  # DALI reads these offsets
    assert transcode(root, "half", transcoder=cpu).skipped and not result.failed

    # further down from the stored images, with the history; back up is refused
    transcode(root, "quarter", cameras=["camera_ring_front"], transcoder=cpu)
    q = read_variant(scene_dir, "camera_ring_front")
    assert q.image_size == (16, 12) and q.history == (ImageRecipe(2, 90, "4:2:0"),)
    assert kitscenes.open(root).scene("s1").frame(0).image("camera_ring_front").shape == (12, 16, 3)
    failed = transcode(root, "half", cameras=["camera_ring_front"], transcoder=cpu).failed
    assert failed and "release's images" in failed[0][2]


def test_transcode_extracted_in_place(tmp_path: Path, cpu: Transcoder) -> None:
    root = _root(tmp_path, packed=False)
    transcode(root, "half", transcoder=cpu)
    scene_dir = root / "data/train/s1"
    assert read_variant(scene_dir, "camera_ring_rear").layout == "folder"
    assert not (scene_dir / "camera_ring_rear.zip").exists() and len(list((scene_dir / "camera_ring_rear").iterdir())) == 3
    assert not any(p.name.startswith(".camera") for p in scene_dir.iterdir())
    _check_half(kitscenes.open(root).scene("s1"), "camera_ring_rear")


def test_transcode_into_a_new_root(tmp_path: Path, cpu: Transcoder) -> None:
    src = _root(tmp_path, packed=False)
    (src / "lanelet2_hd_maps").mkdir()
    (src / "lanelet2_hd_maps" / "Karlsruhe.osm").write_text("<osm/>")
    before = {p: sha(p) for p in src.rglob("*") if p.is_file()}
    dst = tmp_path / "half"
    result = transcode(src, "half", out=dst, transcoder=cpu)
    assert len(result.done) == 2 and {p: sha(p) for p in src.rglob("*") if p.is_file()} == before  # read only
    d = dst / "data/train/s1"
    assert (d / "camera_ring_front").is_dir() and (d / "lidar_top").is_dir() and not list(d.glob("*.zip"))
    assert (dst / "lanelet2_hd_maps" / "Karlsruhe.osm").is_file()
    a, b = kitscenes.open(src).scene("s1"), kitscenes.open(dst).scene("s1")
    _check_half(b)
    np.testing.assert_array_equal(a.frame(1).lidar("lidar_top").xyz, b.frame(1).lidar("lidar_top").xyz)

    # archives stay archives; a camera that is not selected is copied with its record
    packed = _root(tmp_path / "packed")
    transcode(packed, "half", transcoder=cpu)
    out2 = tmp_path / "quarter"
    transcode(packed, "quarter", out=out2, cameras=["front-camera"], transcoder=cpu)
    d2 = out2 / "data/train/s1"
    assert (d2 / "camera_ring_front.zip").is_file() and (d2 / "lidar_top.zip").is_file()
    c = kitscenes.open(out2, verify=True).scene("s1")
    assert c.frame(0).image("camera_ring_front").shape == (12, 16, 3)
    assert c.files.image_variants["camera_ring_front"].history == (ImageRecipe(2, 90, "4:2:0"),)
    assert c.frame(0).image("camera_ring_rear").shape == (24, 32, 3)
    # and so is a camera already in the recipe
    out3 = tmp_path / "same"
    assert len(transcode(packed, "half", out=out3, transcoder=cpu).skipped) == 2
    assert read_variant(out3 / "data/train/s1", "camera_ring_rear") == read_variant(packed / "data/train/s1", "camera_ring_rear")
    _check_half(kitscenes.open(out3, verify=True).scene("s1"))


def test_resized_images_without_a_record_are_refused(tmp_path: Path) -> None:
    root = _root(tmp_path, packed=False)
    path = root / "data/train/s1/camera_ring_front/0000000001.jpg"
    Image.open(path).resize((32, 24)).save(path)
    frame = kitscenes.open(root).scene("s1").frame(1)
    with pytest.raises(ImageSizeError, match="32x24, but the calibration says 64x48"):
        frame.image("camera_ring_front")


def test_instance_masks_at_another_size() -> None:
    from kitscenes.instances import UNLABELLED
    a = np.zeros((48, 64), bool)
    a[10:30, 10:30] = True
    inst = InstanceMasks.from_json(instances_file([(a, 0.9, 58)]), "camera_ring_front")
    labels = inst.label_image(size=(32, 24))
    assert labels.shape == (24, 32) and labels[10, 10] == 58 and labels[0, 0] == UNLABELLED
    np.testing.assert_array_equal(inst.mask(0, size=(32, 24)), a[::2, ::2] | a[1::2, 1::2])
    np.testing.assert_allclose(inst.scaled_boxes((32, 24)), inst.boxes / 2)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def test_download_with_images(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "root", []
    assert run_download(hub, root, ["--splits", "train", "--images", "half", "--transcode-backend", "cpu",
                                 "--staging-gb", "1e-9"], calls) == 0
    scene_dir = root / "data/train/scene_a"
    manifest = {r.split(",")[3]: r.split(",")[4] for r in (hub / "data/sequence_archives.csv").read_text().splitlines()[1:]}
    for cam in CAMS:
        variant = read_variant(scene_dir, cam)
        assert variant.recipe == ImageRecipe.parse("half")
        assert variant.source_sha256 == manifest[f"data/train/scene_a/{cam}.zip"]
    assert sha(scene_dir / "lidar_top.zip") == manifest["data/train/scene_a/lidar_top.zip"]
    assert not (root / ".kitscenes" / "staging").exists()
    assert read_policy(root) == {c: ImageRecipe.parse("half") for c in CAMS}
    _check_half(kitscenes.open(root).scene("scene_a"))

    # nothing to do again, also without --images (the root keeps the recipe), and with --verify
    calls.clear()
    assert run_download(hub, root, ["--splits", "train"], calls) == 0
    assert run_download(hub, root, ["--splits", "train", "--verify"], calls) == 0 and calls == []
    # another recipe is refused; the release's images come back with --images original
    assert run_download(hub, root, ["--splits", "train", "--images", "quarter"], calls) == 2
    assert run_download(hub, root, ["--splits", "train", "--images", "original"], calls) == 0
    assert sorted(calls) == [f"data/train/scene_a/{c}.zip" for c in CAMS]
    assert read_variant(scene_dir, "camera_ring_front") is None and read_policy(root) == {}
    assert all(sha(scene_dir / f"{c}.zip") == manifest[f"data/train/scene_a/{c}.zip"] for c in CAMS)
    assert kitscenes.open(root).scene("scene_a").frame(0).image("camera_ring_front").shape == (48, 64, 3)

    # a root with the release's images: transcoded in place, nothing downloaded
    calls.clear()
    assert run_download(hub, root, ["--splits", "train", "--images", "half", "--transcode-backend", "cpu"], calls) == 0
    assert calls == [] and read_variant(scene_dir, "camera_ring_rear").source_sha256 is not None


def test_download_with_images_and_extract(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "root", []
    assert run_download(hub, root, ["--splits", "val", "--components", "front-camera", "--images", "quarter",
                                 "--transcode-backend", "cpu", "--extract"], calls) == 0
    scene_dir = root / "data/val/scene_b"
    assert (scene_dir / "camera_ring_front").is_dir() and not (scene_dir / "camera_ring_front.zip").exists()
    marker = (scene_dir / ".extracted" / "camera_ring_front.zip.sha256").read_text().split()[0]
    assert marker == read_variant(scene_dir, "camera_ring_front").sha256
    assert kitscenes.open(root).scene("scene_b").frame(0).image("camera_ring_front").shape == (12, 16, 3)
    calls.clear()
    assert run_download(hub, root, ["--splits", "val", "--components", "front-camera"], calls) == 0 and calls == []


def test_update_re_transcodes_a_changed_camera(tmp_path: Path, capsys) -> None:
    import csv

    a = make_hub(tmp_path / "a")
    b = tmp_path / "b" / "hub"
    shutil.copytree(a, b)
    changed = b / "data/train/scene_a/camera_ring_front.zip"
    changed.write_bytes(changed.read_bytes() + b"\0")
    rows = list(csv.DictReader(open(b / "data/sequence_archives.csv")))
    for row in rows:
        if row["archive_path"] == "data/train/scene_a/camera_ring_front.zip":
            row["archive_sha256"], row["archive_size_bytes"] = sha(changed), changed.stat().st_size
    with open(b / "data/sequence_archives.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hub, root = FakeHub({"rev-a": a, "rev-b": b}, main="rev-a"), tmp_path / "root"
    assert hub_download(hub, root, ["--splits", "train", "--images", "half", "--transcode-backend", "cpu"]) == 0
    scene_dir = root / "data/train/scene_a"
    old = read_variant(scene_dir, "camera_ring_front")
    rear = read_variant(scene_dir, "camera_ring_rear")
    hub.main = "rev-b"
    assert hub_download(hub, root, ["--update", "--transcode-backend", "cpu"]) == 0
    new = read_variant(scene_dir, "camera_ring_front")
    assert new.source_sha256 == sha(changed) != old.source_sha256
    assert read_variant(scene_dir, "camera_ring_rear") == rear  # unchanged: left alone
    assert json.loads((root / "dataset_info.json").read_text())["images"]["camera_ring_front"]["reduce"] == 2


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_transcode_pack_and_info_commands(tmp_path: Path, capsys) -> None:
    src = _root(tmp_path, packed=False)
    packed = tmp_path / "packed"
    assert main(["pack", str(src), "--out", str(packed), "--dry-run"]) == 0 and not packed.exists()
    assert "train/s1: would be packed" in capsys.readouterr().out
    assert main(["pack", str(src), "--out", str(packed)]) == 0
    assert (packed / "data/train/s1/camera_ring_front.zip").is_file()
    assert main(["transcode", str(packed), "--images", "half", "--dry-run"]) == 0
    assert "2 camera(s) would be transcoded" in capsys.readouterr().out
    assert read_variant(packed / "data/train/s1", "camera_ring_rear") is None and not (packed / ".kitscenes").exists()
    assert main(["transcode", str(packed), "--images", "half", "--cameras", "camera_ring_rear", "--backend", "cpu",
                 "--jobs", "2"]) == 0
    out = capsys.readouterr().out
    assert "1 camera(s) transcoded" in out
    assert main(["info", str(packed)]) == 0
    assert "camera_ring_rear: 32x24" in capsys.readouterr().out
    assert main(["transcode", str(packed), "--images", "third"]) == 1
    for command in (["pack", "--out", str(tmp_path / "x")], ["transcode", "--images", "half", "--dry-run"]):
        assert main([command[0], str(packed / "data")] + command[1:]) == 1  # the root is the folder that holds data/
        assert "has no data/ folder" in capsys.readouterr().err


def test_dry_runs_write_nothing(tmp_path: Path, capsys) -> None:
    """``kitscenes transcode --dry-run`` and ``kitscenes pack --dry-run`` list their work and leave every file as it is."""
    def files() -> dict:
        return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(tmp_path.rglob("*"))}

    root = _root(tmp_path, packed=True)
    extracted = _root(tmp_path / "extracted", packed=False)
    before = files()
    assert main(["transcode", str(root), "--images", "half", "--dry-run"]) == 0
    assert main(["transcode", str(root), "--images", "half", "--out", str(tmp_path / "half"), "--dry-run"]) == 0
    assert "would be transcoded" in capsys.readouterr().out
    assert main(["pack", str(extracted), "--out", str(tmp_path / "packed"), "--dry-run"]) == 0
    assert "would be packed" in capsys.readouterr().out
    assert files() == before


# ---------------------------------------------------------------------------
# GPU (DALI + nvImageCodec); opt-in: KITSCENES_TEST_GPU=<device>
# ---------------------------------------------------------------------------


@pytest.mark.skipif("KITSCENES_TEST_GPU" not in os.environ, reason="set KITSCENES_TEST_GPU=<device> to test the GPU")
def test_gpu_matches_cpu(tmp_path: Path, cpu: Transcoder) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:128]
    datas = []
    for i in range(20):
        img = np.stack([(xx * 2 + i) % 256, (yy * 2) % 256, ((xx + yy) * 3) % 256], -1).astype(np.uint8)
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=98, subsampling=0)
        datas.append(buf.getvalue())
    device = int(os.environ["KITSCENES_TEST_GPU"])
    with Transcoder("gpu", gpus=[device]) as gpu:
        assert gpu.kind == "gpu"
        for recipe in ("half", "quarter", "full"):
            r = ImageRecipe.parse(recipe)
            g = list(gpu.map(datas, r))
            c = list(cpu.map(datas, r))
            for (gb, gin, gout), (cb, cin, cout) in zip(g, c):
                assert gin == cin and gout == cout
                diff = np.abs(np.asarray(Image.open(io.BytesIO(gb)), float) - np.asarray(Image.open(io.BytesIO(cb)), float))
                assert diff.mean() < 3, (recipe, diff.mean())
        root = _root(tmp_path)
        transcode(root, "half", transcoder=gpu)
        _check_half(kitscenes.open(root).scene("s1"))
        assert read_variant(root / "data/train/s1", "camera_ring_front").backend.startswith("gpu (DALI")


def test_extracted_release_images_are_transcoded_in_place_once(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "root", []
    assert run_download(hub, root, ["--splits", "train", "--extract"], calls) == 0
    calls.clear()
    assert run_download(hub, root, ["--splits", "train", "--images", "half", "--transcode-backend", "cpu"], calls) == 0
    scene_dir = root / "data/train/scene_a"
    assert calls == [] and read_variant(scene_dir, "camera_ring_front").layout == "folder"
    assert run_download(hub, root, ["--splits", "train"], calls) == 0 and calls == []  # nothing again
    _check_half(kitscenes.open(root).scene("scene_a"))


def test_extract_command_checks_a_transcoded_zip_against_its_record(tmp_path: Path, capsys) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "root", []
    assert run_download(hub, root, ["--splits", "train", "--images", "half", "--transcode-backend", "cpu"], calls) == 0
    assert main(["extract", str(root), "--components", "front-camera"]) == 0
    scene_dir = root / "data/train/scene_a"
    assert (scene_dir / "camera_ring_front").is_dir() and not (scene_dir / "camera_ring_front.zip").exists()
    calls.clear()
    assert run_download(hub, root, ["--splits", "train"], calls) == 0 and calls == []
    _check_half(kitscenes.open(root).scene("scene_a"))


# ---------------------------------------------------------------------------
# Downloads, crashes and packing around a transcode
# ---------------------------------------------------------------------------


def test_a_download_after_transcode_keeps_the_transcoded_cameras(tmp_path: Path) -> None:
    hub, root, calls = make_hub(tmp_path), tmp_path / "root", []
    assert run_download(hub, root, ["--splits", "train"], calls) == 0
    assert main(["transcode", str(root), "--images", "half", "--backend", "cpu", "--jobs", "2"]) == 0
    assert read_variant(root / "data/train/scene_a", "camera_ring_front").source_sha256 is not None
    calls.clear()
    assert run_download(hub, root, ["--splits", "train"], calls) == 0 and calls == []
    _check_half(kitscenes.open(root).scene("scene_a"))
    # another recipe for some scenes only would split the root's recipe: refused
    assert main(["transcode", str(root), "--images", "quarter", "--scenes", "scene_a", "--backend", "cpu"]) == 1
    # a new root keeps the recipe for downloads into it too
    out = tmp_path / "out"
    assert main(["transcode", str(root), "--out", str(out), "--images", "half", "--backend", "cpu"]) == 0
    calls.clear()
    assert run_download(hub, out, ["--splits", "train"], calls) == 0 and calls == []


def test_a_transcoder_that_cannot_start_fails_the_download(tmp_path: Path, monkeypatch) -> None:
    import kitscenes.transcode as T

    def broken(self, *args, **kwargs):
        raise T.TranscoderUnavailable("no GPU here")

    monkeypatch.setattr(T.Transcoder, "__init__", broken)
    hub, root = make_hub(tmp_path), tmp_path / "root"
    with pytest.raises(T.TranscoderUnavailable):
        run_download(hub, root, ["--splits", "train", "--images", "half", "--staging-gb", "1e-9"], [])
    assert not (root / ".kitscenes" / "staging").exists() or not any((root / ".kitscenes" / "staging").rglob("*.zip"))


@pytest.mark.parametrize("packed", [True, False])
def test_a_crash_while_swapping_leaves_a_consistent_scene(tmp_path: Path, cpu: Transcoder, monkeypatch, packed) -> None:
    import kitscenes.transcode as T

    root = _root(tmp_path, packed=packed)
    scene_dir = root / "data/train/s1"
    real = os.replace

    def crash_at_the_images(src, dst):
        if Path(dst).name in ("camera_ring_front.zip", "camera_ring_front"):
            raise KeyboardInterrupt("crash")
        return real(src, dst)

    for recipe, before_size in (("half", (64, 48)), ("quarter", (32, 24))):
        monkeypatch.setattr(T.os, "replace", crash_at_the_images)
        with pytest.raises(KeyboardInterrupt):
            T.transcode_camera(scene_dir, "camera_ring_front", recipe, cpu)
        monkeypatch.setattr(T.os, "replace", real)
        if packed:  # the zip on disk is the old one, and so are rig and index (a folder is restored by the rerun)
            scene = kitscenes.open(root).scene("s1")
            assert scene.rig.cameras["camera_ring_front"].image_size == before_size
            assert scene.frame(1).image("camera_ring_front").shape[1::-1] == before_size
        T.transcode_camera(scene_dir, "camera_ring_front", recipe, cpu)  # the rerun finishes the job
        v = read_variant(scene_dir, "camera_ring_front")
        assert v.recipe == ImageRecipe.parse(recipe) and v.original_size == (64, 48)
        assert not list((scene_dir / ".transcoded").glob("*.prev.*"))
    assert read_variant(scene_dir, "camera_ring_front").history == (ImageRecipe.parse("half"),)
