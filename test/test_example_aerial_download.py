"""examples/download_aerial_tiles.py with a fake WMS: the tile folder it writes is read like the release tiles."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import kitscenes
from kitscenes.aerial import AerialCrops, AerialTiles

from aerial_fixture import FRANKFURT, KARLSRUHE, paint, write_scene, write_tiles

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "download_aerial_tiles.py"
ARGS = ["--wms-url", "http://wms.invalid/ows", "--layer", "ortho", "--param", "TIME=2019", "--resolution", "1",
        "--tile-size-px", "20", "--margin-m", "12", "--pause-s", "0"]


@pytest.fixture
def tool(monkeypatch):
    """The script as a module, with its HTTP request replaced: tiles painted with their position, one blank answer."""
    spec = importlib.util.spec_from_file_location("download_aerial_tiles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.calls = []

    def fake(url, layer, bbox, size_px, fmt, params):
        module.calls.append((url, layer, bbox, params))
        pixels = paint(bbox[0], bbox[3], size_px)
        if len(module.calls) == 2:  # a blank tile, as a service sends now and then: asked again
            pixels = np.zeros_like(pixels)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return buffer.getvalue()

    monkeypatch.setattr(module, "fetch_tile", fake)
    monkeypatch.delenv("KITSCENES_ROOT", raising=False)
    return module


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    write_scene(root / "data" / "train" / "scene_frankfurt", FRANKFURT, yaw_rate=0.7)
    write_scene(root / "data" / "train" / "scene_karlsruhe", KARLSRUHE)
    write_scene(root / "data" / "test" / "scene_test")  # no map origin: left out
    return root


def files(folder: Path) -> dict:
    return {p: p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}


def test_tiles_are_read_like_the_release_tiles(tool, root: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "tiles"
    before = files(root)
    assert tool.main([str(root), "--out", str(out), *ARGS, "--dry-run"]) == 0
    assert not out.exists() and tool.calls == [] and "tiles to fetch" in capsys.readouterr().out

    assert tool.main([str(root), "--out", str(out), *ARGS]) == 0
    assert files(root) == before  # the root is only read
    tiles = AerialTiles(out)
    assert tiles.raw_cache_id == "wms_res100_tile20" and len(tiles.area_ids) == 2
    assert len(tool.calls) == tiles.num_tiles + 1  # the blank answer was asked again
    assert all(call[:2] == ("http://wms.invalid/ows", "ortho") and call[3] == {"TIME": "2019"} for call in tool.calls)
    assert f"fetched {tiles.num_tiles}, skipped 0 (already there), failed 0" in capsys.readouterr().out
    source = (out / "SOURCE.md").read_text()
    assert "http://wms.invalid/ows" in source and "`ortho`" in source and "TIME=2019" in source

    ds = kitscenes.open(root)
    crops = ds.use_aerial(roi=(6.0, 4.0), tiles=out)
    assert crops.cache_id == "wms_res100_roi6x4_ego_v1" and crops.coverage(ds).num_uncovered == 0
    reference = AerialCrops(write_tiles(root, tmp_path / "reference"), (6.0, 4.0))  # the same paint, by the fixture
    for frame in ds.filter(splits="train").frames():
        np.testing.assert_array_equal(crops[frame], reference[frame])

    done, written = len(tool.calls), files(out)
    assert tool.main([str(root), "--out", str(out), *ARGS]) == 0  # a second run fetches nothing
    assert len(tool.calls) == done and files(out) == written
    assert tool.main([str(root), "--out", str(out), *ARGS, "--margin-m", "30"]) == 0  # a wider margin: only the rest
    assert AerialTiles(out).num_tiles > tiles.num_tiles
    assert f"skipped {tiles.num_tiles} (already there)" in capsys.readouterr().out


def test_refusals(tool, root: Path, tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="inside the dataset root"):
        tool.main([str(root), "--out", str(root / "aerial"), *ARGS])
    monkeypatch.setenv("KITSCENES_ROOT", str(tmp_path / "release"))
    with pytest.raises(SystemExit, match="inside the dataset root"):
        tool.main([str(root), "--out", str(tmp_path / "release" / "aerial_new"), *ARGS])
    with pytest.raises(SystemExit, match="go together"):
        tool.main([str(root), "--out", str(tmp_path / "tiles"), "--wms-url", "http://wms.invalid/ows"])
    with pytest.raises(SystemExit, match="names no service"):  # synthetic scenes have no city: no state service
        tool.main([str(root), "--out", str(tmp_path / "tiles"), "--dry-run"])
    with pytest.raises(SystemExit, match="no map origin"):
        tool.main([str(root), "--out", str(tmp_path / "tiles"), *ARGS, "--splits", "test"])
    assert tool.calls == [] and not (tmp_path / "tiles").exists() and not (root / "aerial").exists()
