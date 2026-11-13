"""``kitscenes aerial`` / ``python -m kitscenes.aerial``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kitscenes.catalog import Dataset
from kitscenes.cli import main as cli_main
from aerial_fixture import FRANKFURT, KARLSRUHE, write_scene, write_tiles

CACHE_ID = "dop20_res100_roi6x4_ego_v1"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    write_scene(root / "data" / "train" / "scene_frankfurt", FRANKFURT)
    write_scene(root / "data" / "val" / "scene_karlsruhe", KARLSRUHE)
    write_scene(root / "data" / "test" / "scene_test")
    return root


def test_prepare_passes_the_selection_and_parameters(root: Path, monkeypatch, capsys) -> None:
    calls = []
    real = Dataset.prepare_aerials

    def spy(self, roi, **kwargs):
        calls.append((list(self.scene_ids), roi, kwargs))
        return real(self, roi, **kwargs)

    monkeypatch.setattr(Dataset, "prepare_aerials", spy)
    write_tiles(root, scenes=["scene_karlsruhe"])
    assert cli_main([
        "aerial", "prepare", str(root), "--splits", "train", "val", "--scenes", "scene_karlsruhe",
        "--roi", "6", "4", "--jobs", "1",
    ]) == 0
    (scene_ids, roi, kwargs), = calls
    assert scene_ids == ["scene_karlsruhe"]
    assert roi == (6.0, 4.0) and kwargs["workers"] == 1 and kwargs["version"] == "v1"
    assert kwargs["tiles"] is None and kwargs["cache_dir"] is None
    out = capsys.readouterr().out
    assert CACHE_ID in out and "3 frames with a crop in 1 scenes" in out

    assert cli_main(["aerial", "list", str(root)]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("AerialTiles(") and "6 tiles in 1 areas, png" in out[0]
    assert out[1] == f"crop cache {CACHE_ID}: 3 crops in 1 scenes"


def test_prepare_without_tiles_and_check(root: Path, capsys) -> None:
    assert cli_main(["aerial", "prepare", str(root), "--roi", "6", "4"]) == 1
    assert "No raw aerial tiles" in capsys.readouterr().err
    assert cli_main(["aerial", "check", str(root), "--roi", "6", "4"]) == 1
    assert "come with every 'kitscenes download ROOT'" in capsys.readouterr().err
    assert not (root / "aerial").exists()

    # tiles for the Frankfurt scene only: Karlsruhe is reported, the exit code says incomplete
    write_tiles(root, scenes=["scene_frankfurt"])
    assert cli_main(["aerial", "check", str(root), "--roi", "6", "4", "--splits", "train", "val",
                     "--json", str(root / "report.json")]) == 2
    out = capsys.readouterr().out
    assert "3 frames with a crop in 2 scenes; 3 frames lack raw tiles (1 scenes incomplete)" in out
    assert "scene_karlsruhe (val): 3 of 3 frames lack raw tiles" in out
    report = json.loads((root / "report.json").read_text())
    assert report["complete"] is False and report["scenes"]["scene_karlsruhe"]["uncovered_frames"] == [0, 1, 2]
    assert cli_main(["aerial", "prepare", str(root), "--roi", "6", "4", "--jobs", "1"]) == 2  # Karlsruhe gets no crops


def test_prepare_without_georeferenced_scenes_fails_cleanly(root: Path, capsys) -> None:
    assert cli_main(["aerial", "prepare", str(root), "--splits", "test", "--roi", "6", "4"]) == 1
    assert "map origin" in capsys.readouterr().err
    assert not (root / "aerial").exists()


def test_list_without_caches(root: Path, capsys) -> None:
    assert cli_main(["aerial", "list", str(root)]) == 0
    out = capsys.readouterr().out
    assert "No raw tiles" in out and "No crop caches" in out


def test_module_entry_point_needs_roi(root: Path, capsys) -> None:
    from kitscenes.aerial.__main__ import main

    with pytest.raises(SystemExit):
        main(["prepare", str(root)])
    assert "--roi" in capsys.readouterr().err
