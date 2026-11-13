"""Label recipes (kitscenes.map.LABEL_RECIPES): settings, the lanelet2 build check, the IMU refusal, exact_cut."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kitscenes
from kitscenes.map import LABEL_RECIPES, LANELET2_BUILD, LANELET2_PAPER_BUILD, MLConverter, check_label_recipe
from kitscenes.map import _lanelet2

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PAPER = "paper-v2.0.0"


def test_recipes() -> None:
    assert LABEL_RECIPES["default"]["lanelet2"] is None and LABEL_RECIPES[PAPER]["lanelet2"] == LANELET2_PAPER_BUILD
    with pytest.raises(ValueError, match="unknown label recipe"):
        check_label_recipe("paper")
    with pytest.raises(ValueError, match="unknown label recipe"):
        kitscenes.MapLabelSpec(recipe="paper")
    assert kitscenes.MapLabelSpec().recipe == "default" and kitscenes.MapLabelSpec.maptr(recipe=PAPER).recipe == PAPER
    settings = check_label_recipe("default")
    settings["exact_cut"] = False  # a copy
    assert LABEL_RECIPES["default"]["exact_cut"] is True


def test_pyproject_pins_the_build() -> None:
    text = PYPROJECT.read_text()
    pins = re.findall(r'"lanelet2 @ (\S+) ; python_version == \'(3\.\d+)\'', text)
    assert {version for _, version in pins} == {"3.9", "3.10", "3.11", "3.12", "3.13"}
    assert set(re.findall(r"lanelet2-([0-9.]+%2Bkitscenes\d+)-cp3", text)) == {LANELET2_BUILD.replace("+", "%2B")}
    assert all("/download/" + LANELET2_BUILD.replace("+", "-") + "/" in url for url, _ in pins)


@pytest.mark.parametrize("found", ["1.2.3+kitscenes2", "1.2.3+kitscenes4", "1.2.3", None])
def test_paper_recipe_needs_kitscenes3(monkeypatch: pytest.MonkeyPatch, found) -> None:
    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: found)
    with pytest.raises(kitscenes.MissingDependencyError) as err:
        check_label_recipe(PAPER)
    message = str(err.value)
    assert "1.2.3+kitscenes3 exactly" in message and ("no lanelet2" if found is None else found) in message
    assert "https://github.com/KITScenes/Lanelet2/releases/tag/1.2.3-kitscenes3" in message
    check_label_recipe("default")  # the default needs no particular build


@pytest.mark.parametrize("found", ["1.2.3+kitscenes3", "1.2.3+kitscenes4", "1.2.3+kitscenes5", None])
def test_default_recipe_takes_any_build(monkeypatch: pytest.MonkeyPatch, found) -> None:
    """The default recipe takes whichever build is installed; the paper recipe kitscenes3 only."""
    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: found)
    assert check_label_recipe("default")["lanelet2"] is None
    if found == LANELET2_PAPER_BUILD:
        assert check_label_recipe(PAPER)["exact_cut"] is False
    else:
        with pytest.raises(kitscenes.MissingDependencyError, match="kitscenes3"):
            check_label_recipe(PAPER)


def test_paper_recipe_refuses_imu_refined_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: "1.2.3+kitscenes3")

    def scene(use_imu, source):
        def unused():
            raise AssertionError("ego_motion is not needed")
        motion = SimpleNamespace(source=source) if use_imu else property(unused)
        return SimpleNamespace(id="s", use_imu=use_imu, ego_motion=motion)

    with pytest.raises(ValueError, match=r"use_imu=False"):
        check_label_recipe(PAPER, scene(True, "imu"))
    assert check_label_recipe(PAPER, scene(True, "poses"))["exact_cut"] is False  # no IMU data: the poses
    assert check_label_recipe(PAPER, scene(False, None))["use_imu"] is False
    assert check_label_recipe("default", scene(True, "imu"))["use_imu"] is True


def _scene_with_map(tmp_path: Path, use_imu: bool):
    from map_fixture import write_osm
    from release_fixture import make_scene, write_poses

    scene_dir = make_scene(tmp_path / "data" / "train" / "s", n=3, maps=True)
    write_osm(scene_dir / "maps" / "map.osm")
    write_poses(scene_dir, 3, speed=1.0, origin=(1.75, 2.0, 102.0))
    return kitscenes.open(tmp_path, use_imu=use_imu)


def test_recipes_in_the_converter_and_the_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: "1.2.3+kitscenes3")
    ds = _scene_with_map(tmp_path, use_imu=False)
    scene = ds[0]
    assert MLConverter.for_scene(scene).exact_cut is True
    assert MLConverter.for_scene(scene, recipe=PAPER).exact_cut is False
    assert MLConverter.for_scene(scene, exact_cut=False).exact_cut is False  # the default recipe may be overridden
    with pytest.raises(ValueError, match="exact_cut"):
        MLConverter.for_scene(scene, recipe=PAPER, exact_cut=True)
    assert kitscenes.MapLabelSpec.maptr(recipe=PAPER).converter(scene).exact_cut is False
    spec = kitscenes.SampleSpec(frame="road", map_labels=kitscenes.MapLabelSpec.maptr(recipe=PAPER))
    assert len(kitscenes.SampleDataset(ds, spec)) == 3
    with pytest.raises(ValueError, match=r"use_imu=False"):  # a dataset with IMU-refined ego motion
        kitscenes.SampleDataset(kitscenes.open(tmp_path), spec)
    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: "1.2.3+kitscenes2")
    with pytest.raises(kitscenes.MissingDependencyError, match="kitscenes3"):
        kitscenes.SampleDataset(ds, spec)


def test_exact_cut_off_keeps_the_converters_cut(tmp_path_factory) -> None:
    """exact_cut=False: the paper's labels, which reach past the box where the frame pitches nose down."""
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import SceneMap
    from map_fixture import ORIGIN, ego_pose, write_osm

    smap = SceneMap(write_osm(tmp_path_factory.mktemp("map") / "map.osm"), ORIGIN)
    pose = ego_pose(1.75, 10.0, np.pi / 2, pitch=0.05)
    reach = {}
    for exact in (True, False):
        labels = MLConverter(smap, grouping="maptr", crosswalks="polygons", sampling=None, exact_cut=exact)(pose)
        reach[exact] = max(m[:, 0].max() for v in labels.lines.values() for m in v)
    assert reach[True] == pytest.approx(30.0, abs=1e-4) and reach[False] > 30.0 + 0.01
