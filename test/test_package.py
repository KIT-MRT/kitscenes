"""Tests for the kitscenes package surface: version, typing marker, exports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import kitscenes


def test_package_ships_pep561_marker() -> None:
    """Without py.typed, type checkers discard every annotation downstream."""
    assert (Path(kitscenes.__file__).parent / "py.typed").is_file()


def test_every_name_in_all_is_defined() -> None:
    for name in kitscenes.__all__:
        assert hasattr(kitscenes, name), f"kitscenes.{name} is in __all__ but undefined"


def test_star_import_keeps_the_builtin_open() -> None:
    assert "open" not in kitscenes.__all__ and callable(kitscenes.open)


@pytest.mark.parametrize("modules, absent", [
    ("kitscenes, kitscenes.map", ("lanelet2",)),
    ("kitscenes, kitscenes.visualization", ("matplotlib", "cv2", "cairosvg")),
    ("kitscenes, kitscenes.visualization, kitscenes.visualization.gpu_video", ("torch",)),
])
def test_imports_load_nothing_optional(modules: str, absent: tuple) -> None:
    """Optional packages that are slow to import load on use, not with the package."""
    code = f"import sys, {modules}; print(sorted(m for m in {absent!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_missing_viz_dependency_names_the_extra(monkeypatch) -> None:
    import numpy as np
    from kitscenes.errors import MissingDependencyError
    from kitscenes.map import MissingDependencyError as MapMissingDependencyError
    from kitscenes.visualization import camera_grid, style

    assert MapMissingDependencyError is MissingDependencyError
    for name in ("matplotlib", "matplotlib.colors", "cv2"):
        monkeypatch.setitem(sys.modules, name, None)  # import fails as if not installed
    with pytest.raises(MissingDependencyError, match=r'matplotlib .*pip install "kitscenes\[viz\]"'):
        style.colour("Dashed")
    with pytest.raises(ImportError, match=r'opencv-python-headless .*pip install "kitscenes\[viz\]"'):
        camera_grid({"cam_front_center": np.zeros((4, 4, 3), np.uint8)})


def test_missing_overloads() -> None:
    """Every accessor with ``missing=`` has overloads: Optional only for ``missing="none"``."""
    import inspect
    import typing

    if not hasattr(typing, "get_overloads"):
        pytest.skip("typing.get_overloads needs Python 3.11")
    methods = [m for cls in (kitscenes.Frame, kitscenes.Scene) for _, m in inspect.getmembers(cls, inspect.isfunction)
               if not m.__name__.startswith("_") and "missing" in inspect.signature(m).parameters]
    assert {m.__name__ for m in methods} >= {"image", "lidar", "fused_lidar", "future", "past", "aerial", "messages", "imu"}
    for method in methods:
        overloads = typing.get_overloads(method)
        returns = {o.__annotations__["missing"].replace('"', "'"): o.__annotations__["return"] for o in overloads}
        assert set(returns) == {"Literal['raise']", "Literal['none']"}, method.__qualname__
        raised = returns["Literal['raise']"]
        assert returns["Literal['none']"] == f"Optional[{raised}]", method.__qualname__


def test_reprs_of_array_holders_stay_short() -> None:
    """print() of poses, trajectories, map elements and labels is one short line, not every array element."""
    import numpy as np

    from kitscenes.ego_trajectory import EgoTrajectory
    from kitscenes.map.elements import Lane, LaneKind
    from kitscenes.map.scene_map import MapPatch
    from kitscenes.schema import EgoPose, Provenance

    pose = EgoPose(5, xyz=[1.0, 2.0, 3.0], quat=[0.0, 0.0, 0.0, 1.0])
    assert repr(pose) == "EgoPose(t_ns=5, xyz=[1., 2., 3.], quat=[0., 0., 0., 1.])"
    k = 300
    traj = EgoTrajectory(index=0, anchor_t_ns=0, t_ns=np.arange(k), xyz_ego=np.zeros((k, 3)), quat_ego=np.zeros((k, 4)),
                         xyz_world=np.zeros((k, 3)), valid=np.ones(k, bool), provenance=Provenance.GEOMETRY_DERIVED)
    assert "xyz_ego=<float64 (300, 3)>" in repr(traj) and len(repr(traj)) < 300
    lane = Lane(id=1, kind=LaneKind.ROAD, left=2, right=3, left_points=np.zeros((50, 3)), right_points=np.zeros((50, 3)),
                centerline=np.zeros((50, 3)), speed_limit_mps=None, vehicle=True, one_way=True, successors=(), predecessors=(),
                left_neighbor=None, right_neighbor=None, regulatory_ids=())
    assert "kind='road'" in repr(lane) and "left_points=<float64 (50, 3)>" in repr(lane)
    patch = MapPatch(lanes=[lane] * 40, lines=[], stop_lines=[], traffic_signs=[], traffic_lights=[], frame="ego")
    assert repr(patch).startswith("MapPatch(lanes=[40 Lane], lines=[]") and len(repr(patch)) < 150


def test_torch_features_name_the_minimum_torch(monkeypatch) -> None:
    import types

    from kitscenes._optional import TORCH_MIN_VERSION, require_torch
    from kitscenes.errors import MissingDependencyError

    old = types.ModuleType("torch")
    old.__version__ = "1.11.0+cu102"
    monkeypatch.setitem(sys.modules, "torch", old)
    with pytest.raises(MissingDependencyError, match=f"torch>={TORCH_MIN_VERSION}"):
        require_torch("kitscenes.mapping.VoxelMap")
    assert require_torch("kitscenes.decode", "1.10") is old  # a feature may name a lower minimum
    old.__version__ = "1.13.1"
    assert require_torch("x") is old
