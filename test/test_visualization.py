"""kitscenes.visualization: top-down view, reprojection into the cameras, video, output guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR  # noqa: E402
from kitscenes.visualization import camera_grid, write_video  # noqa: E402
from kitscenes.visualization import style  # noqa: E402
from kitscenes.visualization._deps import guard_output_path  # noqa: E402
from kitscenes.visualization.reprojection import draw_map, draw_points  # noqa: E402
from map_fixture import ORIGIN, ego_pose, write_osm  # noqa: E402
from release_fixture import real_camera  # noqa: E402  (focal 1000, 2000 x 1200, 1.6 m up, looking along +x)

# ---------------------------------------------------------------------------
# Without the converter
# ---------------------------------------------------------------------------


def test_output_is_never_written_into_the_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "dataset").mkdir()
    monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, str(tmp_path / "dataset"))
    assert guard_output_path(tmp_path / "out") == (tmp_path / "out").resolve()
    with pytest.raises(ValueError, match="inside the dataset root"):
        guard_output_path(tmp_path / "dataset" / "scene" / "proj")
    with pytest.raises(ValueError, match="inside the dataset root"):  # the dataset read, wherever it is
        guard_output_path(tmp_path / "other" / "x.mp4", roots=[tmp_path / "other"])


def test_points_nearest_on_top() -> None:
    camera = real_camera()
    image = np.zeros((1200, 2000, 3), np.uint8)
    wall = np.column_stack([np.full(400, 20.0), np.repeat(np.linspace(-1, 1, 20), 20), np.tile(np.linspace(0, 2, 20), 20)])
    near = np.array([[10.0, 0.0, 1.0]])
    colours = np.vstack([np.tile([0, 0, 255], (len(wall), 1)), [[255, 0, 0]]]).astype(np.uint8)
    draw_points(image, camera, np.vstack([wall, near]), colours, radius=2)
    u, v = 1000, round(600 + 1000 * 0.6 / 10)  # the near point: 10 m ahead, 0.6 m below the camera
    assert image[v, u].tolist() == [255, 0, 0]
    assert (image[..., 2] == 255).sum() > 100  # the wall around it


def test_grid_layout_and_icons() -> None:
    from kitscenes.visualization import GRID_CAMERAS

    images = {name: np.full((40, 60, 3), i * 40, np.uint8) for i, name in enumerate(GRID_CAMERAS)}
    grid = camera_grid(images, resize=1.0)
    assert grid.shape == (80, 180, 3)
    assert grid[0, 0, 0] == 0 and grid[0, 60, 0] == 40 and grid[40, 0, 0] == 120  # front row, then rear row
    assert style.icon_path("TSStop").name == "206.png"
    assert style.icon_rgba("TSStop").shape[2] == 4


def test_crosswalk_outline_is_a_closed_hull() -> None:
    bow_tie = np.array([[0, 0, 0], [4, 3, 0], [4, 0, 0], [0, 3, 0]], float)
    ring = style.crossing_outline(bow_tie)
    assert len(ring) == 5 and np.allclose(ring[0], ring[-1])


def test_video(tmp_path: Path) -> None:
    for i in range(3):
        for folder, shape in (("grid", (120, 180, 3)), ("top_down", (200, 100, 3))):
            (tmp_path / "r" / folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_path / "r" / folder / f"{i:010d}.jpg"), np.full(shape, 40 * i, np.uint8))
    out = write_video(tmp_path / "r", tmp_path / "out.mp4", fps=2)
    assert out.is_file() and out.stat().st_size > 0
    with pytest.raises(ValueError, match="no rendered frames"):
        write_video(tmp_path / "empty", tmp_path / "x.mp4")


# ---------------------------------------------------------------------------
# With the converter: the synthetic map of test_map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import MLConverter, Region, SceneMap
    smap = SceneMap(write_osm(tmp_path_factory.mktemp("map") / "map.osm"), ORIGIN)
    conv = MLConverter(smap, region=Region(40, 10, 15, 15), sampling=None, te_sampling=None)
    return conv(ego_pose(1.75, 10.0, np.pi / 2, z=100.0))  # base frame on the road (map at 100 m)


def test_map_drawn_into_the_camera(labels) -> None:
    from kitscenes.projection import project_points

    camera = real_camera()
    image = np.zeros((1200, 2000, 3), np.uint8)
    draw_map(image, camera, labels)
    stop = labels.traffic_elements["StopLine"][0]  # 8 m ahead, across the lane
    uv, _, visible = project_points(camera, stop.mean(axis=0, keepdims=True))
    u, v = np.floor(uv[0] + 0.5).astype(int)
    assert visible[0] and image[v, u].tolist() == [255, 0, 0]  # red stop line
    assert 0.005 < image.any(axis=2).mean() < 0.3  # lines drawn, nothing folded across the frame


def test_top_down(labels) -> None:
    from kitscenes.visualization import render_top_down

    image = render_top_down(labels, extent=(40.0, 15.0), show_legend=True)
    assert image.ndim == 3 and image.shape[0] > image.shape[1]  # vertical by default
    assert (np.abs(image.astype(int) - [255, 0, 0]).sum(axis=2) < 60).any()  # the red stop line


def test_top_down_leaves_out_signs_without_a_type(labels) -> None:
    from kitscenes.visualization import render_top_down

    assert labels.untyped_sign_ids == {11}  # the sign with subtype "unknown"; the typed sign is not in it
    options = dict(show_edges=False, show_legend=False)
    default = render_top_down(labels, **options)
    assert np.array_equal(default, render_top_down(labels, exclude_te_ids=[11], **options))
    assert not np.array_equal(default, render_top_down(labels, show_untyped_signs=True, **options))

