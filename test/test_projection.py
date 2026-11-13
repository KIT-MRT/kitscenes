"""kitscenes.projection: the projection path everything else uses."""

from __future__ import annotations

import numpy as np

from kitscenes.projection import MIN_DEPTH, pixel_indices, project_lines, project_points
from release_fixture import real_camera  # focal 1000, 2000 x 1200, at 1.6 m looking along base +x


def test_a_line_through_the_camera_plane_ends_at_the_near_plane() -> None:
    camera = real_camera()
    # From 5 m behind to 20 m ahead, 2 m to the left: the part behind is cut off in 3D.
    (pieces,) = project_lines(camera, [np.array([[-5.0, 2.0, 0.0], [20.0, 2.0, 0.0]])])
    assert len(pieces) == 1
    uv, depth = pieces[0]
    np.testing.assert_allclose(depth, [MIN_DEPTH, 20.0], atol=1e-9)
    assert uv[0, 0] < -1e4  # far outside on the left, not mirrored to the right
    np.testing.assert_allclose(uv[1], [1000 - 1000 * 2 / 20, 600 + 1000 * 1.6 / 20])


def test_lines_behind_or_beyond_are_dropped_or_split() -> None:
    camera = real_camera()
    behind = np.array([[-9.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    out_and_back = np.array([[10.0, 0.0, 0.0], [-10.0, 1.0, 0.0], [10.0, 2.0, 0.0]])
    far = np.array([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    a, b, c = project_lines(camera, [behind, out_and_back, far], max_depth=50.0)
    assert a == [] and len(b) == 2
    assert np.isclose(c[0][1][-1], 50.0)


def test_points_match_lines_and_pixel_centres() -> None:
    camera = real_camera()
    xyz = np.array([[10.0, 0.5, 1.0], [30.0, -3.0, 0.0]])
    uv, depth, visible = project_points(camera, xyz)
    (piece,), = project_lines(camera, [xyz])
    np.testing.assert_allclose(piece[0], uv)
    assert visible.all()
    # Pixel centres are on integers: u 1.4 is pixel 1, u 1.6 pixel 2; at half size half-pixel 0 covers u -0.5..1.5.
    uv = np.array([[1.4, 0.0], [1.6, 0.0], [-0.4, 0.0]])
    assert pixel_indices(uv, (1200, 2000), camera)[1].tolist() == [1, 2, 0]
    assert pixel_indices(uv, (600, 1000), camera)[1].tolist() == [0, 1, 0]
