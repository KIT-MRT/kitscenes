"""The video renderer (kitscenes.visualization.gpu_video): options, layout, drawing primitives, a CPU render."""

from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kitscenes.visualization import gpu_video as gv


# ---------------------------------------------------------------------------
# Options: no torch needed
# ---------------------------------------------------------------------------


def test_resolution_presets_and_custom() -> None:
    assert gv.parse_resolution("1080p") == (1920, 1080)
    assert gv.parse_resolution("2K") == (2560, 1440)
    assert gv.parse_resolution("4k") == (3840, 2160)
    assert gv.parse_resolution("3000x1500") == (3000, 1500)
    assert gv.parse_resolution((640, 360)) == (640, 360)
    for bad in ("1081x720", "100x100", "big", "1920x1080x3"):
        with pytest.raises(ValueError):
            gv.parse_resolution(bad)


def test_speed_gives_the_frame_rate_and_drops_frames_above_the_maximum() -> None:
    assert gv.playback(1) == gv.Playback(1, Fraction(10), 1)
    assert gv.playback(2).fps == 20
    assert gv.playback(0.5).fps == 5
    assert gv.playback(6) == gv.Playback(1, Fraction(60), 1)  # at the maximum: no drop
    assert gv.playback(8) == gv.Playback(2, Fraction(40), 2)  # every 2nd frame at 40 fps: still 8x
    assert gv.playback(20) == gv.Playback(4, Fraction(50), 4)
    p = gv.playback(1, frame_step=3)  # every 3rd frame stays real time
    assert p.step == 3 and p.fps == Fraction(10, 3)
    p = gv.playback(2, max_fps=15)
    assert p.step * float(p.fps) / gv.SOURCE_FPS == pytest.approx(2) and p.fps <= 15
    for kwargs in ({"speed": 0}, {"speed": 1, "frame_step": 0}, {"speed": 1, "max_fps": 0}):
        with pytest.raises(ValueError):
            gv.playback(**kwargs)


def test_region_bitrate_frames() -> None:
    r = gv.parse_region(None)
    assert (r.ahead, r.behind, r.left, r.right) == (60, 60, 30, 30)
    r = gv.parse_region("80x40")
    assert (r.ahead + r.behind, r.left + r.right) == (80, 40)
    with pytest.raises(ValueError):
        gv.parse_region("80")
    assert gv.max_bitrate((3840, 2160), 10) == 80_000_000
    assert gv.max_bitrate((1920, 1080), 10) == 20_000_000
    assert gv.max_bitrate((1920, 1080), 10, "h264") == 30_000_000
    assert gv.max_bitrate((1920, 1080), 40) == 40_000_000
    assert gv.parse_bitrate("20M") == 20_000_000 and gv.parse_bitrate("800k") == 800_000
    assert gv.frame_indices(10) == list(range(10))
    assert gv.frame_indices(10, slice(2, 8), 2) == [2, 4, 6]
    assert gv.frame_indices(10, gv.parse_frames("5:"), 1) == [5, 6, 7, 8, 9]
    assert gv.parse_frames("3") == slice(3, 4)


def test_decode_reduction_keeps_the_tile_size() -> None:
    assert gv.decode_reduction((3504, 2272), (1280, 830)) == 2   # 4K tile
    assert gv.decode_reduction((3504, 2272), (640, 415)) == 4    # 1080p tile
    assert gv.decode_reduction((3504, 2272), (400, 260)) == 8
    assert gv.decode_reduction((3504, 2272), (2000, 1300)) == 1


def _no_overlap(rects) -> None:
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert a.x1 <= b.x or b.x1 <= a.x or a.y1 <= b.y or b.y1 <= a.y, (a, b)


def _window_aspect(window) -> float:
    x0, y0, x1, y1 = window
    return (x1 - x0) * gv.CAMERA_ASPECT / (y1 - y0)


def test_layout_fits_and_does_not_overlap() -> None:
    for preset in ("1080p", "2k", "4k", "1280x720"):
        for layout in gv.LAYOUTS:
            for legend in gv.LEGENDS:
                _check_layout(gv.parse_resolution(preset), layout, legend)


def _check_layout(size: tuple, layout: str, legend: str) -> None:
    L = gv.Layout.compute(size, layout, legend=legend)
    rects = [L.header, *L.tiles.values()] + [r for r in (L.legend, L.bev) if r is not None]
    W, H = size
    for r in rects:
        assert 0 <= r.x and 0 <= r.y and r.x1 <= W and r.y1 <= H and r.w > 0 and r.h > 0
    _no_overlap(rects)
    front, rear = L.tiles["camera_ring_front"], L.tiles["camera_ring_rear"]
    assert front.y < rear.y and front.x == rear.x  # front row on top
    tw, th = L.tile_size
    for name, tile in L.tiles.items():  # cover: the window has the tile's shape, cut at the sides
        assert (tile.w, tile.h) == (tw, th)
        assert _window_aspect(L.windows[name]) == pytest.approx(tw / th, rel=1e-6)
        x0, y0, x1, y1 = L.windows[name]
        assert y0 == 0 and y1 == 1 and x0 == pytest.approx(1 - x1) and 0 < x0 < 0.5
    assert (L.legend is None) == (legend == "none")
    if layout == "cams+bev":
        col = L.bev.w
        assert L.bev.x1 == W and L.bev.y == L.header.h and abs(L.bev.w / L.bev.h - 0.5) < 0.01  # 120 x 60 m, driving up
        assert col == pytest.approx(W * gv.BEV_COLUMN, abs=2)
        if L.legend is not None:  # the legend below the BEV, in its column, to the bottom
            assert (L.legend.x, L.legend.w, L.legend.y, L.legend.y1) == (L.bev.x, col, L.bev.y1, H)
        grid_right = W - col
    else:
        assert L.bev is None
        grid_right = W
        if L.legend is not None:  # a slim strip at the bottom
            assert (L.legend.x, L.legend.w, L.legend.y1) == (0, W, H) and L.legend.h < 0.15 * H
    bottom = L.legend.y if L.legend is not None and layout == "cams" else H
    # the grid fills everything left of the column (or above the strip), up to rounding
    assert min(t.x for t in L.tiles.values()) <= 1 and grid_right - max(t.x1 for t in L.tiles.values()) <= 2
    assert min(t.y for t in L.tiles.values()) == L.header.h and bottom - max(t.y1 for t in L.tiles.values()) <= 1


def test_layout_contain_and_crop_hood() -> None:
    L = gv.Layout.compute((1920, 1080), "cams+bev", fit="contain")
    for name, tile in L.tiles.items():  # the whole image, black bars
        assert L.windows[name] == (0.0, 0.0, 1.0, 1.0)
        assert tile.w / tile.h == pytest.approx(gv.CAMERA_ASPECT, abs=0.01)
    _no_overlap(list(L.tiles.values()))
    hood = gv.parse_crop_hood(True)
    assert hood == gv.HOOD_FRACTIONS
    cover = gv.Layout.compute((1920, 1080), "cams+bev")
    cut = gv.Layout.compute((1920, 1080), "cams+bev", crop_hood=True)
    for name in gv.GRID_CAMERAS:
        x0, y0, x1, y1 = cut.windows[name]
        assert y0 == 0 and y1 == pytest.approx(1 - hood[name])  # the bottom is cut away ...
        assert x1 - x0 < cover.windows[name][2] - cover.windows[name][0]  # ... so the rest is shown larger
        assert _window_aspect(cut.windows[name]) == pytest.approx(cut.tile_size[0] / cut.tile_size[1])
    contain = gv.Layout.compute((1920, 1080), "cams+bev", fit="contain", crop_hood=0.1)
    assert all(w == (0.0, 0.0, 1.0, 0.9) for w in contain.windows.values())
    assert gv.parse_crop_hood(None) == {} and gv.parse_crop_hood(0.0) == {}
    assert gv.parse_crop_hood({"camera_ring_front": 0.2}) == {"camera_ring_front": 0.2}
    for bad in (0.6, {"camera_lidar": 0.1}, "lots"):
        with pytest.raises(ValueError):
            gv.parse_crop_hood(bad)
    for kwargs in ({"legend": "big"}, {"fit": "stretch"}):
        with pytest.raises(ValueError):
            gv.Layout.compute((1920, 1080), "cams+bev", **kwargs)


def test_crop_window_and_tile_camera() -> None:
    from release_fixture import camera_entry

    from kitscenes.rig import Camera

    assert gv.crop_window((3504, 2272), (0.181, 0.0, 0.819, 0.85)) == (632, 0, 2872, 1928)
    assert all(v % 8 == 0 for v in gv.crop_window((3504, 2272), (0.3, 0.1, 0.71, 0.77)))
    cam = Camera.from_calib_entry("camera_ring_front", camera_entry(3504, 2272, focal=2000.0))
    points = np.array([[20.0, 3.0, 1.0], [8.0, -2.0, -1.5], [40.0, 0.5, 0.0]])
    uv, _, _ = cam.project(points)

    class Scene:
        class rig:
            cameras = {"camera_ring_front": cam}

    L = gv.Layout.compute((1920, 1080), "cams+bev", crop_hood=True)
    cams, windows, scale = gv.tile_cameras(Scene, L)
    x0, y0, x1, y1 = windows["camera_ring_front"]
    tile = L.tiles["camera_ring_front"]
    sx, sy = tile.w / (x1 - x0), tile.h / (y1 - y0)
    assert scale == pytest.approx(sx)
    expect = np.column_stack([(uv[:, 0] - x0 + 0.5) * sx - 0.5, (uv[:, 1] - y0 + 0.5) * sy - 0.5])
    got, _, _ = cams["camera_ring_front"].project(points)
    assert np.allclose(got, expect, atol=1e-6)  # the tile camera sees the window, scaled to the tile


def test_rear_row_order_and_mirrored_projection() -> None:
    from release_fixture import camera_entry

    from kitscenes.rig import Camera

    def row(L, y):
        return [n for n, t in sorted(L.tiles.items(), key=lambda kv: kv[1].x) if t.y == y]

    for layout in gv.LAYOUTS:
        swap, plain, mirror = (gv.Layout.compute((1920, 1080), layout, rear_row=r) for r in ("swap", "plain", "mirror"))
        front_y = swap.tiles["camera_ring_front"].y
        rear_y = swap.tiles["camera_ring_rear"].y
        assert row(swap, front_y) == ["camera_ring_front_left", "camera_ring_front", "camera_ring_front_right"]
        assert row(swap, rear_y) == ["camera_ring_rear_right", "camera_ring_rear", "camera_ring_rear_left"]
        assert row(plain, rear_y) == row(mirror, rear_y) == ["camera_ring_rear_left", "camera_ring_rear",
                                                             "camera_ring_rear_right"]
        assert swap.mirrored == plain.mirrored == frozenset()
        assert mirror.mirrored == frozenset(gv.GRID_CAMERAS[3:])
    assert gv.grid_order("swap")[3:] == ("camera_ring_rear_right", "camera_ring_rear", "camera_ring_rear_left")
    with pytest.raises(ValueError):
        gv.Layout.compute((1920, 1080), "cams", rear_row="flip")

    cam = Camera.from_calib_entry("camera_ring_rear", camera_entry(3504, 2272, focal=2000.0))

    class Scene:
        class rig:
            cameras = {"camera_ring_rear": cam}

    points = np.array([[20.0, 3.0, 1.0], [8.0, -2.0, -1.5], [40.0, 0.5, 0.0]])
    plain_cams = gv.tile_cameras(Scene, gv.Layout.compute((1920, 1080), "cams+bev", rear_row="plain"))[0]
    mirror_L = gv.Layout.compute((1920, 1080), "cams+bev", rear_row="mirror")
    mirror_cams = gv.tile_cameras(Scene, mirror_L)[0]
    a, depth_a, _ = plain_cams["camera_ring_rear"].project(points)
    b, depth_b, _ = mirror_cams["camera_ring_rear"].project(points)
    w = mirror_L.tiles["camera_ring_rear"].w
    assert np.allclose(b[:, 0], w - 1 - a[:, 0]) and np.allclose(b[:, 1], a[:, 1]) and np.allclose(depth_a, depth_b)


def test_direction_heads_follow_the_line() -> None:
    from kitscenes.map import Region

    region = Region(60, 60, 30, 30)
    line = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [30.0, 25.0, 0.0]])  # east 30 m, then north 25 m, ends inside
    tips, backs = gv.direction_heads([line], region, spacing=20.0, head=1.6)
    assert np.allclose(tips, [[10.0, 0.0], [30.0, 0.0], [30.0, 25.0]])  # 10 m, 30 m, the end (50 m: too near it)
    d = tips - backs
    assert np.allclose(d, [[1.6, 0.0], [0.0, 1.6], [0.0, 1.6]])  # pointing along the line, as its points run
    tips_rev, backs_rev = gv.direction_heads([line[::-1]], region, spacing=20.0, head=1.6)
    assert np.allclose((tips_rev - backs_rev)[0], [0.0, -1.6])  # reversed line, reversed heads
    cut = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]])  # ends on the region's front edge: no head at the end
    tips, _ = gv.direction_heads([cut], region, spacing=20.0, head=1.6)
    assert np.allclose(tips, [[10.0, 0.0], [30.0, 0.0], [50.0, 0.0]])
    assert gv.direction_heads([np.zeros((1, 3))], region)[0].shape == (0, 2)


def test_forget_drops_the_page_cache_best_effort(tmp_path: Path) -> None:
    from kitscenes._layout import sensor_member
    from kitscenes._storage import DirStorage

    name = sensor_member("camera_ring_front", 3)
    (tmp_path / name).parent.mkdir(parents=True)
    (tmp_path / name).write_bytes(b"\xff\xd8" + bytes(100))
    gv._forget(DirStorage(tmp_path), "camera_ring_front", 3)  # an existing image
    gv._forget(DirStorage(tmp_path), "camera_ring_front", 4)  # a missing one: no error
    assert (tmp_path / name).read_bytes()[:2] == b"\xff\xd8"


def test_compact_legend_lists_what_the_scene_shows() -> None:
    full = gv.legend_entries()
    keys = [k for k, _ in full]
    assert "Centerline" in keys and "TLCar" in keys and keys[-1] == "_edge"
    assert "_edge" not in [k for k, _ in gv.legend_entries(layout="cams")]  # the edges are drawn in the BEV only
    compact = gv.legend_entries(({"Dashed", "Solid", "Unknown"}, {"TLPedestrian", "StopLine", "ArrowTurnLeft"}, False))
    assert compact == [("Dashed", "Dashed"), ("Solid", "Solid"), ("Unknown", "Other line"), ("StopLine", "Stop line"),
                       ("TLCar", "Traffic light"), ("ArrowGoStraight", "Road marking")]
    assert gv.legend_entries(((), (), False)) == []


def test_legend_image_fits_its_area() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    for w, h in ((384, 272), (1920, 59), (60, 30)):
        img = gv._legend_image(w, h, 0.5, gv.legend_entries())
        assert img.shape == (h, w, 3)
    img = gv._legend_image(384, 272, 0.5, gv.legend_entries(({"Dashed"}, (), False)))
    assert (img != 24).any()


def test_scene_labels_cut_like_the_converter() -> None:
    """The per-frame cut of the scene's labels (in the map frame) gives the pieces clip_polyline gives."""
    from kitscenes.map._vector import clip_polyline
    from kitscenes.schema import EgoPose
    from kitscenes.visualization._scene_labels import _Polylines

    zigzag = np.array([[-80.0, 0, 0], [0, 0, 0], [0, 40, 0], [10, 40, 0], [10, -10, 0], [100, -10, 0]])
    inside = np.array([[1.0, 1, 0], [2, 2, 0]])
    lines = _Polylines.of({"A": [zigzag, inside]}, ["A"], {"A": [(1,), (2,)]}, lambda p: p + (500.0, 200.0, 0.0))
    pose = EgoPose(0, np.array([500.0, 200.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  # at the offset, no rotation
    R = np.eye(3)
    got = lines.pieces(R, np.asarray(pose.xyz), (0.0, 0.0, 60.0, 30.0), 0.1)
    expect = clip_polyline(zigzag, 60.0, 30.0)
    assert [k for k, _ in got] == [0] * len(expect) + [1]
    for (_, a), b in zip(got, expect):
        assert np.allclose(a, b)
    assert np.allclose(got[-1][1], inside)
    assert lines.present([(R, np.asarray(pose.xyz))], (0.0, 0.0, 60.0, 30.0)).tolist() == [True, True]
    far = np.asarray(pose.xyz) + (1000.0, 0, 0)
    assert lines.present([(R, far)], (0.0, 0.0, 60.0, 30.0)).tolist() == [False, False]


def test_scene_order(tmp_path: Path) -> None:
    from release_fixture import make_dataset

    import kitscenes
    from kitscenes.splits import scene_table

    known = sorted((i for i in scene_table().values() if i.recording_order and i.split == "val"),
                   key=lambda i: i.recording_order)[:2]
    ids = {"val": [known[1].scene_id, known[0].scene_id], "test-e2e": ["zz-unknown", "aa-unknown"]}
    ds = kitscenes.open(make_dataset(tmp_path / "root", ids, n=2, lidars=(), radars=()))
    order = gv.select_scenes(ds)
    assert order == [known[0].scene_id, known[1].scene_id, "aa-unknown", "zz-unknown"]
    assert gv.select_scenes(ds, splits="test-e2e") == ["aa-unknown", "zz-unknown"]
    assert gv.select_scenes(ds, max_scenes=1) == [known[0].scene_id]
    assert gv.select_scenes(ds, order="id")[:2] == sorted(ids["val"])


def test_cli_dry_run(tmp_path: Path, capsys) -> None:
    from release_fixture import make_dataset

    from kitscenes.cli import main

    root = make_dataset(tmp_path / "root", {"val": ["s1", "s2"]}, n=5, lidars=(), radars=())
    assert main(["video", str(root), "--out", str(tmp_path / "v.mp4"), "--speed", "8", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "6 frames" in out and "40 fps" in out
    assert main(["video", str(root), "--out", str(root / "v.mp4"), "--no-map", "--device", "cpu"]) == 1
    assert "inside the dataset root" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Drawing primitives (torch, on the CPU; the same code runs on CUDA)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
gd = pytest.importorskip("kitscenes.visualization._gpu_draw")


def _segments(p0, p1, *, colour=(255, 0, 0), radius=1.0, alpha=1.0, priority=0, bounds=(0, 0, 99, 99), group=None,
              dash=0.0, gap=0.0, depth=25.0, depth_scaled=False):
    p0 = torch.tensor(p0, dtype=torch.float32)
    p1 = torch.tensor(p1, dtype=torch.float32)
    n = len(p0)

    def full(v, dtype=torch.float32):
        t = torch.as_tensor(v, dtype=dtype)
        return t.expand(n).clone() if t.ndim == 0 else t

    return gd.Segments(p0=p0, p1=p1, colour=torch.tensor([colour] * n, dtype=torch.uint8), radius=full(radius),
                       alpha=full(alpha), priority=full(priority, torch.int64),
                       bounds=torch.tensor([bounds] * n, dtype=torch.float32),
                       group=full(group if group is not None else list(range(n)), torch.int64),
                       dash=full(dash), gap=full(gap), min_dash=full(0.0), min_gap=full(0.0),
                       depth_scaled=full(depth_scaled, torch.bool), d0=full(depth), d1=full(depth))


def test_line_is_drawn_where_it_is() -> None:
    img = torch.zeros((3, 100, 100), dtype=torch.uint8)
    gd.draw_segments(img, _segments([[10, 50]], [[90, 50]], radius=1.0))
    red = img[0] == 255
    assert red[50, 10:91].all() and red[49, 20] and red[51, 20]
    assert not red[47].any() and not red[50, :9].any() and not red[50, 92:].any()


def test_layering_by_priority_not_by_order() -> None:
    img = torch.zeros((3, 100, 100), dtype=torch.uint8)
    seg = gd.Segments.cat([_segments([[50, 10]], [[50, 90]], colour=(0, 255, 0), priority=5),
                           _segments([[10, 50]], [[90, 50]], colour=(255, 0, 0), priority=1)])
    gd.draw_segments(img, seg)
    assert tuple(img[:, 50, 50].tolist()) == (0, 255, 0)  # the crossing: the higher priority wins
    assert tuple(img[:, 50, 20].tolist()) == (255, 0, 0)


def test_lines_stay_in_their_rectangle_and_blend() -> None:
    img = torch.full((3, 100, 100), 100, dtype=torch.uint8)
    gd.draw_segments(img, _segments([[-50, 50]], [[150, 50]], colour=(200, 200, 200), bounds=(20, 0, 59, 99), alpha=0.5))
    assert (img[0, 50, 20:60] == 150).all()
    assert (img[0, 50, :20] == 100).all() and (img[0, 50, 60:] == 100).all()


def test_dashes_run_on_along_the_polyline_and_shrink_with_depth() -> None:
    def dashes(depth, scaled):
        img = torch.zeros((3, 20, 1000), dtype=torch.uint8)
        xs = np.linspace(0, 999, 6)
        seg = _segments([[x, 10] for x in xs[:-1]], [[x, 10] for x in xs[1:]], radius=0.5, dash=40, gap=40,
                        group=[0] * 5, depth=depth, depth_scaled=scaled, bounds=(0, 0, 999, 19))
        gd.draw_segments(img, seg)
        on = (img[0, 10] > 0).numpy().astype(int)
        starts = np.flatnonzero(np.diff(on) == 1) + 1
        return on, starts

    on, starts = dashes(25.0, False)  # factor 1: 40 on, 40 off, from the start of the polyline
    assert on[:35].all() and not on[45:75].any() and np.allclose(np.diff(starts), 80, atol=2)
    _, near = dashes(10.0, True)  # nearer: longer dashes (factor 1.75 at most)
    _, far = dashes(80.0, True)  # far: shorter (factor 0.45 at least)
    assert np.median(np.diff(near)) == pytest.approx(140, abs=3)
    assert np.median(np.diff(far)) == pytest.approx(36, abs=3)


def test_projection_clips_at_the_near_plane() -> None:
    from release_fixture import camera_entry

    from kitscenes.rig import Camera

    cam = Camera.from_calib_entry("camera_ring_front", camera_entry(640, 480, focal=500.0))
    M = np.eye(4)
    M[:3] = cam.K @ cam.T_sensor_base[:3]
    P0 = torch.tensor([[10.0, 0.0, 1.6], [-5.0, 0.0, 1.6], [-5.0, 0.0, 1.6]])
    P1 = torch.tensor([[20.0, 1.0, 1.6], [10.0, 0.0, 1.6], [-1.0, 0.0, 1.6]])
    p0, p1, d0, d1, keep = gd.project_segments(P0, P1, torch.tensor(M[None], dtype=torch.float32), near=0.1)
    uv, depth, _ = cam.project(np.array([[10.0, 0.0, 1.6], [20.0, 1.0, 1.6]]))
    assert np.allclose(p0[0, 0].numpy(), uv[0], atol=1e-3) and np.allclose(p1[0, 0].numpy(), uv[1], atol=1e-3)
    assert keep[0].tolist() == [True, True, False]  # the third lies behind the camera
    assert d0[0, 1] == pytest.approx(0.1, abs=1e-4)  # cut at the near plane, not wrapped around
    assert abs(float(p0[0, 1, 1]) - 240) < 1e-3  # on the horizon row, as the line


def _nv12_to_rgb(nv12: np.ndarray) -> np.ndarray:
    """NV12 (H * 3 / 2, W) uint8 -> (H, W, 3) uint8 RGB, BT.709 limited range."""
    h = nv12.shape[0] * 2 // 3
    Y = (nv12[:h].astype(np.float64) - 16) / 219
    uv = nv12[h:].reshape(h // 2, -1, 2).astype(np.float64)
    U = np.repeat(np.repeat((uv[..., 0] - 128) / 224, 2, 0), 2, 1)
    V = np.repeat(np.repeat((uv[..., 1] - 128) / 224, 2, 0), 2, 1)
    r = Y + 1.5748 * V
    b = Y + 1.8556 * U
    g = (Y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.stack([r, g, b], -1) * 255 + 0.5, 0, 255).astype(np.uint8)


def test_nv12_round_trip_and_icons() -> None:
    rng = np.random.default_rng(0)
    img = np.repeat(np.repeat(rng.integers(30, 220, (3, 8, 8)), 2, 1), 2, 2).astype(np.uint8)
    nv12 = gd.rgb_to_nv12(torch.from_numpy(img))
    assert nv12.shape == (24, 16)
    back = _nv12_to_rgb(nv12.numpy()).transpose(2, 0, 1)
    assert np.abs(back.astype(int) - img).max() <= 3
    canvas = torch.zeros((3, 20, 20), dtype=torch.uint8)
    icon = torch.zeros((4, 4, 4), dtype=torch.uint8)
    icon[0], icon[3] = 255, 255
    gd.paste_rgba(canvas, icon, 18, 5, 0.5, rect=(0, 0, 19, 20))  # cut at the rectangle
    assert (canvas[0, 5:9, 18] == 128).all() and (canvas[0, :, 19] == 0).all()


# ---------------------------------------------------------------------------
# End to end on the CPU: a tiny synthetic dataset, software encoder
# ---------------------------------------------------------------------------


def test_rate_control_arguments() -> None:
    rc = gv.RateControl(quality=19, max_rate=20_000_000)
    args = rc.args("hevc_nvenc", gpu=1)
    assert args[args.index("-cq") + 1] == "19" and args[args.index("-b:v") + 1] == "0"
    assert args[args.index("-maxrate") + 1] == "20000000" and args[args.index("-gpu") + 1] == "1"
    assert "-spatial-aq" in args and "-tune" in args
    assert "-gpu" not in rc.args("hevc_nvenc")
    sw = rc.args("libx265")
    assert sw[sw.index("-crf") + 1] == "19" and "-gpu" not in sw and "-maxrate" in sw
    avg = gv.RateControl(bitrate=8_000_000).args("h264_nvenc")
    assert "-cq" not in avg and avg[avg.index("-b:v") + 1] == "8000000" and avg[avg.index("-maxrate") + 1] == "16000000"
    basic = gv.RateControl(quality=19, max_rate=1, basic=True).args("hevc_nvenc")
    assert "-tune" not in basic and basic[basic.index("-preset") + 1] == "slow"
    assert "cq 19" in rc.describe("hevc_nvenc") and "crf 21" in gv.RateControl(quality=21).describe("libx265")
    rc = gv.rate_control("libx264", (1920, 1080), 10, "h264")
    assert rc.quality == gv.DEFAULT_QUALITY["libx264"] and rc.max_rate == 30_000_000
    assert gv.rate_control("libx264", (1920, 1080), 10, "h264", bitrate="5M").bitrate == 5_000_000
    with pytest.raises(ValueError):
        gv.rate_control("libx264", (1920, 1080), 10, "h264", quality=20, bitrate="5M")
    with pytest.raises(ValueError):
        gv.rate_control("libx264", (1920, 1080), 10, "h264", quality=60)


def _ffmpeg_can(encoder: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    return gv.probe_encoder(encoder) is None


@pytest.mark.skipif(not _ffmpeg_can("libx264"), reason="needs ffmpeg with libx264")
def test_render_on_the_cpu(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from release_fixture import make_dataset

    import kitscenes
    from kitscenes.map import has_ml_converter
    from kitscenes.visualization import GRID_CAMERAS

    root = make_dataset(tmp_path / "root", {"val": ["a", "b"]}, n=4, cameras=GRID_CAMERAS, lidars=(), radars=(),
                        maps=has_ml_converter())
    ds = kitscenes.open(root)
    out = tmp_path / "out" / "video.mp4"
    result = gv.render_video(ds, out, resolution="640x360", device="cpu", codec="h264", encoder="software", workers=0,
                             speed=2, per_scene=True, map_labels=has_ml_converter(), progress=False)
    assert result.path == out.resolve() and out.is_file()
    assert result.frames == 8 and result.fps == 20 and result.duration_s == pytest.approx(0.4)
    assert len(result.scene_videos) == 2 and all(p.is_file() for p in result.scene_videos)
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == 8 and frames[0].shape == (360, 640, 3)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20)
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".")]  # no temporary parts left


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_draw_on_cuda_matches_the_cpu() -> None:
    seg = gd.Segments.cat([_segments([[10, 50]], [[90, 52]], colour=(0, 255, 0), priority=5, dash=10, gap=6),
                           _segments([[50, 10]], [[52, 90]], colour=(255, 0, 0), priority=1, alpha=0.5)])
    a = gd.draw_segments(torch.zeros((3, 100, 100), dtype=torch.uint8), seg)
    cuda = gd.Segments(**{f: getattr(seg, f).cuda() for f in gd.Segments.FIELDS})
    b = gd.draw_segments(torch.zeros((3, 100, 100), dtype=torch.uint8, device="cuda"), cuda)
    assert (a == b.cpu()).float().mean() > 0.999
