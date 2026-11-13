"""Map labels, lidar and radar drawn into the camera images: the full reprojection.

Everything is projected with :mod:`kitscenes.projection` (the same rules as
painting and the lidar maps), so what lines up here lines up everywhere.

- **Map** (:class:`MapLayer`): the ML converter's labels around the frame's
  pose, in the style of the top-down view: line types by colour, markings
  over a grey underlay (solid vs dashed), dashes that shrink with distance,
  crosswalk outlines, red stop lines, road markings as icons laid onto the
  road, signs and lights as boxes with their icon beside them. Lines are
  clipped at the camera in 3D, so nothing folds across the image.
- **Lidar**: the frame's points coloured by distance, nearest on top,
  without the vehicle's body.
- **Radar** (optional): detections as rings coloured by Doppler velocity
  (ego motion removed: static returns are white).

::

    from kitscenes.visualization import MapLayer, render_frame, camera_grid

    layer = MapLayer(scene)
    images = render_frame(frame, map_layer=layer, lidar=True)   # {camera: RGB image}
    grid = camera_grid(images)

For a whole scene on disk (per-camera images, grids, top-down views; then
:func:`kitscenes.visualization.video.write_video`): :func:`render_scene`, as
``examples/render_reprojection.py`` does.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np

from kitscenes.constants import CAMERA_RING_NAMES, EGO_BOX, CameraNameLike, sensor_names
from kitscenes.projection import project_lines, project_points
from kitscenes.visualization import _deps, style

if TYPE_CHECKING:
    from kitscenes.frame import Frame
    from kitscenes.map import MapLabels
    from kitscenes.rig import Camera
    from kitscenes.scene import Scene

#: The ring cameras in grid order: the front row left to right, then the rear row.
GRID_CAMERAS = tuple(sorted(CAMERA_RING_NAMES,
                            key=lambda n: ("rear" in n, not n.endswith("left"), n.endswith("right"))))
#: Map elements beyond this distance from the camera are not drawn (metres along the optical axis).
MAX_DEPTH = 150.0
GREY, RED, YELLOW = (128, 128, 128), (255, 0, 0), (255, 255, 0)


# ---------------------------------------------------------------------------
# Map labels for drawing
# ---------------------------------------------------------------------------


class MapLayer:
    """The map labels drawn for a frame: wide for everything, nearer for centerlines.

    Centerlines of a whole junction 150 m out are a tangle, so they come from
    a second, smaller region. Map vertices are kept as they are (no
    resampling): a straight 3D segment is a straight segment in the image.

    Args:
        scene: The scene (its map; map_change scenes are flagged in the labels).
        region, centerline_region: Labelled regions (default 300 x 150 m and 150 x 75 m).
        grouping: The converter's line type grouping.
    """

    def __init__(self, scene: "Scene", *, region=None, centerline_region=None, grouping: str = "default") -> None:
        from kitscenes.map import MLConverter, Region

        region = region or Region(150, 150, 75, 75)
        centerline_region = centerline_region or Region(75, 75, 37.5, 37.5)
        self.wide = MLConverter.for_scene(scene, region=region, grouping=grouping, sampling=None, te_sampling=None)
        self.near = MLConverter.for_scene(scene, region=centerline_region, grouping=grouping, sampling=None,
                                          te_sampling=None)
        self.hidden_ids = style.unknown_sign_ids(scene.map)

    def __call__(self, pose) -> tuple["MapLabels", "MapLabels"]:
        """``(wide, near)`` labels for *pose*: draw centerlines from ``near``, the rest from ``wide``."""
        return self.wide(pose), self.near(pose)


# ---------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------


def render_frame(
    frame: "Frame",
    *,
    map_layer: Optional[MapLayer] = None,
    labels: Optional[tuple["MapLabels", "MapLabels"]] = None,
    lidar: bool = False,
    radar: bool = False,
    cameras: Sequence[CameraNameLike] = GRID_CAMERAS,
    reduce: int = 1,
) -> dict[str, np.ndarray]:
    """The frame's camera images with the chosen layers drawn in (RGB uint8).

    Args:
        map_layer: Draw the map labels (or pass *labels* computed already).
        lidar: Draw the lidar points (``frame.fused_lidar()``).
        radar: Draw the radar detections.
        cameras: Cameras to render; missing images are skipped.
        reduce: Decode and draw at 1/reduce size.
    """
    if map_layer is not None and labels is None:
        pose = _pose_or_none(frame)
        labels = map_layer(pose) if pose is not None else None  # no pose at this frame: no map labels
    hidden = map_layer.hidden_ids if map_layer is not None else frozenset()
    points = _lidar_points(frame) if lidar else None
    detections = _radar_points(frame) if radar else None
    out = {}
    for name in sensor_names(cameras):
        camera = frame.scene.rig.cameras.get(name)
        image = frame.image(name, reduce=reduce, missing="none") if camera is not None else None
        if image is None:
            continue
        image = np.ascontiguousarray(image)
        camera = camera.scaled(image.shape[1] / camera.image_size[0]) if reduce != 1 else camera
        scale = image.shape[1] / frame.scene.rig.cameras[name].image_size[0]
        if points is not None:
            draw_points(image, camera, *points, radius=max(1, round(1.5 * scale)), alpha=0.7)
        if detections is not None:
            draw_radar(image, camera, *detections, radius=max(3, round(9 * scale)))
        if labels is not None:
            draw_map(image, camera, *labels, scale=scale, exclude_te_ids=hidden)
        out[name] = image
    return out


def camera_grid(images: dict[str, np.ndarray], *, resize: float = 0.5, title: Optional[str] = None) -> np.ndarray:
    """The six ring cameras as a 3 x 2 grid (front row on top, as seen from the car), RGB."""
    cv2 = _deps.cv2()

    shapes = [im.shape for im in images.values()]
    h, w = (shapes[0][0], shapes[0][1]) if shapes else (2272, 3504)
    size = (int(w * resize), int(h * resize))
    tiles = []
    for name in GRID_CAMERAS:
        image = images.get(name)
        if image is None:
            tile = np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(tile, f"no {name}", (20, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREY, 3)
        else:
            tile = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    if title:
        cv2.putText(grid, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
    return grid


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def draw_map(
    image: np.ndarray,
    camera: "Camera",
    wide: "MapLabels",
    near: Optional["MapLabels"] = None,
    *,
    scale: float = 1.0,
    exclude_te_ids: Iterable[int] = (),
) -> None:
    """Draw map labels (ego frame = base frame of the image's pose) into *image*, in place.

    Centerlines come from *near* when given. *scale*: the image's size
    relative to full size (line widths and icons follow it).
    """
    near = near or wide
    lines = {k: v for k, v in wide.lines.items() if k not in style.CENTERLINE_TYPES}
    lines.update({k: v for k, v in near.lines.items() if k in style.CENTERLINE_TYPES})
    for name, items in lines.items():
        closed = name in style.CROSSING_TYPES
        shapes = [style.crossing_outline(line) for line in items] if closed else items
        line_style = style.LINE_STYLES.get(name, style.DEFAULT_LINE_STYLE)
        for pieces in project_lines(camera, shapes, max_depth=MAX_DEPTH):
            for uv, depth in pieces:
                _draw_styled(image, uv, depth, style.colour(name), line_style, scale)
    excluded = frozenset(exclude_te_ids)
    for name, items in wide.traffic_elements.items():
        ids = wide.traffic_element_ids[name]
        keep = [element for element, map_id in zip(items, ids) if map_id not in excluded]
        for element, pieces in zip(keep, project_lines(camera, keep, max_depth=MAX_DEPTH)):
            for uv, depth in pieces:
                _draw_traffic_element(image, name, uv, depth, scale)


def draw_points(image: np.ndarray, camera: "Camera", xyz: np.ndarray, colours: np.ndarray, *,
                radius: int = 1, alpha: float = 1.0) -> None:
    """Base-frame points as discs of their colours, nearest on top (in place).

    Points the camera cannot see are drawn too (the lidars sit elsewhere
    than the camera); they show between the discs of nearer ones. *alpha*
    blends the points over the image (the map labels drawn after them stay
    readable on dense clouds).
    """
    uv, depth, visible = project_points(camera, xyz)
    idx = np.flatnonzero(visible)
    idx = idx[np.argsort(-depth[idx])]  # far first, near drawn over it
    px = np.floor(uv[idx] + 0.5).astype(np.int64)
    h, w = image.shape[:2]
    layer = image.copy() if alpha < 1.0 else image
    covered = np.zeros((h, w), bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            r, c = px[:, 1] + dy, px[:, 0] + dx
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            layer[r[ok], c[ok]] = colours[idx[ok]]
            covered[r[ok], c[ok]] = True
    if alpha < 1.0:
        image[covered] = (alpha * layer[covered] + (1 - alpha) * image[covered]).astype(np.uint8)


def draw_radar(image: np.ndarray, camera: "Camera", xyz: np.ndarray, colours: np.ndarray, *, radius: int = 9) -> None:
    """Radar detections as rings, nearest on top (in place)."""
    cv2 = _deps.cv2()

    uv, depth, visible = project_points(camera, xyz)
    for i in np.flatnonzero(visible)[np.argsort(-depth[visible])]:
        centre = tuple(int(v) for v in np.floor(uv[i] + 0.5))
        cv2.circle(image, centre, radius + 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(image, centre, radius, tuple(int(c) for c in colours[i]), 3, cv2.LINE_AA)


def _pose_or_none(frame: "Frame"):
    """The frame's pose, None without one (no map labels there)."""
    from kitscenes.errors import PoseUnavailable

    try:
        return frame.pose
    except PoseUnavailable:
        return None


def _lidar_points(frame: "Frame", max_depth: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz, colours)``: the frame's lidar points off the vehicle body, coloured near red to far blue."""
    matplotlib = _deps.matplotlib()

    if not frame.sensors("lidar"):
        from kitscenes.constants import LIDAR_NAMES
        from kitscenes.errors import Availability

        for name in LIDAR_NAMES:  # no lidar here: say how to get one, if one is not downloaded
            if frame.scene.availability(name) is Availability.NOT_DOWNLOADED:
                frame.lidar(name)  # raises SensorUnavailable with the download command
    cloud = frame.fused_lidar()
    xyz = np.asarray(cloud.xyz, np.float32)
    xyz = xyz[~(np.abs(xyz) < np.asarray(EGO_BOX)).all(axis=1)]
    distance = np.linalg.norm(xyz, axis=1)
    colours = (matplotlib.colormaps["turbo_r"](np.clip(distance / max_depth, 0, 1))[:, :3] * 255).astype(np.uint8)
    return xyz, colours


def _radar_points(frame: "Frame", max_speed: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz, colours)`` of all radars: blue approaching, red receding, white static."""
    matplotlib = _deps.matplotlib()

    xyz, rate = [], []
    for name in frame.scene.rig.radars:
        cloud = frame.radar(name, missing="none")
        if cloud is None:
            continue
        xyz.append(cloud.xyz[cloud.measured] if cloud.measured is not None else cloud.xyz)
        rr = cloud.attrs.get("range_rate", np.zeros(len(cloud.xyz)))
        rate.append(rr[cloud.measured] if cloud.measured is not None else rr)
    if not xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    rate = np.concatenate(rate)
    colours = matplotlib.colormaps["bwr"](np.clip(rate / max_speed / 2 + 0.5, 0, 1))[:, :3] * 255
    return np.concatenate(xyz).astype(np.float64), colours.astype(np.uint8)


def _draw_styled(image: np.ndarray, uv: np.ndarray, depth: np.ndarray, colour, line_style: style.LineStyle,
                 scale: float) -> None:
    """One projected polyline: grey underlay, then the line (dashed and blended as its style says)."""
    if line_style.under:
        _polyline(image, uv, depth, GREY, _px(line_style.under, scale), line_style, scale)
    if line_style.alpha >= 1.0:
        _polyline(image, uv, depth, colour, _px(line_style.width, scale), line_style, scale)
        return
    # Blend only the box around the line, not the whole image.
    pad = _px(max(line_style.width, line_style.under), scale) + 4
    h, w = image.shape[:2]
    lo = np.floor(np.clip(uv.min(axis=0) - pad, 0, [w, h])).astype(int)
    hi = np.ceil(np.clip(uv.max(axis=0) + pad, 0, [w, h])).astype(int)
    if (hi <= lo).any():
        return
    cv2 = _deps.cv2()

    box = image[lo[1]:hi[1], lo[0]:hi[0]]
    overlay = box.copy()
    _polyline(overlay, uv - lo, depth, colour, _px(line_style.width, scale), line_style, scale)
    image[lo[1]:hi[1], lo[0]:hi[0]] = cv2.addWeighted(overlay, line_style.alpha, box, 1 - line_style.alpha, 0)


def _polyline(image, uv, depth, colour, width: int, line_style: style.LineStyle, scale: float) -> None:
    cv2 = _deps.cv2()

    if not line_style.dash:
        cv2.polylines(image, [_int(uv)], False, colour, width, cv2.LINE_AA)
        return
    _dashed(image, uv, depth, colour, width, line_style.dash * scale, line_style.gap * scale)


def _dashed(image, uv, depth, colour, width: int, dash: float, gap: float) -> None:
    """Dashes along the pixel polyline, longer near the camera and shorter far away (each dash by its own depth)."""
    cv2 = _deps.cv2()

    seg = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1:
        return
    min_dash, min_gap = max(6, int(np.ceil(2.0 * width))), max(4, int(np.ceil(1.5 * width)))
    start = 0.0
    while start < total:
        d = float(np.interp(start, s, depth))
        factor = np.clip(25.0 / max(d, 1.0), 0.45, 1.75)
        length, space = max(min_dash, dash * factor), max(min_gap, gap * factor)
        end = min(start + length, total)
        if end - start >= min(length, max(5.0, 1.5 * width)):
            inner = (s > start) & (s < end)
            pts = np.vstack([_at(uv, s, start), uv[inner], _at(uv, s, end)])
            cv2.polylines(image, [_int(pts)], False, colour, width, cv2.LINE_AA)
        start += length + space


def _at(uv: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, s, uv[:, 0]), np.interp(t, s, uv[:, 1])])


def _draw_traffic_element(image: np.ndarray, te_type: str, uv: np.ndarray, depth: np.ndarray, scale: float) -> None:
    cv2 = _deps.cv2()

    if te_type == "StopLine":
        cv2.polylines(image, [_int(uv)], False, RED, _px(5, scale), cv2.LINE_AA)
    elif style.is_upright(te_type):
        cv2.polylines(image, [_int(uv)], True, YELLOW, _px(3, scale), cv2.LINE_AA)
        icon = style.icon_rgba(te_type)
        if icon is not None:
            # Beside the box, sized by distance (64 px at 30 m and nearer, 16 px at most far).
            height = int(np.clip(64 * min(2.0, 30.0 / max(1.0, float(depth.mean()))), 16, 64) * scale)
            _paste(image, _resized(icon, height), (int(uv[:, 0].max()) + height // 2 + 4, int(uv[:, 1].mean())), 0.4)
    else:  # markings on the road: the icon laid along the line, then the line
        icon = style.icon_rgba(te_type)
        a, b = uv[0], uv[-1]
        length = float(np.linalg.norm(b - a))
        if icon is not None and length > 5 * scale:
            icon = np.rot90(icon, k=1)
            along = (b - a) / length
            across = np.array([-along[1], along[0]])
            width = max(16 * scale, min(64 * scale, 0.5 * length))
            height = width * icon.shape[1] / icon.shape[0]
            centre = (a + b) / 2
            quad = np.array([centre - across * width / 2 + along * height / 2, centre + across * width / 2 + along * height / 2,
                             centre + across * width / 2 - along * height / 2, centre - across * width / 2 - along * height / 2])
            _warp(image, icon, quad, 0.4)
        cv2.polylines(image, [_int(uv)], False, YELLOW, _px(3, scale), cv2.LINE_AA)


def _resized(icon: np.ndarray, height: int) -> np.ndarray:
    cv2 = _deps.cv2()

    width = max(1, round(icon.shape[1] * height / icon.shape[0]))
    return cv2.resize(icon, (width, max(1, height)), interpolation=cv2.INTER_AREA)


def _paste(image: np.ndarray, icon: np.ndarray, centre: tuple[int, int], alpha: float) -> None:
    h, w = icon.shape[:2]
    x0, y0 = centre[0] - w // 2, centre[1] - h // 2
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(image.shape[1], x0 + w), min(image.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    part = icon[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = part[..., 3:4] / 255.0 * alpha
    region = image[Y0:Y1, X0:X1]
    image[Y0:Y1, X0:X1] = (a * part[..., :3] + (1 - a) * region).astype(np.uint8)


def _warp(image: np.ndarray, icon: np.ndarray, quad: np.ndarray, alpha: float) -> None:
    cv2 = _deps.cv2()

    h, w = image.shape[:2]
    lo = np.maximum(np.floor(quad.min(axis=0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + 2, [w, h])
    if (hi <= lo).any():
        return
    src = np.float32([[0, 0], [icon.shape[1] - 1, 0], [icon.shape[1] - 1, icon.shape[0] - 1], [0, icon.shape[0] - 1]])
    matrix = cv2.getPerspectiveTransform(src, np.float32(quad - lo))
    warped = cv2.warpPerspective(np.ascontiguousarray(icon), matrix, (int(hi[0] - lo[0]), int(hi[1] - lo[1])))
    a = warped[..., 3:4] / 255.0 * alpha
    region = image[lo[1]:hi[1], lo[0]:hi[0]]
    image[lo[1]:hi[1], lo[0]:hi[0]] = (a * warped[..., :3] + (1 - a) * region).astype(np.uint8)


def _px(width: float, scale: float) -> int:
    return max(1, int(round(width * scale)))


def _int(uv: np.ndarray) -> np.ndarray:
    """Pixel coordinates for OpenCV, kept inside int32 (lines far off-screen are clipped by OpenCV)."""
    return np.clip(np.floor(uv + 0.5), -1e6, 1e6).astype(np.int32).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# A whole scene on disk
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RenderOptions:
    """What :func:`render_scene` draws and writes."""

    map: bool = True
    lidar: bool = False
    radar: bool = False
    top_down: bool = True
    grid: bool = True
    cameras: tuple[str, ...] = GRID_CAMERAS
    reduce: int = 1
    grouping: str = "default"
    quality: int = 90


def render_scene(
    scene: "Scene",
    out_dir: Path | str,
    *,
    frames: Optional[Iterable[int]] = None,
    options: RenderOptions = RenderOptions(),
    workers: int = 4,
    progress: bool = True,
) -> Path:
    """Render frames of *scene* into *out_dir*: ``<camera>/``, ``grid/`` and ``top_down/`` JPEGs per frame.

    Frames are spread over *workers* processes (each makes its own map
    layer). *out_dir* must not lie inside the dataset (``$KITSCENES_ROOT``).

    Returns:
        *out_dir*.

    Raises:
        ValueError: If *out_dir* lies inside the dataset.
    """
    from concurrent.futures import ProcessPoolExecutor

    out = _deps.guard_output_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = list(range(len(scene)) if frames is None else frames)
    jobs = [(scene, i, out, options) for i in frames]
    if workers <= 1:
        results = map(_render_one, jobs)
    else:
        import multiprocessing as mp

        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        results = pool.map(_render_one, jobs)
    try:
        if progress:
            from tqdm import tqdm

            results = tqdm(results, total=len(jobs), desc=f"rendering {scene.id[:8]}")
        for _ in results:
            pass
    finally:
        if workers > 1:
            pool.shutdown()
    return out


_LAYERS: dict = {}


def _render_one(job) -> int:
    scene, index, out, options = job
    from PIL import Image

    key = (scene.id, options.grouping)
    if options.map and key not in _LAYERS:
        _LAYERS[key] = MapLayer(scene, grouping=options.grouping)
    layer = _LAYERS.get(key) if options.map else None
    frame = scene.frame(index)
    pose = _pose_or_none(frame)
    labels = layer(pose) if layer is not None and pose is not None else None
    images = render_frame(frame, labels=labels, map_layer=layer, lidar=options.lidar, radar=options.radar,
                          cameras=options.cameras, reduce=options.reduce)
    name = f"{index:010d}.jpg"

    def save(image: np.ndarray, folder: str) -> None:
        (out / folder).mkdir(exist_ok=True)
        Image.fromarray(image).save(out / folder / name, quality=options.quality)

    for camera, image in images.items():
        save(image, camera)
    if options.grid and images:
        save(camera_grid(images, title=f"Frame {index:010d}"), "grid")
    if options.top_down and labels is not None:
        from kitscenes.visualization.topdown import render_top_down

        save(render_top_down(labels[0], extent=(60.0, 30.0), exclude_te_ids=layer.hidden_ids), "top_down")
    return index
