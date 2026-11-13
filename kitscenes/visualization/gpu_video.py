"""Videos of scenes: the six ring cameras with the map labels drawn in, a BEV panel and a legend.

A visualization of the data (no model runs)::

    render_video("/data/kitscenes", "/tmp/val.mp4", splits="val", resolution="4k", speed=2)
    kitscenes video /data/kitscenes --splits val --out /tmp/val.mp4

Reader processes decode the JPEGs and prepare the labels; the renderer draws each frame with
torch and pipes it into ffmpeg (NVENC on a GPU, a software encoder without). One MP4 per scene,
joined at the end. Needs ``kitscenes[video,map]`` and the ffmpeg binary. See docs/visualization.md.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

import numpy as np

from kitscenes.decode import _decode_reduced
from kitscenes.visualization import _deps, style
from kitscenes.visualization.reprojection import GREY, RED, GRID_CAMERAS, YELLOW, _pose_or_none
from kitscenes.visualization.topdown import _point_in_band

__all__ = ["render_video", "VideoResult", "HOOD_FRACTIONS", "parse_frames", "main"]

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from kitscenes.catalog import Dataset
    from kitscenes.map import MapLabels, Region
    from kitscenes.scene import Scene

#: Frames per second of the dataset: ``speed=1`` plays at this rate.
SOURCE_FPS = 10.0
#: Resolution presets (width, height).
RESOLUTIONS = {"1080p": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160)}
#: Default labelled region: length (along the vehicle) x width in metres, centred on it.
DEFAULT_REGION = (120.0, 60.0)
# The values of render_video's options (explained there).
LAYOUTS = ("cams", "cams+bev")
LEGENDS = ("compact", "full", "none")
FITS = ("cover", "contain")
REAR_ROWS = ("swap", "mirror", "plain")
CODECS = ("hevc", "h264")
ENCODERS = ("auto", "nvenc", "software")
ORDERS = ("recording", "id")
#: Aspect (width / height) of the ring cameras (3504 x 2272).
CAMERA_ASPECT = 3504 / 2272
#: Width of the right column (BEV on top, legend below) of layout ``cams+bev``, a fraction of the frame width.
BEV_COLUMN = 0.2
#: Part of each ring camera image, from the bottom, that shows the ego vehicle (hood, body, the sensor
#: mounts at the front corners), cut away with ``crop_hood=True`` (fractions of the image height).
HOOD_FRACTIONS = {"camera_ring_front_left": 0.22, "camera_ring_front": 0.15, "camera_ring_front_right": 0.22,
                  "camera_ring_rear_left": 0.10, "camera_ring_rear": 0.16, "camera_ring_rear_right": 0.08}
#: Default constant quality per encoder (NVENC ``-cq``, software ``-crf``; lower is better).
DEFAULT_QUALITY = {"hevc_nvenc": 19, "h264_nvenc": 19, "libx265": 21, "libx264": 20}
#: HEVC bitrate cap of the constant-quality mode at 3840 x 2160 and 10 fps; scaled by the pixels, sqrt(fps / 10)
#: and the codec (:data:`CODEC_RATE`), see :func:`max_bitrate`.
MAX_BITRATE_4K = 80e6
#: Bitrate of each codec relative to HEVC at the same quality (roughly).
CODEC_RATE = {"hevc": 1.0, "h264": 1.5}
#: Oldest torch the renderer runs with (``torch.meshgrid(indexing=)``, ``torch.frombuffer``).
VIDEO_TORCH_MIN_VERSION = "1.10"
CAPTIONS = {"camera_ring_front_left": "front left", "camera_ring_front": "front",
            "camera_ring_front_right": "front right", "camera_ring_rear_left": "rear left",
            "camera_ring_rear": "rear", "camera_ring_rear_right": "rear right"}


# ---------------------------------------------------------------------------
# Options (plain Python, no torch)
# ---------------------------------------------------------------------------


def parse_resolution(spec: Union[str, Sequence[int]]) -> tuple[int, int]:
    """``"1080p"``, ``"2k"``, ``"4k"`` or ``"WxH"`` (or a ``(W, H)`` pair) -> ``(W, H)``, both even, at least 320 x 180."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in RESOLUTIONS:
            return RESOLUTIONS[key]
        parts = key.replace("*", "x").split("x")
        try:
            w, h = (int(p) for p in parts)
        except ValueError:
            raise ValueError(f"resolution: one of {', '.join(RESOLUTIONS)} or WxH (e.g. 3000x1500), got {spec!r}") from None
    else:
        w, h = (int(v) for v in spec)
    if w % 2 or h % 2 or w < 320 or h < 180:
        raise ValueError(f"resolution must be even and at least 320x180 (NV12 frames), got {w}x{h}")
    return w, h


@dataclasses.dataclass(frozen=True)
class Playback:
    """How a selection plays: every ``step``-th frame at ``fps`` (see :func:`playback`).

    Attributes:
        step: Frames advanced per video frame (``frame_step`` x ``dropped``).
        fps: Video frame rate, exact (``Fraction``).
        dropped: The factor of frames dropped to stay at or below ``max_fps`` (1: none).
    """

    step: int
    fps: Fraction
    dropped: int


def playback(speed: float = 1.0, frame_step: int = 1, max_fps: float = 60.0) -> Playback:
    """Video frame rate for a playback *speed* (1 = real time: 10 fps for every frame).

    ``fps = 10 * speed / frame_step``. If that is above *max_fps*, frames are
    dropped (every 2nd, 3rd, ... of the selection) until it is not:
    ``speed=8`` gives every 2nd frame at 40 fps.
    """
    if not speed > 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    if not max_fps > 0:
        raise ValueError(f"max_fps must be > 0, got {max_fps}")
    rate = Fraction(SOURCE_FPS).limit_denominator() * Fraction(speed).limit_denominator(1000) / frame_step
    dropped = max(1, math.ceil(rate / Fraction(max_fps).limit_denominator(1000)))
    return Playback(frame_step * dropped, rate / dropped, dropped)


def parse_region(spec: Union[str, Sequence[float], "Region", None]) -> "Region":
    """``"120x60"`` / ``(120, 60)`` (length x width in metres, centred on the vehicle) or a Region -> Region."""
    from kitscenes.map import Region

    if spec is None:
        spec = DEFAULT_REGION
    if isinstance(spec, Region):
        return spec
    if isinstance(spec, str):
        try:
            length, width = (float(p) for p in spec.lower().split("x"))
        except ValueError:
            raise ValueError(f"region: LENGTHxWIDTH in metres, e.g. 120x60, got {spec!r}") from None
    else:
        length, width = (float(v) for v in spec)
    if not (length > 0 and width > 0):
        raise ValueError(f"region must have a positive length and width, got {length} x {width}")
    return Region(length / 2, length / 2, width / 2, width / 2)


def max_bitrate(size: tuple[int, int], fps: float, codec: str = "hevc") -> int:
    """Bitrate cap (bits per second) of the constant-quality mode for *size* and *fps* (80 Mbit/s HEVC at 4K and
    10 fps, scaled by the pixels, sqrt(fps / 10) and the codec)."""
    rate = MAX_BITRATE_4K * size[0] * size[1] / (3840 * 2160) * math.sqrt(max(1.0, float(fps) / SOURCE_FPS))
    return int(round(rate * CODEC_RATE.get(codec, 1.0), -5))


def parse_bitrate(value: Union[str, int, float, None]) -> Optional[int]:
    """``"20M"``, ``"8000k"``, ``12e6`` -> bits per second (None stays None)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip().lower()
    factor = {"k": 1e3, "m": 1e6, "g": 1e9}.get(text[-1:], 1)
    try:
        return int(float(text[:-1] if factor != 1 else text) * factor)
    except ValueError:
        raise ValueError(f"bitrate: e.g. 20M or 8000k, got {value!r}") from None


@dataclasses.dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h


def parse_crop_hood(spec: Union[bool, float, dict, None]) -> dict:
    """The bottom fraction cut from each camera: False/None (none), True (:data:`HOOD_FRACTIONS`), a fraction
    for every camera, or ``{camera: fraction}`` (the others: none). Fractions in [0, 0.5)."""
    if spec is None or spec is False:
        return {}
    if spec is True or (isinstance(spec, str) and spec.strip().lower() in ("default", "true", "yes", "")):
        out = dict(HOOD_FRACTIONS)
    elif isinstance(spec, (int, float, str)):
        try:
            value = float(spec)
        except ValueError:
            raise ValueError(f"crop_hood: True, a fraction of the image height or {{camera: fraction}}, got {spec!r}") \
                from None
        out = {name: value for name in GRID_CAMERAS}
    else:
        unknown = [k for k in spec if k not in GRID_CAMERAS]
        if unknown:
            raise ValueError(f"crop_hood: unknown camera(s) {unknown}; the ring cameras are {list(GRID_CAMERAS)}")
        out = {k: float(v) for k, v in spec.items()}
    for name, value in out.items():
        if not 0.0 <= value < 0.5:
            raise ValueError(f"crop_hood: fractions must be in [0, 0.5), got {value} for {name}")
    return {k: v for k, v in out.items() if v > 0}


@dataclasses.dataclass(frozen=True)
class Layout:
    """Where everything goes in a video frame.

    ``cams+bev``: header; on the right a column (:data:`BEV_COLUMN` of the
    width) with the BEV panel on top (driving up, as
    :func:`~kitscenes.visualization.plot_top_down`; as high as the region is
    long at the column's width) and the legend below it; the 3 x 2 camera
    grid (front row on top) fills the rest, the full height left of the column.
    ``cams``: header, the camera grid over the full width, the legend in a
    slim strip at the bottom. The options are those of :func:`render_video`.

    Attributes:
        tiles: camera -> where its image goes in the frame.
        windows: camera -> the part of its image shown, ``(x0, y0, x1, y1)`` in fractions of the image.
        legend: The legend area (None: no legend).
        legend_mode: ``"compact"``, ``"full"`` or ``"none"``.
        bev: The BEV panel (None for ``cams``).
        mirrored: Cameras whose images (and labels) are mirrored left-right (``rear_row="mirror"``).
    """

    size: tuple[int, int]
    header: Rect
    tiles: dict
    windows: dict
    legend: Optional[Rect]
    bev: Optional[Rect]
    legend_mode: str = "compact"
    mirrored: frozenset = frozenset()

    @property
    def tile_size(self) -> tuple[int, int]:
        """Size of the first tile (all tiles have it with ``fit="cover"``)."""
        t = next(iter(self.tiles.values()))
        return t.w, t.h

    @property
    def font(self) -> float:
        """Text scale of the frame (1 at 4K, 0.5 at 1080p)."""
        return self.header.h / 80.0

    @classmethod
    def compute(cls, size: tuple[int, int], layout: str = "cams+bev", region=None, *, legend: str = "compact",
                crop_hood=None, fit: str = "cover", rear_row: str = "swap",
                aspect: float = CAMERA_ASPECT) -> "Layout":
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        if legend not in LEGENDS:
            raise ValueError(f"legend must be one of {LEGENDS}, got {legend!r}")
        if fit not in FITS:
            raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
        if rear_row not in REAR_ROWS:
            raise ValueError(f"rear_row must be one of {REAR_ROWS}, got {rear_row!r}")
        hood = parse_crop_hood(crop_hood)
        W, H = size
        header_h = max(24, int(round(H * 0.037 / 2)) * 2)
        body = H - header_h
        font = header_h / 80.0
        bev = legend_rect = None
        if layout == "cams+bev":
            region = parse_region(region)
            length, width = region.ahead + region.behind, region.left + region.right
            col = int(round(W * BEV_COLUMN / 2)) * 2
            room = body if legend == "none" else body - int(0.2 * body)  # the legend gets at least a fifth
            bev_h = min(int(round(col * length / width)), room)
            bev = Rect(W - col, header_h, col, bev_h)
            if legend != "none":
                legend_rect = Rect(W - col, header_h + bev_h, col, body - bev_h)
            grid = Rect(0, header_h, W - col, body)
        else:
            strip = 0
            if legend != "none":  # rows of entries: two (compact; more entries get a smaller font), three (full)
                rows = 2 if legend == "compact" else 3
                strip = int(round((rows * LEGEND_ROW + 2 * LEGEND_PAD) * LEGEND_SCALE * font))
                legend_rect = Rect(0, H - strip, W, strip)
            grid = Rect(0, header_h, W, body - strip)
        cw, ch = grid.w // 3, grid.h // 2
        x0 = grid.x + (grid.w - 3 * cw) // 2
        tiles, windows = {}, {}
        for k, name in enumerate(grid_order(rear_row)):
            cell = Rect(x0 + (k % 3) * cw, grid.y + (k // 3) * ch, cw, ch)
            keep = 1.0 - hood.get(name, 0.0)  # height of the image left
            shown = aspect / keep  # its aspect
            if fit == "cover":
                tiles[name] = cell
                if shown >= cw / ch:  # cut the sides
                    f = (cw / ch) / shown
                    windows[name] = ((1 - f) / 2, 0.0, (1 + f) / 2, keep)
                else:  # cut top and bottom (of the part left)
                    f = keep * shown / (cw / ch)
                    windows[name] = (0.0, (keep - f) / 2, 1.0, (keep + f) / 2)
            else:
                tw, th = (cw, int(round(cw / shown))) if shown >= cw / ch else (int(round(ch * shown)), ch)
                tiles[name] = Rect(cell.x + (cw - tw) // 2, cell.y + (ch - th) // 2, tw, th)
                windows[name] = (0.0, 0.0, 1.0, keep)
        mirrored = frozenset(GRID_CAMERAS[3:]) if rear_row == "mirror" else frozenset()
        return cls(size, Rect(0, 0, W, header_h), tiles, windows, legend_rect, bev, legend, mirrored)


def grid_order(rear_row: str = "swap") -> tuple:
    """The ring cameras in grid order (front row left to right, then the rear row) for *rear_row*."""
    front, rear = GRID_CAMERAS[:3], GRID_CAMERAS[3:]
    return tuple(front) + (tuple(reversed(rear)) if rear_row == "swap" else tuple(rear))


#: Legend metrics per unit of text scale: row height, padding (pixels), and the text scale per unit of
#: :attr:`Layout.font` the legend aims at (0.45 at 1080p: about 10 px capitals); it shrinks to fit.
LEGEND_ROW, LEGEND_PAD, LEGEND_SCALE = 52, 14, 0.9


def legend_entries(present: Optional[tuple] = None, layout: str = "cams+bev") -> list[tuple[str, str]]:
    """``(key, text)`` of the legend: every class (*present* None) or those in *present*.

    Args:
        present: ``(line classes, traffic element classes, edges)`` of a scene
            (:class:`~kitscenes.visualization._scene_labels.SceneLabels`), or None for all.
        layout: The edges ("Applies to") are only drawn in the BEV (``cams+bev``).
    """
    lines, tes, edges = present if present is not None else (None, None, True)
    names = dict(style.LEGEND_NAMES)
    stop = names.pop("StopLine")
    entries = list(names.items())
    centre = [k for k, _ in entries].index("Centerline")
    entries[centre:centre] = [("BikeMarkingDashed", "Bike marking dashed"), ("BikeMarkingSolid", "Bike marking solid")]
    if lines is not None:
        entries = [e for e in entries if e[0] in lines]
        entries += [(k, "Other line" if k == "Unknown" else k) for k in sorted(lines) if k not in style.LEGEND_NAMES
                    and k not in ("BikeMarkingDashed", "BikeMarkingSolid")]
    te_entries = [("StopLine", stop), ("TLCar", "Traffic light"), ("TSMisc", "Traffic sign"),
                  ("ArrowGoStraight", "Road marking")]
    if tes is not None:
        def shown(key: str) -> bool:
            if key == "StopLine":
                return "StopLine" in tes
            if key in ("TLCar", "TSMisc"):
                return any(t.startswith(key[:2]) for t in tes)
            return any(t != "StopLine" and not style.is_upright(t) and style.icon_rgba(t) is not None for t in tes)
        te_entries = [e for e in te_entries if shown(e[0])]
    entries += te_entries
    if layout == "cams+bev" and edges:
        entries.append(("_edge", "Applies to (BEV)"))
    return entries


def crop_window(image_size: tuple[int, int], window: tuple) -> tuple[int, int, int, int]:
    """A window in fractions of the image -> pixels ``(x0, y0, x1, y1)``, on multiples of 8 (exact at
    every JPEG DCT reduction)."""
    w, h = image_size

    def snap(v: float, n: int) -> int:
        return int(min(max(round(v * n / 8.0) * 8, 0), n))

    x0, y0, x1, y1 = snap(window[0], w), snap(window[1], h), snap(window[2], w), snap(window[3], h)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def decode_reduction(image_size: tuple[int, int], tile_size: tuple[int, int]) -> int:
    """The largest JPEG DCT reduction (8, 4, 2, 1) whose output is still at least the tile size."""
    w, h = image_size
    for r in (8, 4, 2):
        if -(-w // r) >= tile_size[0] and -(-h // r) >= tile_size[1]:
            return r
    return 1


def frame_indices(count: int, frames: Union[slice, range, Sequence[int], None] = None, step: int = 1) -> list[int]:
    """The frames of a scene with *count* frames: *frames* (a slice, range or indices; None: all), then every *step*-th."""
    if frames is None:
        chosen = list(range(count))
    elif isinstance(frames, slice):
        chosen = list(range(count)[frames])
    else:
        chosen = [int(i) for i in frames if 0 <= int(i) < count]
    return chosen[::max(1, int(step))]


def parse_frames(spec: Optional[str]) -> Optional[slice]:
    """``"START:STOP[:STEP]"`` (a Python slice; ends may be left out) or one index -> slice."""
    if not spec:
        return None
    try:
        parts = [int(p) if p.strip() else None for p in spec.split(":")]
    except ValueError:
        raise ValueError(f"frames: START:STOP[:STEP], got {spec!r}") from None
    if len(parts) == 1:
        return slice(parts[0], parts[0] + 1)
    if len(parts) > 3:
        raise ValueError(f"frames: START:STOP[:STEP], got {spec!r}")
    return slice(*parts)


def select_scenes(
    ds: "Dataset",
    *,
    splits=None,
    scene_ids: Optional[Sequence[str]] = None,
    city=None,
    lidar=None,
    where: Optional[dict] = None,
    order: str = "recording",
    max_scenes: Optional[int] = None,
) -> list[str]:
    """Scene IDs of *ds* matching the criteria (as :meth:`kitscenes.Dataset.filter`), in *order*.

    ``"recording"``: by the scene table's ``recording_order``; scenes without
    one (test-e2e withholds it) follow, by split and ID. ``"id"``: by split, then ID.
    """
    from kitscenes.splits import scene_table, split_sort_key

    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    sub = ds.filter(splits=splits, scene_ids=scene_ids, city=city, lidar=lidar, **(where or {}))
    table = scene_table()

    def key(scene_id: str):
        info = table.get(scene_id)
        rank = info.recording_order if order == "recording" and info is not None and info.recording_order else None
        return (rank is None, rank or 0, split_sort_key(sub.split_of(scene_id)), scene_id)

    ids = sorted(sub.scene_ids, key=key)
    return ids[:max_scenes] if max_scenes else ids


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------


def ffmpeg_path() -> str:
    """The ffmpeg binary (``$KITSCENES_FFMPEG`` or on ``PATH``).

    Raises:
        MissingDependencyError: If there is none.
    """
    from kitscenes.errors import MissingDependencyError

    path = os.environ.get("KITSCENES_FFMPEG") or shutil.which("ffmpeg")
    if not path:
        raise MissingDependencyError(
            "kitscenes.visualization.gpu_video encodes with the ffmpeg binary, which is not on PATH. Install it "
            "(apt install ffmpeg, conda install ffmpeg, or a static build; for the GPU encoder one built with "
            "NVENC) or point $KITSCENES_FFMPEG to it.")
    return path


_PROBED: dict = {}


def probe_encoder(name: str, args: Sequence[str] = ()) -> Optional[str]:
    """None if ffmpeg can encode a test frame with *name* (and the options *args*), else why not (ffmpeg's message)."""
    key = (name, tuple(args))
    if key in _PROBED:
        return _PROBED[key]
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=10",
           "-frames:v", "2", "-pix_fmt", "nv12", "-c:v", name, *args, "-f", "null", "-"]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        err = None if run.returncode == 0 else (run.stderr.strip().splitlines() or ["failed"])[-1]
    except (OSError, subprocess.TimeoutExpired) as exc:
        err = str(exc)
    _PROBED[key] = err
    return err


def choose_encoder(codec: str = "hevc", encoder: str = "auto", gpu: bool = True) -> str:
    """The ffmpeg encoder: NVENC on a GPU (``auto`` falls back to the software one, with a warning).

    Raises:
        MissingDependencyError: Without ffmpeg, or for ``encoder="nvenc"`` when NVENC does not work.
    """
    from kitscenes.errors import MissingDependencyError

    if codec not in CODECS:
        raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    nvenc, software = {"hevc": ("hevc_nvenc", "libx265"), "h264": ("h264_nvenc", "libx264")}[codec]
    if encoder == "nvenc" or (encoder == "auto" and gpu):
        err = probe_encoder(nvenc)
        if err is None:
            return nvenc
        hint = (f"ffmpeg cannot encode with {nvenc}: {err}. It needs an ffmpeg built with NVENC, the NVIDIA "
                "driver's libnvidia-encode, and in a container NVIDIA_DRIVER_CAPABILITIES including 'video' "
                "(e.g. docker run --gpus ... -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video).")
        if encoder == "nvenc":
            raise MissingDependencyError(hint + " Or use encoder='software'.")
        warnings.warn(hint + f" Encoding with {software} on the CPU instead.", stacklevel=3)
    err = probe_encoder(software)
    if err is not None:
        raise MissingDependencyError(f"ffmpeg cannot encode with {software}: {err} (an ffmpeg built with it is needed).")
    return software


@dataclasses.dataclass(frozen=True)
class RateControl:
    """How the encoder spends bits: constant *quality* (default) up to *max_rate*, or an average *bitrate*.

    Attributes:
        quality: Constant quality (NVENC ``-cq``, software ``-crf``; lower is better), None with *bitrate*.
        max_rate: Bitrate cap of the constant-quality mode (bits per second).
        bitrate: Average bitrate (VBR, peaks up to twice), instead of constant quality.
        basic: NVENC without the newer options (an older ffmpeg; see :func:`rate_control`).
    """

    quality: Optional[float] = None
    max_rate: Optional[int] = None
    bitrate: Optional[int] = None
    basic: bool = False

    def args(self, encoder: str, gpu: Optional[int] = None) -> list[str]:
        """ffmpeg options for *encoder* (``-gpu``: the NVENC GPU, a CUDA index as torch counts them)."""
        q = None if self.quality is None else f"{self.quality:g}"
        if self.bitrate:
            rate = ["-b:v", str(int(self.bitrate)), "-maxrate", str(2 * int(self.bitrate)),
                    "-bufsize", str(2 * int(self.bitrate))]
        else:
            rate = ["-b:v", "0"] if encoder.endswith("_nvenc") else []
            if self.max_rate:
                rate += ["-maxrate", str(int(self.max_rate)), "-bufsize", str(2 * int(self.max_rate))]
        if encoder.endswith("_nvenc"):
            if self.basic:
                args = ["-preset", "slow", "-rc", "vbr"]
            else:  # quality preset, adaptive quantization (dark and flat areas), lookahead, B-frames
                args = ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-spatial-aq", "1", "-temporal-aq", "1",
                        "-rc-lookahead", "20", "-bf", "3"]
                if encoder == "hevc_nvenc":
                    args += ["-b_ref_mode", "middle"]
            args += ([] if self.bitrate else ["-cq", q]) + rate
            if gpu is not None:
                args += ["-gpu", str(int(gpu))]
            return args
        args = ["-preset", "veryfast" if encoder == "libx264" else "fast"] + ([] if self.bitrate else ["-crf", q])
        args += rate + ["-pix_fmt", "yuv420p"]
        if encoder == "libx265":
            args += ["-x265-params", "log-level=error"]
        return args

    def describe(self, encoder: str) -> str:
        if self.bitrate:
            return f"{encoder} {self.bitrate / 1e6:g} Mbit/s"
        kind = "cq" if encoder.endswith("_nvenc") else "crf"
        cap = f" (max {self.max_rate / 1e6:g} Mbit/s)" if self.max_rate else ""
        return f"{encoder} {kind} {self.quality:g}{cap}" + (" basic" if self.basic else "")


def rate_control(encoder: str, size: tuple[int, int], fps: float, codec: str = "hevc",
                 quality: Optional[float] = None, bitrate: Union[str, int, None] = None,
                 max_rate: Union[str, int, None] = None) -> RateControl:
    """The rate control of *encoder*: constant quality (*quality*, default :data:`DEFAULT_QUALITY`) up to *max_rate*
    (default :func:`max_bitrate`), or the average *bitrate* if given.

    An ffmpeg whose NVENC lacks the newer options (``-preset p5``, ``-tune hq``, temporal AQ, B-frames) gets
    constant quality with ``-preset slow``, with a warning.
    """
    if bitrate is not None and quality is not None:
        raise ValueError("give either a quality or a bitrate, not both")
    rate = parse_bitrate(bitrate)
    if rate is not None:
        if rate <= 0:
            raise ValueError(f"bitrate must be > 0, got {bitrate!r}")
        rc = RateControl(bitrate=rate)
    else:
        q = DEFAULT_QUALITY.get(encoder, 23) if quality is None else float(quality)
        if not 0 < q <= 51:
            raise ValueError(f"quality must be in (0, 51] (NVENC -cq, software -crf; lower is better), got {quality!r}")
        rc = RateControl(quality=q, max_rate=parse_bitrate(max_rate) or max_bitrate(size, fps, codec))
    if encoder.endswith("_nvenc"):
        err = probe_encoder(encoder, rc.args(encoder))
        if err is not None:
            basic = dataclasses.replace(rc, basic=True)
            if probe_encoder(encoder, basic.args(encoder)) is not None:
                raise RuntimeError(f"ffmpeg cannot encode with {encoder} and {' '.join(rc.args(encoder))}: {err}")
            warnings.warn(f"this ffmpeg's {encoder} lacks some options ({err}); encoding with -preset slow and "
                          "without adaptive quantization (a newer ffmpeg, 5.0 or later, looks better)", stacklevel=2)
            rc = basic
    return rc


class Encoder:
    """ffmpeg fed with NV12 frames through a pipe by a writer thread; writes *path* when closed.

    :meth:`write` takes a numpy array, or ``(pinned tensor, CUDA event, pool)``: the thread waits for the
    event (the copy from the GPU), writes the buffer and puts it back into the pool.

    Args:
        rate: The rate control (:func:`rate_control`).
        gpu: NVENC: the GPU to encode on (the CUDA index torch uses, as ``cuda:K``); None: ffmpeg's choice
            (the first GPU).
    """

    QUEUE = 8

    def __init__(self, path: Path, size: tuple[int, int], fps: Fraction, encoder: str, rate: RateControl,
                 gpu: Optional[int] = None) -> None:
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial.mp4")
        W, H = size
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "nv12",
               "-s", f"{W}x{H}", "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-", "-c:v", encoder]
        cmd += rate.args(encoder, gpu if encoder.endswith("_nvenc") else None)
        if encoder in ("hevc_nvenc", "libx265"):
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-movflags", "+faststart", "-f", "mp4", str(self.partial)]
        self._log = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        self.frames = 0
        self._error: Optional[BaseException] = None
        self._q: "queue.Queue" = queue.Queue(maxsize=self.QUEUE)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            buf = self._q.get()
            if buf is None:
                return
            pool = None
            if isinstance(buf, tuple):
                buf, ready, pool = buf
                ready.synchronize()
                data = buf.numpy()
            else:
                data = buf
            if self._error is None:
                try:
                    self.proc.stdin.write(memoryview(data))
                except (BrokenPipeError, OSError) as exc:
                    self._error = exc
            if pool is not None:
                pool.put(buf)

    def write(self, nv12) -> None:
        if self._error is not None:
            self._fail()
        self._q.put(nv12)
        self.frames += 1

    def _stderr(self) -> str:
        self._log.seek(0)
        return self._log.read().decode(errors="replace").strip()

    def _fail(self):
        self.proc.kill()
        raise RuntimeError(f"ffmpeg failed writing {self.path}: {self._stderr() or self._error}")

    def close(self) -> Path:
        self._q.put(None)
        self._thread.join()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        if code != 0 or self._error is not None:
            raise RuntimeError(f"ffmpeg failed writing {self.path} (exit {code}): {self._stderr() or self._error}")
        self._log.close()
        os.replace(self.partial, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait()
        finally:
            self._log.close()
            if self.partial.exists():
                self.partial.unlink()


def concat_videos(parts: Sequence[Path], out: Path, list_dir: Path) -> Path:
    """Join MP4s with the same encoding into *out* without re-encoding (ffmpeg's concat demuxer)."""
    out = Path(out)
    if len(parts) == 1:
        shutil.copyfile(parts[0], out)
        return out
    listing = Path(list_dir) / "concat.txt"
    listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))
    run = subprocess.run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                          "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)],
                         capture_output=True, text=True)
    listing.unlink()
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg could not join the scene videos into {out}: {run.stderr.strip()}")
    return out


# ---------------------------------------------------------------------------
# Data: read, label and (CPU) decode, in DataLoader workers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Item:
    position: int
    scene_id: str
    frame: int
    decoded: tuple  # (cameras, (n, th, tw, 3) uint8 tiles, or a list of (th, tw, 3) for tiles of different sizes, or None)
    plan: "FramePlan"
    note: str  # why there are no labels ("" if there are)
    legend: Optional[list] = None  # the scene's legend entries (compact legend)


def _identity(item):
    return item


def _scene_labels_task(args) -> tuple:
    """``(scene id, SceneLabels or None, note)`` (in a process of a pool)."""
    ds, scene_id, region, grouping, frames = args
    import warnings as _warnings

    from kitscenes.errors import KITScenesError
    from kitscenes.visualization._scene_labels import SceneLabels

    scene = ds.scene(scene_id)
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # map problems are reported by the map API, not per video
            poses = [p for p in (_pose_or_none(scene.frame(k)) for k in frames) if p is not None]
            if not poses:
                return scene_id, None, "no map labels: no poses"
            return scene_id, SceneLabels.build(scene, region, grouping, poses=poses), ""
    except KITScenesError as err:  # MapUnavailable (test splits), LabelsUnavailable, ...
        return scene_id, None, _short_reason(err)


def scene_labels(ds: "Dataset", scenes: Sequence[tuple[str, Sequence[int]]], region: "Region",
                 grouping: str = "default", processes: int = 8) -> dict:
    """``{scene id: (SceneLabels or None, note)}`` for ``(scene id, frames)``: the map labels of each scene,
    computed once (in *processes* forked processes), cut per frame by the readers."""
    import multiprocessing

    tasks = [(ds, scene_id, region, grouping, tuple(frames)) for scene_id, frames in scenes]
    if processes > 1 and len(tasks) > 1 and "fork" in multiprocessing.get_all_start_methods():
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(processes, len(tasks)), mp_context=multiprocessing.get_context("fork")) as pool:
            done = list(pool.map(_scene_labels_task, tasks))
    else:
        done = [_scene_labels_task(t) for t in tasks]
    return {scene_id: (labels, note) for scene_id, labels, note in done}


class _FrameSource:
    """Map-style dataset of the frames to render (a torch DataLoader indexes it): images, labels, the frame plan."""

    def __init__(self, ds: "Dataset", items: Sequence[tuple[str, int]], *, layout: Layout, region: "Region",
                 line_width: float, speed: float = 1.0, labels: Optional[dict] = None) -> None:
        self.ds, self.items = ds, list(items)
        self.speed = speed
        self.counts: dict = {}
        for scene_id, _ in self.items:
            self.counts[scene_id] = self.counts.get(scene_id, 0) + 1
        self.rank = {}
        seen: dict = {}
        for scene_id, k in self.items:
            seen[scene_id] = seen.get(scene_id, 0) + 1
            self.rank[(scene_id, k)] = seen[scene_id]
        self.layout, self.region, self.line_width = layout, region, line_width
        self.labels = dict(labels or {})  # scene id -> (SceneLabels or None, note); empty: map labels off
        self._scenes: dict = {}
        self._pool = None

    def __len__(self) -> int:
        return len(self.items)

    def scene_state(self, scene: "Scene"):
        """``(SceneLabels or None, note, tile cameras, pixel windows, line scale)`` of *scene*, cached for the last scenes."""
        if scene.id not in self._scenes:
            layer, note = self.labels.get(scene.id, (None, "map labels off"))
            cams, windows, scale = tile_cameras(scene, self.layout)
            if len(self._scenes) >= 2:
                self._scenes.pop(next(iter(self._scenes)))
            self._scenes[scene.id] = (layer, note, cams, windows, scale)
        return self._scenes[scene.id]

    def __getitem__(self, position: int) -> _Item:
        cv2 = _deps.cv2()
        scene_id, index = self.items[position]
        scene = self.ds.scene(scene_id)
        frame = scene.frame(index)
        layer, note, cams, windows, scale = self.scene_state(scene)

        # Images go to the renderer as one numpy array (pickled through the pipe): torch tensors would
        # each get a shared-memory segment, which costs more than the copy.
        def load(camera: str):
            data = frame.image_bytes(camera, missing="none")
            if data is None:
                return camera, None
            _forget(scene.files.storage, camera, index)
            x0, y0, x1, y1 = windows[camera]
            tile = self.layout.tiles[camera]
            reduce = decode_reduction((x1 - x0, y1 - y0), (tile.w, tile.h))
            array = _decode_reduced(data, reduce)
            array = array[y0 // reduce:-(-y1 // reduce), x0 // reduce:-(-x1 // reduce)]
            if array.shape[1::-1] != (tile.w, tile.h):
                array = cv2.resize(array, (tile.w, tile.h), interpolation=cv2.INTER_AREA)
            return camera, np.ascontiguousarray(array)

        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(DECODE_THREADS)
        loaded = self._pool.map(load, list(cams))
        labels = None
        if layer is not None:
            pose = _pose_or_none(frame)
            labels = layer(pose) if pose is not None else None
            note = "" if pose is not None else "no pose at this frame"
        images = {c: im for c, im in loaded if im is not None}
        names = [c for c in GRID_CAMERAS if c in images]
        arrays = [images[c] for c in names]
        if arrays and all(a.shape == arrays[0].shape for a in arrays):
            arrays = np.stack(arrays)
        decoded = (names, arrays if names else None)
        plan = plan_frame(self.layout, self.region, labels, cams, list(images), scale, self.line_width,
                          layer.hidden_ids if layer is not None else ())
        plan.text = header_text(self.layout, f"frame {index}  ({self.rank[(scene_id, index)]}/{self.counts[scene_id]})"
                                             f"   {index / SOURCE_FPS:5.1f} s   {self.speed:g}x")
        legend = None
        if self.layout.legend_mode == "compact":
            present = (layer.classes, layer.te_classes, layer.edges) if layer is not None else ((), (), False)
            legend = legend_entries(present, "cams+bev" if self.layout.bev is not None else "cams")
        return _Item(position, scene_id, index, decoded, plan.pack(), note, legend)


def _forget(storage, camera: str, index: int) -> None:
    """Drop a camera image's pages from the page cache (each image is read once)."""
    if not hasattr(os, "posix_fadvise"):  # pragma: no cover
        return
    from kitscenes._layout import sensor_member

    name = sensor_member(camera, index)
    try:
        entry = storage.packed(name) if hasattr(storage, "packed") else None
        if entry is not None:  # one range of the scene's zip
            component, offset, size, _ = entry
            os.posix_fadvise(storage._handle(component), offset, size, os.POSIX_FADV_DONTNEED)
            return
        fd = os.open(storage.root / name, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):  # best effort
        pass


def _short_reason(err: BaseException) -> str:
    text = str(err)
    if "ships no maps" in text:
        return "no map labels: " + text.split(":")[-1].strip()
    return f"no map labels ({type(err).__name__})"


def _worker_init(_):
    import torch

    _deps.cv2().setNumThreads(1)
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Drawing a frame
# ---------------------------------------------------------------------------


def _text(lines: Sequence[str], size: tuple[int, int], scale: float, colour=(255, 255, 255), bg=(0, 0, 0),
          thick: int = 2, align: str = "left") -> np.ndarray:
    """Text on a (h, w, 3) uint8 RGB image (OpenCV's Hershey font)."""
    cv2 = _deps.cv2()
    w, h = size
    img = np.full((h, w, 3), bg, np.uint8)
    y = 0
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        y += th + base + max(2, int(6 * scale))
        x = max(4, w - tw - int(12 * scale)) if align == "right" else int(12 * scale)
        cv2.putText(img, line, (x, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)
    return img


#: Line widths in the camera tiles follow the tile's scale (tile width / image width), but not below this
#: (thin lines vanish at 1080p).
MIN_LINE_SCALE = 0.4
#: Icon size in the tiles relative to :func:`~kitscenes.visualization.draw_map` (64 px at 30 m in a full-size
#: image), and the icons' opacity there.
ICON_SCALE, ICON_ALPHA = 1.0, 0.6
#: Threads per reader decoding the cameras of a frame.
DECODE_THREADS = 4
#: Colour of the BEV's edges (traffic element -> traffic element, -> its lane).
EDGE_GREY = (169, 169, 169)


@dataclasses.dataclass
class FramePlan:
    """What :class:`FrameRenderer` draws into one frame, prepared on the CPU (in the reader workers).

    Attributes:
        cams: Camera segments in 3D ``(P0, P1, attrs)`` (base frame), or None.
        projections: (C, 4, 4) float32 base frame -> tile pixels of the shown cameras (row 2: depth).
        offsets: (C, 2) tile positions in the frame; ``rects``: (C, 4) inclusive tile rectangles.
        bev: BEV segments in frame pixels ``(P0, P1, attrs)``, or None.
        icons: ``("paste", te_type, height, x0, y0, rect, alpha)`` (a cached icon) or
            ``("patch", rgba, x0, y0, rect, alpha)`` (rect exclusive), in drawing order.
        icon_pixels: The icons as sparse pixels (:func:`icon_pixels`), drawn in one blend.
        text: The frame's part of the header (RGB), or None.
    """

    cams: Optional[tuple] = None
    projections: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None
    rects: Optional[np.ndarray] = None
    bev: Optional[tuple] = None
    icons: list = dataclasses.field(default_factory=list)
    icon_pixels: Optional[tuple] = None
    text: Optional[np.ndarray] = None
    packed: Optional[tuple] = None

    def arrays(self) -> dict:
        """The arrays the renderer uploads, by name."""
        out = {}
        if self.cams is not None:
            out.update(zip(("cam_P0", "cam_P1", "cam_colour", "cam_floats", "cam_ints"), self.cams))
            out.update(projections=self.projections, offsets=self.offsets, rects=self.rects)
        if self.bev is not None:
            out.update(zip(("bev_P0", "bev_P1", "bev_colour", "bev_floats", "bev_ints"), self.bev))
        if self.icon_pixels is not None:
            out.update(zip(("icon_flat", "icon_colour", "icon_alpha"), self.icon_pixels))
        if self.text is not None:
            out["text"] = self.text
        return out

    def pack(self) -> "FramePlan":
        """All arrays in one byte buffer (``packed``: buffer, index, groups), so that the renderer uploads a frame
        in one asynchronous copy (many small copies from pageable memory each wait for the GPU)."""
        arrays = {k: np.ascontiguousarray(v) for k, v in self.arrays().items()}
        index, offset = {}, 0
        for name, a in arrays.items():
            index[name] = (offset, a.dtype.str, a.shape)
            offset += -(-a.nbytes // 8) * 8
        blob = np.zeros(max(offset, 8), np.uint8)
        for name, a in arrays.items():
            start = index[name][0]
            blob[start:start + a.nbytes] = a.reshape(-1).view(np.uint8)
        groups = tuple(int(part[4][:, 1].max()) + 1 if part is not None else 0 for part in (self.cams, self.bev))
        self.packed = (blob, index, groups)
        self.cams = self.projections = self.offsets = self.rects = self.bev = self.icon_pixels = self.text = None
        self.icons = []
        return self


def tile_cameras(scene: "Scene", layout: Layout) -> tuple[dict, dict, float]:
    """The ring cameras of *scene* as their tiles show them: ``(cameras, windows, scale)``.

    ``cameras``: camera -> the camera with its intrinsics for the tile (the
    image cut to its window, then scaled per axis to the tile, pixel centres
    kept); ``windows``: camera -> the window in full-size pixels
    (:func:`crop_window`); ``scale``: tile pixels per image pixel (of the
    front camera), for the line widths. A mirrored camera
    (:attr:`Layout.mirrored`) projects to ``u' = tile width - 1 - u``.
    """
    from kitscenes.decode import scale_intrinsics

    cams, windows, scales = {}, {}, {}
    for name in GRID_CAMERAS:
        cam = scene.rig.cameras.get(name)
        if cam is None or not cam.image_size[0] or name not in layout.tiles:
            continue
        x0, y0, x1, y1 = window = crop_window(cam.image_size, layout.windows[name])
        tile = layout.tiles[name]
        K = np.array(cam.K, dtype=np.float64)
        K[0, 2] -= x0
        K[1, 2] -= y0
        sx, sy = tile.w / (x1 - x0), tile.h / (y1 - y0)
        K = scale_intrinsics(K, (sx, sy))
        if name in layout.mirrored:  # u -> (w - 1) - u, pixel centres on integers
            K[0] = (tile.w - 1) * K[2] - K[0]
        cams[name] = dataclasses.replace(cam, K=K, image_size=(tile.w, tile.h))
        windows[name] = window
        scales[name] = sx
    return cams, windows, scales.get("camera_ring_front", next(iter(scales.values()), 1.0))


def plan_frame(layout: Layout, region: "Region", labels: Optional["MapLabels"], cameras: dict, shown: Sequence[str],
               scale: float, line_width: float = 1.0, hidden_ids: Iterable[int] = ()) -> FramePlan:
    """The segments and icons of a frame (numpy; :class:`FrameRenderer` draws them).

    Args:
        cameras: Tile cameras (:func:`tile_cameras`).
        shown: Cameras with an image in this frame.
        scale: Tile pixels per image pixel (the line widths of the styles are for full-size images).
        line_width: Multiplies the line widths in the camera tiles and the BEV.
    """
    plan = FramePlan()
    if labels is None:
        return plan
    hidden = frozenset(hidden_ids)
    hulls = {name: [style.crossing_outline(line) for line in labels.lines.get(name, [])]
             for name in style.CROSSING_TYPES}
    shown = [c for c in GRID_CAMERAS if c in shown and c in cameras]
    if shown:
        b = _SegmentBuilder()
        _camera_lines(b, labels, hidden, scale, line_width, hulls)
        if b.P0:
            plan.cams = b.arrays()
            Ms, offs, rects = [], [], []
            for name in shown:
                cam = cameras[name]
                M = np.eye(4)
                M[:3] = np.asarray(cam.K) @ np.asarray(cam.T_sensor_base)[:3]
                Ms.append(M)
                r = layout.tiles[name]
                offs.append((r.x, r.y))
                rects.append((r.x, r.y, r.x1 - 1, r.y1 - 1))
            plan.projections = np.stack(Ms).astype(np.float32)
            plan.offsets = np.asarray(offs, np.float32)
            plan.rects = np.asarray(rects, np.float32)
        _camera_icons(plan.icons, layout, labels, shown, cameras, hidden, scale)
    if layout.bev is not None:
        px, ppm = bev_transform(layout, region)
        b = _SegmentBuilder()
        _bev_lines(b, labels, hidden, px, ppm, line_width, hulls)
        if b.P0:
            plan.bev = b.arrays()
        _bev_icons(plan.icons, layout, labels, hidden, px, ppm)
    plan.icon_pixels = icon_pixels(plan.icons, layout.size)
    return plan


@functools.lru_cache(maxsize=512)
def _sized_icon(te_type: str, height: int) -> Optional[np.ndarray]:
    """The icon of *te_type* resized to *height* (area interpolation), RGBA."""
    rgba = style.icon_rgba(te_type)
    if rgba is None:
        return None
    width = max(1, int(round(rgba.shape[1] * height / rgba.shape[0])))
    return _deps.cv2().resize(np.ascontiguousarray(rgba), (width, max(1, int(height))), interpolation=_deps.cv2().INTER_AREA)


def icon_pixels(icons: Sequence[tuple], size: tuple[int, int]) -> Optional[tuple]:
    """The icon jobs of a frame as sparse pixels ``(flat index, RGB, alpha 0-255)``; overlaps: the later icon.

    One blend on the device for all icons of a frame, instead of a few kernels per icon.
    """
    W, H = size
    flats, colours, alphas = [], [], []
    for job in icons:
        if job[0] == "paste":
            _, te_type, height, x0, y0, rect, alpha = job
            patch = _sized_icon(te_type, int(height))
        else:
            _, patch, x0, y0, rect, alpha = job
        if patch is None:
            continue
        h, w = patch.shape[:2]
        X0, Y0, X1, Y1 = max(rect[0], 0, x0), max(rect[1], 0, y0), min(rect[2], W, x0 + w), min(rect[3], H, y0 + h)
        if X1 <= X0 or Y1 <= Y0:
            continue
        part = patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        a = part[..., 3].astype(np.float32) * alpha
        ys, xs = np.nonzero(a >= 1.0)
        if not len(ys):
            continue
        flats.append((ys + Y0) * W + (xs + X0))
        colours.append(part[ys, xs, :3])
        alphas.append(np.round(a[ys, xs]).astype(np.uint8))
    if not flats:
        return None
    flat = np.concatenate(flats)
    _, last = np.unique(flat[::-1], return_index=True)  # the last icon drawn at a pixel
    keep = len(flat) - 1 - last
    return flat[keep].astype(np.int64), np.concatenate(colours)[keep], np.concatenate(alphas)[keep]


def bev_transform(layout: Layout, region: "Region"):
    """``(px, ppm)``: ego (x forward, y left) -> BEV pixels in the frame, and pixels per metre."""
    b, r = layout.bev, region
    length, width = r.ahead + r.behind, r.left + r.right
    ppm = min(b.w / width, b.h / length)
    ox = b.x + (b.w - width * ppm) / 2
    oy = b.y + (b.h - length * ppm) / 2

    def px(pts):
        pts = np.asarray(pts, np.float64)
        return np.stack([ox + (r.left - pts[..., 1]) * ppm, oy + (r.ahead - pts[..., 0]) * ppm], -1)
    return px, ppm


def _camera_icons(icons: list, layout: Layout, labels, shown, cameras, hidden, scale) -> None:
    """Icons of signs and lights beside their box, road markings laid along their line (as ``draw_map``).

    All elements are projected into a camera at once; only those partly behind it go through
    :func:`kitscenes.projection.project_lines` (the near-plane clipping), those off the tile are skipped.
    """
    from kitscenes.projection import MIN_DEPTH, project_lines

    icon_scale = max(scale, MIN_LINE_SCALE) * ICON_SCALE
    elements = []
    for te_type, items in labels.traffic_elements.items():
        if te_type == "StopLine" or not items or style.icon_rgba(te_type) is None:
            continue
        elements += [(te_type, np.asarray(e, np.float64)) for e, i in zip(items, labels.traffic_element_ids[te_type])
                     if i not in hidden and len(e) >= 2]
    if not elements:
        return
    vertices = np.concatenate([e for _, e in elements])[:, :3]
    starts = np.cumsum([0] + [len(e) for _, e in elements])[:-1]
    margin = 64 * icon_scale * 2.5
    for name in shown:
        cam, r = cameras[name], layout.tiles[name]
        rect = (r.x, r.y, r.x1, r.y1)
        T = np.asarray(cam.T_sensor_base)
        q = vertices @ T[:3, :3].T + T[:3, 3]
        depth = q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = q @ np.asarray(cam.K).T
            uv = p[:, :2] / depth[:, None]
        min_depth = np.minimum.reduceat(depth, starts)
        max_depth = np.maximum.reduceat(depth, starts)
        front = np.where(depth[:, None] >= MIN_DEPTH, uv, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.fmin.reduceat(front, starts)
            hi = np.fmax.reduceat(front, starts)
        for k, (te_type, element) in enumerate(elements):
            if max_depth[k] < MIN_DEPTH:
                continue
            if min_depth[k] >= MIN_DEPTH:  # wholly in front: cull by the tile, then as project_lines would
                if hi[k, 0] < -margin or lo[k, 0] > r.w + margin or hi[k, 1] < -margin or lo[k, 1] > r.h + margin:
                    continue
                n = len(element)
                pieces = [(uv[starts[k]:starts[k] + n], depth[starts[k]:starts[k] + n])]
            else:
                pieces = project_lines(cam, [element], max_depth=150.0)[0]
            for piece_uv, piece_depth in pieces:
                _camera_icon(icons, te_type, piece_uv + (r.x, r.y), piece_depth, rect, icon_scale)


def _camera_icon(icons: list, te_type: str, uv: np.ndarray, depth: np.ndarray, rect, icon_scale: float) -> None:
    if not np.isfinite(uv).all():
        return
    rgba = style.icon_rgba(te_type)
    if style.is_upright(te_type):  # beside the box, 64 px at 30 m and nearer (full size), 16 px at most far
        height = max(4, int(np.clip(64 * min(2.0, 30.0 / max(1.0, float(depth.mean()))), 16, 64) * icon_scale))
        width = max(1, int(round(rgba.shape[1] * height / rgba.shape[0])))
        cx, cy = int(uv[:, 0].max()) + height // 2 + 4, int(uv[:, 1].mean())
        x0, y0 = cx - width // 2, cy - height // 2
        if x0 < rect[2] and y0 < rect[3] and x0 + width > rect[0] and y0 + height > rect[1]:
            icons.append(("paste", te_type, height, x0, y0, rect, ICON_ALPHA))
        return
    a, b = uv[0], uv[-1]  # on the road: the icon (turned) along the line
    length = float(np.linalg.norm(b - a))
    if length <= 5 * icon_scale:
        return
    along = (b - a) / length
    across = np.array([-along[1], along[0]])
    width = max(16 * icon_scale, min(64 * icon_scale, 0.5 * length))
    height = width * rgba.shape[1] / rgba.shape[0]
    c = (a + b) / 2
    quad = np.array([c - across * width / 2 + along * height / 2, c + across * width / 2 + along * height / 2,
                     c + across * width / 2 - along * height / 2, c - across * width / 2 - along * height / 2])
    patch = _warped_icon(te_type, quad, rect)
    if patch is not None:
        icons.append(("patch", patch[0], patch[1], patch[2], rect, ICON_ALPHA))


@functools.lru_cache(maxsize=64)
def _turned_icon(te_type: str) -> Optional[np.ndarray]:
    rgba = style.icon_rgba(te_type)
    return None if rgba is None else np.ascontiguousarray(np.rot90(rgba, k=1))


def _warped_icon(te_type: str, quad: np.ndarray, rect) -> Optional[tuple]:
    """``(rgba patch, x0, y0)``: the (turned) icon warped onto *quad* (as ``reprojection._warp``), or None off *rect*."""
    icon = _turned_icon(te_type)
    if icon is None or not np.isfinite(quad).all():
        return None
    lo = np.maximum(np.floor(quad.min(axis=0)).astype(int) - 2, rect[:2])
    hi = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + 2, rect[2:])
    if (hi <= lo).any():
        return None
    cv2 = _deps.cv2()
    h, w = icon.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, np.float32(quad - lo))
    patch = cv2.warpPerspective(icon, matrix, (int(hi[0] - lo[0]), int(hi[1] - lo[1])))
    return patch, int(lo[0]), int(lo[1])


def _bev_icons(icons: list, layout: Layout, labels, hidden, px, ppm: float) -> None:
    """Icons at the elements' centres, 3 m (``plot_top_down``'s ``icon_size``), alpha 0.9."""
    bev = layout.bev
    rect = (bev.x + 3, bev.y, bev.x1, bev.y1)
    size = max(4, int(round(3.0 * ppm)))
    for te_type, items in labels.traffic_elements.items():
        rgba = style.icon_rgba(te_type) if te_type != "StopLine" else None
        if rgba is None:
            continue
        h, w = rgba.shape[:2]
        height = size if h >= w else max(2, int(round(size * h / w)))
        width = max(1, int(round(w * height / h)))
        for element, map_id in zip(items, labels.traffic_element_ids[te_type]):
            if map_id in hidden:
                continue
            cx, cy = px(np.asarray(element)[:, :2].mean(axis=0))
            icons.append(("paste", te_type, height, int(round(cx - width / 2)), int(round(cy - height / 2)), rect, 0.9))


def _right_width(layout: Layout) -> int:
    return int(680 * layout.header.h / 80.0)


def header_text(layout: Layout, text: str) -> np.ndarray:
    """The frame's part of the header (frame number, time, speed), right-aligned (RGB)."""
    font = layout.header.h / 80.0
    return _text([text], (_right_width(layout), layout.header.h), 0.85 * font, thick=max(1, int(round(2 * font))),
                 align="right")


class FrameRenderer:
    """Composes video frames on a torch device: header, camera tiles, labels (:class:`FramePlan`), BEV, legend.

    Args:
        layout: :class:`Layout` of the frame.
        device: torch device (``cuda:K`` or ``cpu``).
        region: The labelled region (the BEV shows it).
    """

    def __init__(self, layout: Layout, device, region: "Region") -> None:
        import torch

        from kitscenes.visualization import _gpu_draw

        self.torch, self.gd = torch, _gpu_draw
        self.layout, self.device, self.region = layout, torch.device(device), region
        W, H = layout.size
        self.font = layout.header.h / 80.0
        self.thick = max(1, int(round(2 * self.font)))
        self.right_w = _right_width(layout)
        base = torch.zeros((3, H, W), dtype=torch.uint8, device=self.device)
        if layout.legend is not None:
            g = layout.legend
            base[:, g.y:g.y1, g.x:g.x1] = 24
        if layout.bev is not None:
            b = layout.bev
            base[:, b.y:b.y1, b.x:b.x1] = 255
            base[:, b.y:H, b.x:b.x + 2] = 80  # separator
        self.base = self.scene_base = base
        self._legends: dict = {}
        self.captions = {c: self._tensor(_text([CAPTIONS[c]], (int(200 * self.font), int(44 * self.font)),
                                               0.85 * self.font, thick=self.thick)) for c in GRID_CAMERAS}
        self.missing = self._tensor(_text(["no image"], (int(200 * self.font), int(50 * self.font)), 0.9 * self.font,
                                          colour=(160, 160, 160), thick=self.thick))
        self._car = self._car_icon() if layout.bev is not None else None

    # -- helpers --------------------------------------------------------------------

    def _tensor(self, img: np.ndarray) -> "torch.Tensor":
        img = np.ascontiguousarray(img) if img.flags.writeable else np.array(img)
        return self.torch.from_numpy(img).to(self.device).permute(2, 0, 1)

    def _paste_np(self, frame, img: np.ndarray, x: int, y: int) -> None:
        self._paste(frame, self._tensor(img), x, y)

    @staticmethod
    def _paste(frame, t, x: int, y: int) -> None:
        h, w = t.shape[1:]
        h, w = min(h, frame.shape[1] - y), min(w, frame.shape[2] - x)
        if h > 0 and w > 0:
            frame[:, y:y + h, x:x + w] = t[:, :h, :w]

    def _car_icon(self):
        """The ego car of the top-down view, at its size there (``car.png``)."""
        path = style.car_image()
        if path is None:
            return None
        from PIL import Image

        px, ppm = bev_transform(self.layout, self.region)
        image = Image.open(path).convert("RGBA").rotate(-90, expand=True)
        w, h = max(2, int(round(2 * image.width / 600 * ppm))), max(2, int(round(2 * image.height / 600 * ppm)))
        rgba = np.asarray(image.resize((w, h), Image.LANCZOS))
        cx, cy = px(np.zeros(2))
        return self._tensor(rgba), int(round(cx - w / 2)), int(round(cy - h / 2))

    # -- per scene ------------------------------------------------------------------

    def legend(self, entries: Sequence[tuple[str, str]]) -> "torch.Tensor":
        """The legend image of *entries* on the device (cached)."""
        key = tuple(entries)
        if key not in self._legends:
            g = self.layout.legend
            if len(self._legends) > 32:
                self._legends.clear()
            self._legends[key] = self._tensor(_legend_image(g.w - (2 if self.layout.bev is not None else 0), g.h,
                                                            self.font, entries, arrows=self.layout.bev is not None))
        return self._legends[key]

    def set_scene(self, scene_id: str, split: Optional[str], note: str = "",
                  legend: Optional[Sequence[tuple[str, str]]] = None) -> None:
        """The header of a scene (from the scene table) and its legend (:func:`legend_entries`; None: every
        class) on the frame background."""
        from kitscenes.splits import scene_info

        info = scene_info(scene_id)
        fields = ["KITScenes", split or "-", f"scene {scene_id}"]
        if info is not None:
            fields.append(", ".join(x for x in (info.city, info.recording_week) if x))
            fields.append(", ".join(x for x in (info.time_of_day, info.light) if x))
            if info.weather:
                fields.append(f"{info.weather}" + (f", road {info.road_surface}" if info.road_surface else ""))
        r = self.region
        fields.append(note or f"labels {r.ahead + r.behind:g} x {r.left + r.right:g} m")
        h = self.layout.header
        self.scene_base = self.base.clone()
        self._paste_np(self.scene_base, _text(["   |   ".join(f for f in fields if f)], (h.w - self.right_w, h.h),
                                               0.85 * self.font, thick=self.thick), 0, 0)
        g = self.layout.legend
        if g is not None:
            entries = legend if legend is not None else legend_entries(None, "cams+bev" if self.layout.bev else "cams")
            self._paste(self.scene_base, self.legend(entries), g.x + (2 if self.layout.bev is not None else 0), g.y)
        b = self.layout.bev
        if note and b is not None:  # say in the empty BEV why it is empty
            lines = [part.strip() for part in note.replace("no map labels:", "no map labels:|").split("|") if part.strip()]
            text = _text(lines, (b.w - 20, int(90 * self.font)), 0.7 * self.font, colour=(90, 90, 90),
                         bg=(255, 255, 255), thick=self.thick)
            self._paste_np(self.scene_base, text, b.x + 10, b.y + b.h // 3)

    # -- per frame ------------------------------------------------------------------

    def render(self, tiles: dict, plan: FramePlan) -> "torch.Tensor":
        """One frame (3, H, W) uint8 on the device.

        Args:
            tiles: camera -> (3, th, tw) uint8 tile on the device (missing cameras absent).
            plan: The labels and the header text to draw (:func:`plan_frame`), packed (:meth:`FramePlan.pack`).
        """
        L = self.layout
        f = self.scene_base.clone()
        for name, rect in L.tiles.items():
            tile = tiles.get(name)
            if tile is not None:
                f[:, rect.y:rect.y1, rect.x:rect.x1] = tile.flip(-1) if name in L.mirrored else tile
            else:
                self._paste(f, self.missing, rect.x + rect.w // 3, rect.y + rect.h // 2)
        arrays, (cam_groups, bev_groups) = self._upload(plan.packed)
        n_cams = arrays["projections"].shape[0] if "projections" in arrays else 0
        self.gd.draw_segments(f, self.gd.Segments.cat([self._camera_segments(arrays, cam_groups), self._bev_segments(arrays)],
                                                      [n_cams * cam_groups, bev_groups]))
        if "icon_flat" in arrays:  # all icons in one blend
            flat, colour, alpha = arrays["icon_flat"], arrays["icon_colour"], arrays["icon_alpha"]
            view = f.view(3, -1)
            a = alpha.float()[None] / 255.0
            view[:, flat] = (a * colour.t().float() + (1 - a) * view[:, flat].float()).round_().to(self.torch.uint8)
        if self._car is not None:
            icon, x0, y0 = self._car
            b = L.bev
            self.gd.paste_rgba(f, icon, x0, y0, 1.0, (b.x, b.y, b.x1, b.y1))
        for name, rect in L.tiles.items():
            self._paste(f, self.captions[name], rect.x, rect.y)
        if "text" in arrays:
            text = arrays["text"].permute(2, 0, 1)
            self._paste(f, text, L.header.w - text.shape[2], 0)
        return f

    def _upload(self, packed: tuple) -> tuple[dict, tuple]:
        """The plan's arrays on the device (one copy, through pinned memory on CUDA), and the number of
        polylines (groups) of its camera and BEV segments."""
        torch = self.torch
        blob, index, groups = packed
        t = torch.from_numpy(blob)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        out = {}
        for name, (offset, dtype, shape) in index.items():
            nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            out[name] = t[offset:offset + nbytes].view(_torch_dtype(np.dtype(dtype).str)).reshape(shape)
        return out, groups

    def _segments(self, P0, P1, colour, floats, ints, bounds, d0, d1, depth_scaled) -> "Segments":
        F = floats
        return self.gd.Segments(p0=P0, p1=P1, colour=colour, radius=F[:, 0], alpha=F[:, 1], dash=F[:, 2], gap=F[:, 3],
                                min_dash=F[:, 4], min_gap=F[:, 5], priority=ints[:, 0], group=ints[:, 1],
                                bounds=bounds, depth_scaled=depth_scaled, d0=d0, d1=d1)

    def _camera_segments(self, arrays: dict, n_groups: int):
        """The 3D segments projected into every shown tile at once.

        Every (camera, segment) pair is kept (sizes known on the host: no wait for the GPU); a segment
        wholly behind a camera becomes a point far outside every rectangle (length 0: it adds nothing to
        the dash phase; :func:`~kitscenes.visualization._gpu_draw.draw_segments` drops it).
        """
        if "cam_P0" not in arrays:
            return None
        torch, dev = self.torch, self.device
        from kitscenes.projection import MIN_DEPTH

        P0, P1, colour, floats, ints = (arrays[k] for k in ("cam_P0", "cam_P1", "cam_colour", "cam_floats", "cam_ints"))
        p0, p1, d0, d1, keep = self.gd.project_segments(P0, P1, arrays["projections"], near=MIN_DEPTH)
        C, S = keep.shape
        k = keep.reshape(-1)
        src = torch.arange(S, device=dev).repeat(C)  # the 3D segment of each (camera, segment)
        cam = torch.arange(C, device=dev).repeat_interleave(S)
        off = arrays["offsets"][cam]
        ints = ints[src].clone()
        ints[:, 1] += cam * n_groups  # dashes run per camera and polyline
        far = torch.tensor([-1e7, -1e7], device=dev)
        a = torch.where(k[:, None], p0.reshape(-1, 2) + off, far)
        b = torch.where(k[:, None], p1.reshape(-1, 2) + off, far)
        return self._segments(a, b, colour[src], floats[src], ints, arrays["rects"][cam], d0.reshape(-1), d1.reshape(-1),
                              torch.ones(C * S, dtype=torch.bool, device=dev))

    def _bev_segments(self, arrays: dict):
        if "bev_P0" not in arrays:
            return None
        torch, dev = self.torch, self.device
        P0, P1, colour, floats, ints = (arrays[k] for k in ("bev_P0", "bev_P1", "bev_colour", "bev_floats", "bev_ints"))
        b = self.layout.bev
        n = len(P0)
        bounds = torch.tensor([b.x + 3, b.y, b.x1 - 1, b.y1 - 1], dtype=torch.float32, device=dev).expand(n, 4)
        depth = torch.full((n,), 25.0, device=dev)
        return self._segments(P0, P1, colour, floats, ints, bounds, depth, depth,
                              torch.zeros(n, dtype=torch.bool, device=dev))


class _SegmentBuilder:
    """Collects polylines with their attributes into segment arrays (numpy)."""

    def __init__(self) -> None:
        self.P0, self.P1, self.attrs = [], [], []
        self.group = 0

    def add(self, polylines, *, colour, width: float, alpha: float = 1.0, priority: int = 0, dash: float = 0.0,
            gap: float = 0.0, min_dash: float = 0.0, min_gap: float = 0.0, closed: bool = False) -> None:
        from kitscenes.visualization import _gpu_draw as gd

        if not len(polylines):
            return
        P0, P1, idx = gd.polylines_to_segments(polylines, closed=closed)
        if not len(P0):
            return
        n = len(P0)
        self.P0.append(P0)
        self.P1.append(P1)
        self.attrs.append(dict(
            colour=np.repeat(np.asarray(colour, np.uint8)[None], n, 0),
            radius=np.full(n, max(0.5, width / 2), np.float32), alpha=np.full(n, alpha, np.float32),
            priority=np.full(n, priority, np.int64), group=idx + self.group,
            dash=np.full(n, dash, np.float32), gap=np.full(n, gap, np.float32),
            min_dash=np.full(n, min_dash, np.float32), min_gap=np.full(n, min_gap, np.float32)))
        self.group += int(idx.max()) + 1

    def arrays(self) -> tuple:
        """``(P0, P1, colour (S, 3) uint8, floats (S, 6), ints (S, 2))``: floats are radius, alpha, dash, gap,
        min_dash, min_gap; ints priority and group (few arrays: few host-to-device copies)."""
        a = {k: np.concatenate([x[k] for x in self.attrs]) for k in self.attrs[0]}
        floats = np.stack([a["radius"], a["alpha"], a["dash"], a["gap"], a["min_dash"], a["min_gap"]], 1)
        ints = np.stack([a["priority"], a["group"]], 1)
        return np.concatenate(self.P0), np.concatenate(self.P1), a["colour"], floats.astype(np.float32), ints


def _camera_lines(b: _SegmentBuilder, labels, hidden, scale: float, line_width: float, hulls: dict) -> None:
    """The lines of :func:`kitscenes.visualization.reprojection.draw_map`, in its order and styles."""
    ws = max(scale, MIN_LINE_SCALE) * line_width  # line widths
    names = [n for n in labels.lines if n not in style.CENTERLINE_TYPES] + \
            [n for n in labels.lines if n in style.CENTERLINE_TYPES]
    for rank, name in enumerate(names):
        items = labels.lines[name]
        if not items:
            continue
        closed = name in style.CROSSING_TYPES
        shapes = hulls[name] if closed else items
        # bike centerlines in the centerlines' dashes (their own short dots break up into specks)
        ls = style.LINE_STYLES.get("Centerline" if name == "BikeCenterline" else name, style.DEFAULT_LINE_STYLE)
        width, under = max(1.0, ls.width * ws), max(1.0, ls.under * ws)
        dash = dict(dash=ls.dash * scale, gap=ls.gap * scale, min_dash=max(6, math.ceil(2.0 * width)),
                    min_gap=max(4, math.ceil(1.5 * width))) if ls.dash else {}
        if ls.under:
            b.add(shapes, colour=GREY, width=under, priority=2 * rank, **dash)
        b.add(shapes, colour=style.colour(name), width=width, alpha=ls.alpha, priority=2 * rank + 1, **dash)
    base = 2 * len(names) + 2
    for name, items in labels.traffic_elements.items():
        keep = [e for e, i in zip(items, labels.traffic_element_ids[name]) if i not in hidden]
        if not keep:
            continue
        if name == "StopLine":
            b.add(keep, colour=RED, width=max(1.0, 5 * ws), priority=base)
        elif style.is_upright(name):
            b.add(keep, colour=YELLOW, width=max(1.0, 3 * ws), priority=base + 1, closed=True)
        else:
            b.add(keep, colour=YELLOW, width=max(1.0, 3 * ws), priority=base + 1)


def _bev_lines(b: _SegmentBuilder, labels, hidden, px, ppm: float, line_width: float, hulls: dict) -> None:
    """The lines of :func:`kitscenes.visualization.plot_top_down` (its widths in points, 1 pt = 1/12 m there)."""
    pt = ppm / 12.0 * line_width  # pixels per matplotlib point at the top-down view's scale

    def add(lines, colour, lw, z, order, *, alpha=1.0, pattern=None, closed=False):
        width = max(1.0, lw * pt)
        dash = {}
        if pattern is not None:  # matplotlib dash pattern in units of the line width, butt caps -> round caps
            on, off = pattern[0] * lw * pt, pattern[1] * lw * pt
            dash = dict(dash=max(0.5, on - width * 0.8), gap=off + width * 0.8)
        b.add([px(np.asarray(line)[:, :2]) for line in lines], colour=colour, width=width, alpha=alpha,
              priority=100 * z + order, closed=closed, **dash)

    lines = labels.lines
    order = 0
    for name in style.BORDER_TYPES:
        add(lines.get(name, []), style.colour(name), 7.0, 5, order)
        order += 1
    for name in style.DIVIDER_TYPES:
        items = lines.get(name, [])
        if name == "Dashed":
            add(items, GREY, 12.0, 5, order, pattern=(5, 5))
        elif name in ("Solid", "SolidSolid"):
            add(items, GREY, 12.0, 5, order)
        add(items, style.colour(name), 3.0, 5, order + 1)
        order += 2
    add(lines.get("Centerline", []), style.colour("Centerline"), 2.0, 4, 0, pattern=(3.7, 1.6))
    add(lines.get("BikeCenterline", []), style.colour("BikeCenterline"), 2.0, 4, 1, pattern=(3.7, 1.6))
    from kitscenes.visualization import _gpu_draw as gd

    for k, name in enumerate(("Centerline", "BikeCenterline")):  # the driving direction
        tips, backs = direction_heads(lines.get(name, []), labels.region)
        if len(tips):
            heads0, heads1 = gd.arrow_heads([np.stack([p, q]) for p, q in zip(px(backs), px(tips))],
                                            max(4.0, DIRECTION_HEAD_M * ppm))
            b.add([np.stack([p, q]) for p, q in zip(heads0, heads1)], colour=style.colour(name),
                  width=max(1.5, 2.5 * pt), priority=400 + k)
    for k, name in enumerate(style.CROSSING_TYPES):
        add(hulls[name], style.colour(name), 5.0, 6, k)
    stops = [e for e, i in zip(labels.traffic_elements.get("StopLine", []), labels.traffic_element_ids.get("StopLine", []))
             if i not in hidden]
    add(stops, RED, 4.0, 7, 0)
    # edges: traffic element -> traffic element, traffic element -> its lane (plot_top_down._edges)
    r = labels.region if labels.region is not None else None
    span = 2 * max(r.ahead, r.behind, r.left, r.right) if r is not None else 120.0
    longest = 0.3 * span

    def centre(te_type, index):
        items = labels.traffic_elements.get(te_type, [])
        if index >= len(items) or labels.traffic_element_ids[te_type][index] in hidden:
            return None
        return np.asarray(items[index])[:, :2].mean(axis=0)

    edges = {3.0: [], 2.0: []}
    for a, i, c, j in labels.te_to_te:
        if a in style.ARROW_TYPES or c in style.ARROW_TYPES:
            continue
        s, t = centre(a, i), centre(c, j)
        if s is not None and t is not None and np.linalg.norm(t - s) <= longest:
            edges[3.0].append(np.stack([s, t]))
    centerlines = lines.get("Centerline", [])
    for te_type, i, k in labels.te_to_centerline:
        s = centre(te_type, i)
        if te_type in style.ARROW_TYPES or s is None or k >= len(centerlines):
            continue
        t = _point_in_band(np.asarray(centerlines[k])[:, :2], s, 3.0, 5.0)
        if np.linalg.norm(t - s) <= longest:
            edges[2.0].append(np.stack([s, t]))
    for k, (lw, items) in enumerate(edges.items()):
        if not items:
            continue
        add(items, EDGE_GREY, lw, 1, k, alpha=0.7)
        heads0, heads1 = gd.arrow_heads([px(e) for e in items], max(6.0, 0.6 * ppm))
        width = max(1.0, lw * pt)
        b.add([np.stack([p, q]) for p, q in zip(heads0, heads1)], colour=EDGE_GREY, width=width, alpha=0.7,
              priority=100 + k)


#: BEV arrowheads on centerlines (the driving direction): one every :data:`DIRECTION_SPACING_M` metres along the
#: line (the first half of that from its start), one at its end unless the region cuts it there, each
#: :data:`DIRECTION_HEAD_M` long.
DIRECTION_SPACING_M, DIRECTION_HEAD_M = 20.0, 1.6


def direction_heads(lines, region=None, spacing: float = DIRECTION_SPACING_M,
                    head: float = DIRECTION_HEAD_M) -> tuple[np.ndarray, np.ndarray]:
    """``(tips, backs)`` (N, 2) in the lines' x-y: arrowheads pointing along each line (its point order: the
    driving direction for centerlines), at ``spacing / 2, 3 spacing / 2, ...`` along it and at its end, unless
    the end lies on the border of *region* (the line goes on outside) or near the last one. ``backs`` are
    *head* behind the tips, on the line's direction there."""
    tips, backs = [], []
    if region is not None:
        lo, hi = np.array([-region.behind, -region.right]), np.array([region.ahead, region.left])
    for line in lines:
        xy = np.asarray(line, np.float64)[:, :2]
        if len(xy) < 2:
            continue
        seg = np.diff(xy, axis=0)
        length = np.linalg.norm(seg, axis=1)
        keep = length > 1e-6
        if not keep.any():
            continue
        xy = np.concatenate([xy[:1], xy[1:][keep]])
        seg, length = seg[keep], length[keep]
        s = np.concatenate([[0.0], np.cumsum(length)])
        at = list(np.arange(spacing / 2, s[-1] - head, spacing))
        cut = region is not None and bool(np.any(np.abs(xy[-1] - lo) < 1e-3) or np.any(np.abs(xy[-1] - hi) < 1e-3))
        if not cut and s[-1] >= head:
            if at and s[-1] - at[-1] <= spacing / 4:
                at.pop()
            at.append(s[-1])
        for a in at:
            k = min(int(np.searchsorted(s, a, side="right")) - 1, len(seg) - 1)
            d = seg[k] / length[k]
            tip = xy[k] + d * (a - s[k])
            tips.append(tip)
            backs.append(tip - d * head)
    if not tips:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(tips), np.asarray(backs)


def _legend_image(w: int, h: int, font: float, entries: Sequence[tuple[str, str]], arrows: bool = False) -> np.ndarray:
    """The legend: *entries* (:func:`legend_entries`) in as many columns as give the largest text, at most
    :data:`LEGEND_SCALE` x *font* (row by row in a wide strip, column by column otherwise). *arrows*: the
    centerline swatches end in an arrowhead (the BEV's driving direction)."""
    cv2 = _deps.cv2()
    img = np.full((h, w, 3), 24, np.uint8)
    if not entries or w < 20 or h < 10:
        return img
    longest = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] for _, t in entries)
    entry_w = 70 + 16 + longest + 34  # swatch, gap, text, space to the next column; per unit of text scale
    best = None
    for cols in range(1, len(entries) + 1):
        rows = -(-len(entries) // cols)
        scale = min(h / (rows * LEGEND_ROW + 2 * LEGEND_PAD), w / (cols * entry_w + 2 * LEGEND_PAD),
                    LEGEND_SCALE * font)
        if best is None or scale > best[0] + 1e-9:
            best = (scale, cols, rows)
    scale, cols, rows = best
    thick = max(1, int(round(1.6 * scale)))
    row = LEGEND_ROW * scale
    swatch = int(round(70 * scale))
    pad = int(round(LEGEND_PAD * scale))
    col_w = (w - 2 * pad) / cols
    y_top = pad + max(0.0, (h - 2 * pad - rows * row) / 2)  # centred vertically
    by_rows = w > 4 * h  # a strip: fill row by row
    for k, (key, text) in enumerate(entries):
        c, r = (k % cols, k // cols) if by_rows else (k // rows, k % rows)
        x, y = int(round(pad + c * col_w)), int(round(y_top + r * row))
        ym = int(round(y + row / 2))
        a, b = (x, ym), (x + swatch, ym)
        if key in style.LINE_COLOURS or key == "StopLine" or (key not in style.ICONS and key != "_edge"):
            colour = RED if key == "StopLine" else style.colour(key)
            ls = style.LINE_STYLES.get(key, style.DEFAULT_LINE_STYLE)
            wide = max(2, int(round(ls.under * 1.6 * scale))) if ls.under else 0
            if wide:
                cv2.line(img, a, b, GREY, wide + 2, cv2.LINE_AA)
            lw = max(1, int(round(max(ls.width, 3) * 1.2 * scale)))
            if ls.dash:
                n = 4
                for i in range(n):
                    xa = x + i * swatch // n
                    cv2.line(img, (xa, ym), (xa + swatch // (2 * n), ym), colour, lw + 1, cv2.LINE_AA)
            else:
                cv2.line(img, a, b, colour, lw + 1, cv2.LINE_AA)
            if arrows and key in style.CENTERLINE_TYPES:
                tip = max(4, int(round(14 * scale)))
                for dy in (-1, 1):
                    cv2.line(img, b, (b[0] - tip, ym + dy * int(round(tip * 0.55))), colour, lw + 1, cv2.LINE_AA)
        elif key == "_edge":
            cv2.arrowedLine(img, a, b, EDGE_GREY, thick, cv2.LINE_AA, tipLength=0.3)
        else:
            icon = style.icon_rgba(key)
            if icon is not None:
                ih = max(4, int(row) - 4)
                iw = max(1, int(round(icon.shape[1] * ih / icon.shape[0])))
                icon = cv2.resize(np.ascontiguousarray(icon), (iw, ih), interpolation=cv2.INTER_AREA)
                x0, y0 = x + (swatch - iw) // 2, ym - ih // 2
                part = img[y0:y0 + ih, x0:x0 + iw]
                al = icon[:part.shape[0], :part.shape[1], 3:4] / 255.0
                part[:] = (al * icon[:part.shape[0], :part.shape[1], :3] + (1 - al) * part).astype(np.uint8)
        (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, (x + swatch + int(round(16 * scale)), ym + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (230, 230, 230), thick, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Rendering scenes in one process
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Job:
    """What one renderer process does."""

    slot: int
    dataset: Any
    scenes: tuple  # ((order, scene_id, frames), ...)
    parts_dir: str
    device: str
    size: tuple
    layout: str
    region: Any
    playback: Playback
    encoder: str
    rate: RateControl
    workers: int
    map_labels: bool
    grouping: str
    line_width: float
    speed: float
    progress: bool
    legend: str = "compact"
    crop_hood: Any = None
    fit: str = "cover"
    rear_row: str = "swap"


def _part_path(parts_dir: str, order: int, split: Optional[str], scene_id: str) -> Path:
    return Path(parts_dir) / f"{order:05d}_{split or 'scene'}_{scene_id}.mp4"


def _run_job(job: _Job) -> None:
    """Render the scenes of *job* into one MP4 each."""
    import torch

    started = time.perf_counter()
    device = torch.device(job.device)
    # a few threads per process: torch and OpenCV would size their pools to all cores, and many
    # processes with hundreds of spinning threads each slow each other down
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
    else:
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    _deps.cv2().setNumThreads(1)
    layout = Layout.compute(job.size, job.layout, job.region, legend=job.legend, crop_hood=job.crop_hood, fit=job.fit,
                            rear_row=job.rear_row)
    items = [(scene_id, k) for _, scene_id, frames in job.scenes for k in frames]
    labels = scene_labels(job.dataset, [(scene_id, frames) for _, scene_id, frames in job.scenes], job.region,
                          job.grouping, processes=max(1, min(16, job.workers))) if job.map_labels else {}
    source = _FrameSource(job.dataset, items, layout=layout, region=job.region, line_width=job.line_width,
                          speed=job.speed, labels=labels)
    if job.workers > 0:
        import multiprocessing

        # fork: a renderer started by torch.multiprocessing.spawn would spawn its readers too (each importing
        # torch and unpickling the dataset)
        context = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        loader = torch.utils.data.DataLoader(source, batch_size=None, shuffle=False, num_workers=job.workers,
                                             collate_fn=_identity, worker_init_fn=_worker_init, prefetch_factor=4,
                                             multiprocessing_context=context)
    else:
        loader = source
    it = iter(loader)  # the workers fork before CUDA starts in this process
    renderer = FrameRenderer(layout, device, job.region)

    def upload_tiles(item: _Item) -> dict:
        tiles = {}
        names, decoded = item.decoded
        if names:  # decoded and resized in the workers: (n, th, tw, 3), or a list of (th, tw, 3)
            if isinstance(decoded, np.ndarray):
                tiles.update(zip(names, _to_device(decoded, device).permute(0, 3, 1, 2)))
            else:
                tiles.update((n, _to_device(a, device).permute(2, 0, 1)) for n, a in zip(names, decoded))
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()  # complete before the drawing thread uses them
        return tiles

    order_of = {scene_id: order for order, scene_id, _ in job.scenes}
    ds = job.dataset
    enc, current, host = None, None, None
    done, last_print, total = 0, started, len(items)
    bar = None
    if job.progress and job.slot == 0:
        from tqdm import tqdm

        bar = tqdm(total=total, desc="video", unit="frame", disable=None)
    try:
        for item, tiles in _decoded(it, upload_tiles):
            if item.scene_id != current:
                if enc is not None:
                    enc.close()
                current = item.scene_id
                split = ds.split_of(current)
                renderer.set_scene(current, split, item.note if item.note.startswith("no map") else "", item.legend)
                enc = Encoder(_part_path(job.parts_dir, order_of[current], split, current), job.size, job.playback.fps,
                              job.encoder, job.rate, gpu=device.index if device.type == "cuda" else None)
            frame = renderer.render(tiles, item.plan)
            nv12 = renderer.gd.rgb_to_nv12(frame)
            if device.type == "cuda":  # into a pinned buffer, asynchronously; the encoder's thread waits for it
                if host is None:
                    host = queue.Queue()
                    for _ in range(Encoder.QUEUE + 3):
                        host.put(torch.empty(nv12.shape, dtype=torch.uint8, pin_memory=True))
                buf = host.get()
                buf.copy_(nv12, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record()
                nv12 = (buf, ready, host)
            else:
                nv12 = nv12.numpy()
            enc.write(nv12)
            done += 1
            now = time.perf_counter()
            if bar is not None:
                bar.update(1)
            elif job.progress and now - last_print > 30:
                last_print = now
                print(f"[video process {job.slot}] {done}/{total} frames, {done / (now - started):.1f} fps", flush=True)
        if enc is not None:
            enc.close()
            enc = None
    finally:
        if bar is not None:
            bar.close()
        if enc is not None:
            enc.abort()


@functools.lru_cache(maxsize=None)
def _torch_dtype(numpy_dtype: str):
    import torch

    return {"|u1": torch.uint8, "<f4": torch.float32, "<f8": torch.float64, "<i8": torch.int64, "<i4": torch.int32,
            "|b1": torch.bool}[numpy_dtype]


def _to_device(array: np.ndarray, device) -> "torch.Tensor":
    """A numpy array on *device* (through pinned memory on CUDA: an asynchronous copy)."""
    import torch

    t = torch.from_numpy(array)
    if device.type == "cuda":
        t = t.pin_memory()
        return t.to(device, non_blocking=True)
    return t


def _decoded(items, decode, depth: int = 3):
    """``(item, tiles)`` for *items*, uploaded by a thread one to *depth* frames ahead of the drawing.

    The uploads release the GIL, so the next frame's tiles are on their way while this one is drawn
    (the drawing is mostly Python, launching small GPU kernels).
    """
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()
    end = object()

    def run():
        try:
            for item in items:
                if stop.is_set():
                    return
                q.put((item, decode(item)))
            q.put(end)
        except BaseException as exc:  # handed to the consumer
            q.put(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        while True:
            got = q.get()
            if got is end:
                return
            if isinstance(got, BaseException):
                raise got
            yield got
    finally:
        stop.set()
        while thread.is_alive():  # unblock a producer waiting on a full queue
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


def _spawned(slot: int, jobs: list) -> None:
    _run_job(jobs[slot])


def _assign(scenes: list, processes: int) -> list:
    """Whole scenes to processes, largest first to the least loaded; each keeps the global order."""
    load, out = [0] * processes, [[] for _ in range(processes)]
    for entry in sorted(scenes, key=lambda e: -len(e[2])):
        p = load.index(min(load))
        out[p].append(entry)
        load[p] += len(entry[2])
    return [sorted(a, key=lambda e: e[0]) for a in out if a]


# ---------------------------------------------------------------------------
# The public function
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VideoResult:
    """What :func:`render_video` made.

    Attributes:
        path: The MP4 of the whole selection (None with ``concat=False``).
        scene_videos: The per-scene MP4s, if kept (``per_scene=True``).
        frames: Video frames.
        fps: Frame rate.
        duration_s: ``frames / fps``.
        seconds: Wall time of the rendering.
        encoder: The ffmpeg encoder used.
        device: Where it ran.
    """

    path: Optional[Path]
    scene_videos: tuple
    frames: int
    fps: Fraction
    duration_s: float
    seconds: float
    encoder: str
    device: str

    @property
    def render_fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device!r}: torch sees no CUDA device (torch {torch.__version__}); use device='cpu'")
    return dev


def render_video(
    data: Union["Dataset", str, Path],
    out: Union[str, Path],
    *,
    splits=None,
    scenes: Optional[Sequence[str]] = None,
    city=None,
    lidar=None,
    where: Optional[dict] = None,
    order: str = "recording",
    max_scenes: Optional[int] = None,
    frames: Union[slice, str, None] = None,
    frame_step: int = 1,
    layout: str = "cams+bev",
    region: Union[str, Sequence[float], "Region", None] = DEFAULT_REGION,
    legend: str = "compact",
    crop_hood: Union[bool, float, dict, None] = False,
    fit: str = "cover",
    rear_row: str = "swap",
    map_labels: bool = True,
    grouping: str = "default",
    line_width: float = 1.0,
    resolution: Union[str, Sequence[int]] = "1080p",
    speed: float = 1.0,
    max_fps: float = 60.0,
    codec: str = "hevc",
    quality: Optional[float] = None,
    bitrate: Union[str, int, None] = None,
    max_bitrate: Union[str, int, None] = None,
    encoder: str = "auto",
    device: str = "auto",
    gpus: int = 1,
    procs_per_gpu: int = 4,
    workers: int = 8,
    per_scene: bool = False,
    concat: bool = True,
    progress: bool = True,
) -> VideoResult:
    """Render the selected scenes into one MP4 (and optionally one per scene).

    Every selected frame becomes one video frame: a header (scene, split, city, time of day, frame), the six
    ring cameras with the map labels drawn in, optionally the labels from above (BEV), and a legend. Each
    scene is rendered into its own MP4; at the end they are joined without re-encoding.

    Args:
        data: A :class:`kitscenes.Dataset` or a dataset root.
        out: The MP4 to write (not inside the dataset root).
        splits, scenes, city, lidar, where: The selection, as :meth:`kitscenes.Dataset.filter`
            (``where``: other scene-table columns, e.g. ``{"weather": "rain"}``).
        order: ``"recording"`` (scene table order; default) or ``"id"``.
        max_scenes: Only the first N scenes of the selection.
        frames: Frames of each scene, a slice or ``"START:STOP[:STEP]"``.
        frame_step: Every k-th of those frames, at the same speed (10 / k fps for ``speed=1``).
        layout: ``"cams+bev"`` (the camera grid, on the right a column with the BEV and the legend
            below it) or ``"cams"`` (the camera grid, the legend in a strip at the bottom).
        region: Labelled region, ``"LENGTHxWIDTH"`` / ``(length, width)`` in metres or a
            :class:`kitscenes.map.Region` (default 120 x 60 m).
        legend: ``"compact"`` (the classes the scene shows, the same for all its frames),
            ``"full"`` (every class) or ``"none"``.
        crop_hood: Cut the ego vehicle off the bottom of the cameras: True
            (:data:`HOOD_FRACTIONS`), one fraction of the image height for all, or
            ``{camera: fraction}``; the rest of the image is shown larger.
        fit: ``"cover"`` (images fill their tiles, cut at the sides) or ``"contain"``
            (whole images, black bars).
        rear_row: The rear row of the grid. Looking back, the vehicle's left is on the image's right:
            ``"swap"`` (default) shows rear right | rear | rear left, one panorama as seen looking back;
            ``"mirror"`` shows rear left | rear | rear right with each image and its labels mirrored (a
            rear-view mirror; icons and text are not mirrored); ``"plain"`` is that order, unmirrored.
        map_labels: Draw the map labels (scenes without a map are rendered without).
        grouping: The ML converter's line type grouping.
        line_width: Multiplies the line widths in the camera tiles and the BEV.
        resolution: ``"1080p"``, ``"2k"``, ``"4k"`` or ``"WxH"``.
        speed: Playback speed: 1 is real time (10 fps, the dataset's rate), 2 plays at 20 fps.
        max_fps: Highest frame rate; above it frames are dropped instead (``speed=8``: every second
            frame at 40 fps).
        codec: ``"hevc"`` or ``"h264"``.
        quality: Constant quality (NVENC ``-cq``, software ``-crf``; lower is better; default
            :data:`DEFAULT_QUALITY`, e.g. 19 for ``hevc_nvenc``).
        bitrate: An average bitrate instead of constant quality, e.g. ``"20M"``.
        max_bitrate: The cap of the constant-quality mode (default :func:`max_bitrate`).
        encoder: ``"auto"`` (NVENC on a GPU, else software), ``"nvenc"`` or ``"software"``.
        device: ``"auto"`` (the first GPU, else the CPU), ``"cuda"``, ``"cuda:K"`` or ``"cpu"``.
        gpus: GPUs to use (``cuda:0`` .. ``cuda:gpus-1``; whole scenes per process).
        procs_per_gpu: Renderer processes per GPU.
        workers: DataLoader processes per renderer (reading, labels, decoding); 0: in the renderer.
        per_scene: Keep one MP4 per scene in ``<out stem>_scenes/`` next to *out*.
        concat: Write the joined MP4 *out* (``False`` with *per_scene*: only the scene videos).
        progress: Show progress.

    Raises:
        ValueError: For a bad option, an empty selection, or an output inside the dataset.
        MissingDependencyError: Without torch, OpenCV, ffmpeg, the ML converter (map labels)
            or a requested NVENC.
    """
    import kitscenes
    from kitscenes._optional import require_torch

    torch = require_torch("kitscenes.visualization.gpu_video", VIDEO_TORCH_MIN_VERSION)
    _deps.cv2()
    _deps.matplotlib()
    ds = data if not isinstance(data, (str, Path)) else kitscenes.open(data)
    out = _deps.guard_output_path(out, roots=[ds.root])
    if out.suffix.lower() != ".mp4":
        raise ValueError(f"out must be an .mp4 file, got {out}")
    if not concat and not per_scene:
        raise ValueError("concat=False writes nothing without per_scene=True")
    size = parse_resolution(resolution)
    region = parse_region(region)
    Layout.compute(size, layout, region, legend=legend, crop_hood=crop_hood, fit=fit, rear_row=rear_row)  # checks
    play = playback(speed, frame_step, max_fps)
    if isinstance(frames, str):
        frames = parse_frames(frames)
    if map_labels:
        from kitscenes.map import has_ml_converter

        if not has_ml_converter():
            from kitscenes.errors import MissingDependencyError

            raise MissingDependencyError(
                "map labels need lanelet2.ml_converter (the KITScenes build of lanelet2: pip install "
                "'kitscenes[map]'); or render without them (map_labels=False, --no-map)")
    dev = _resolve_device(device)
    enc_name = choose_encoder(codec, encoder, gpu=dev.type == "cuda")
    rate = rate_control(enc_name, size, float(play.fps), codec, quality=quality, bitrate=bitrate, max_rate=max_bitrate)
    ids = select_scenes(ds, splits=splits, scene_ids=scenes, city=city, lidar=lidar, where=where, order=order,
                        max_scenes=max_scenes)
    selection = []
    for k, scene_id in enumerate(ids):
        chosen = frame_indices(len(ds.scene(scene_id)), frames, play.step)
        if chosen:
            selection.append((k, scene_id, tuple(chosen)))
    if not selection:
        raise ValueError("the selection has no frames to render")
    out.parent.mkdir(parents=True, exist_ok=True)
    if per_scene:
        parts_dir = _deps.guard_output_path(out.with_name(out.stem + "_scenes"), roots=[ds.root])
        parts_dir.mkdir(parents=True, exist_ok=True)
    else:
        parts_dir = Path(tempfile.mkdtemp(prefix=f".{out.stem}.parts.", dir=out.parent))
    if dev.type == "cuda":
        n_gpu = max(1, min(int(gpus), torch.cuda.device_count())) if dev.index is None else 1
        devices = [f"cuda:{i}" for i in range(n_gpu)] if dev.index is None else [str(dev)]
        n_proc = max(1, min(len(devices) * max(1, int(procs_per_gpu)), len(selection)))
    else:
        devices, n_proc = ["cpu"], 1
    groups = _assign(selection, n_proc)
    jobs = [_Job(slot, ds, tuple(g), str(parts_dir), devices[slot % len(devices)], size, layout, region, play,
                 enc_name, rate, int(workers), map_labels, grouping, float(line_width),
                 float(speed), progress, legend, parse_crop_hood(crop_hood), fit, rear_row)
            for slot, g in enumerate(groups)]
    n_frames = sum(len(f) for _, _, f in selection)
    if progress:
        print(f"kitscenes video: {len(selection)} scene(s), {n_frames} frames at {float(play.fps):g} fps "
              f"({n_frames / float(play.fps):.1f} s of video"
              + (f", every {play.dropped}. frame kept to stay at <= {max_fps:g} fps" if play.dropped > 1 else "")
              + f"), {size[0]}x{size[1]} {layout}, {rate.describe(enc_name)}, "
              f"{len(jobs)} process(es) on {', '.join(sorted(set(j.device for j in jobs)))}", flush=True)
    started = time.perf_counter()
    parts = [_part_path(str(parts_dir), k, ds.split_of(s), s) for k, s, _ in selection]
    try:
        if len(jobs) == 1:
            _run_job(jobs[0])
        else:
            import torch.multiprocessing as tmp

            tmp.spawn(_spawned, args=(jobs,), nprocs=len(jobs), join=True)
        missing = [p for p in parts if not p.is_file()]
        if missing:
            raise RuntimeError(f"{len(missing)} scene video(s) missing, e.g. {missing[0]}")
        path = concat_videos(parts, out, parts_dir) if concat else None
    finally:
        if not per_scene:
            for p in parts_dir.iterdir():
                p.unlink()
            parts_dir.rmdir()
    seconds = time.perf_counter() - started
    result = VideoResult(path, tuple(parts) if per_scene else (), n_frames, play.fps, n_frames / float(play.fps),
                         seconds, enc_name, ", ".join(sorted(set(j.device for j in jobs))))
    if progress:
        print(f"kitscenes video: wrote {path or parts_dir}: {n_frames} frames in {seconds:.1f} s "
              f"({result.render_fps:.1f} frames/s)", flush=True)
    return result


# ---------------------------------------------------------------------------
# Command line: kitscenes video
# ---------------------------------------------------------------------------


def build_parser():
    import argparse

    from kitscenes.cli import where_criterion

    p = argparse.ArgumentParser(
        prog="kitscenes video",
        description="A video of the dataset: the ring cameras with the map labels drawn in and a BEV panel, "
                    "drawn and encoded on the GPU (or the CPU). Scenes in recording order; one MP4 for "
                    "the selection (and with --per-scene one per scene). Speed 1 = real time (10 fps).",
        epilog="examples:\n"
               "  kitscenes video ROOT --splits val --out /tmp/val.mp4\n"
               "  kitscenes video ROOT --scenes ID ... --out /tmp/a.mp4 --resolution 4k --speed 2 --layout cams\n"
               "  kitscenes video ROOT --splits train --where weather=rain --max-scenes 5 --out /tmp/rain.mp4\n"
               "  kitscenes video ROOT --splits val --out /tmp/val.mp4 --gpus 8 --procs-per-gpu 2",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_argument_group("selection")
    s.add_argument("root", metavar="ROOT", help="dataset root (holds data/<split>/<scene>/)")
    s.add_argument("--splits", nargs="+", metavar="SPLIT", help="split folders, e.g. val test-e2e")
    s.add_argument("--scenes", nargs="+", metavar="ID", help="scene IDs")
    s.add_argument("--cities", nargs="+", metavar="CITY", help="Karlsruhe, Frankfurt, Sindelfingen")
    s.add_argument("--lidar", metavar="TYPE", help="velodyne or hesai")
    s.add_argument("--where", action="append", type=where_criterion, metavar="COLUMN=VALUE",
                   help="scenes by a scene-table column, as 'kitscenes download --where' (repeat for several)")
    s.add_argument("--order", choices=ORDERS, default="recording", help="scene order (default: recording order)")
    s.add_argument("--max-scenes", type=int, metavar="N", help="only the first N scenes of the selection")
    s.add_argument("--frames", metavar="START:STOP[:STEP]", help="frames of each scene (Python slice)")
    s.add_argument("--frame-step", type=int, default=1, metavar="K", help="every K-th frame (real time stays real time)")
    s.add_argument("--dry-run", action="store_true", help="print the selection and the video's frame rate; render nothing")
    c = p.add_argument_group("content")
    c.add_argument("--layout", choices=LAYOUTS, default="cams+bev",
                   help="the camera grid with a BEV column (default) or without")
    c.add_argument("--region", default="120x60", metavar="LxW", help="labelled region in metres (default 120x60)")
    c.add_argument("--legend", choices=LEGENDS, default="compact",
                   help="the classes the scene shows (compact, default), every class (full) or none")
    c.add_argument("--crop-hood", nargs="?", const="default", default=None, metavar="FRACTION",
                   help="cut the ego vehicle off the bottom of the cameras (or one FRACTION of the image height)")
    c.add_argument("--fit", choices=FITS, default="cover",
                   help="images fill their tiles, cut at the sides (cover, default), or whole images (contain)")
    c.add_argument("--rear-row", choices=REAR_ROWS, default="swap",
                   help="rear row: rear right | rear | rear left (swap, default), mirrored images (mirror) or "
                        "rear left | rear | rear right (plain)")
    c.add_argument("--no-map", action="store_true", help="no map labels")
    c.add_argument("--grouping", default="default", help="ML converter line type grouping")
    c.add_argument("--line-width", type=float, default=1.0, help="line width factor (default 1)")
    v = p.add_argument_group("video")
    v.add_argument("--out", type=Path, required=True, help="output .mp4 (not inside ROOT)")
    v.add_argument("--resolution", default="1080p", help="1080p (1920x1080), 2k (2560x1440), 4k (3840x2160) or WxH")
    v.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time = 10 fps (default 1)")
    v.add_argument("--max-fps", type=float, default=60.0, help="above this, frames are dropped (default 60)")
    v.add_argument("--codec", choices=CODECS, default="hevc", help="hevc (default) or h264")
    v.add_argument("--quality", type=float, metavar="Q",
                   help="constant quality, NVENC -cq / software -crf, lower is better (default 19 to 21, by encoder)")
    v.add_argument("--max-bitrate", metavar="RATE",
                   help="cap of the constant quality (default: 80M at 4K and 10 fps, scaled by pixels and sqrt(fps/10))")
    v.add_argument("--bitrate", metavar="RATE", help="an average bitrate instead of constant quality, e.g. 20M")
    v.add_argument("--encoder", choices=ENCODERS, default="auto",
                   help="auto: NVENC on a GPU (else the software encoder), nvenc, software (libx265/libx264)")
    v.add_argument("--per-scene", action="store_true", help="also keep one MP4 per scene (in <OUT stem>_scenes/)")
    v.add_argument("--no-concat", action="store_true", help="with --per-scene: only the per-scene MP4s")
    d = p.add_argument_group("device")
    d.add_argument("--device", default="auto", help="auto (first GPU, else CPU), cuda, cuda:K or cpu")
    d.add_argument("--gpus", type=int, default=1, help="GPUs to use, whole scenes per process (default 1)")
    d.add_argument("--procs-per-gpu", type=int, default=4, help="renderer processes per GPU (default 4)")
    d.add_argument("--workers", type=int, default=8,
                   help="reader processes per renderer: reading, labels, decoding (default 8)")
    d.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``kitscenes video``; returns the exit code."""
    from kitscenes.errors import KITScenesError

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from kitscenes.cli import _where

        where = _where(args)
        if args.dry_run:
            return _dry_run(args, where)
        result = render_video(
            args.root, args.out, splits=args.splits, scenes=args.scenes, city=args.cities, lidar=args.lidar,
            where=where, order=args.order, max_scenes=args.max_scenes, frames=args.frames,
            frame_step=args.frame_step, layout=args.layout, region=args.region, legend=args.legend,
            crop_hood=args.crop_hood, fit=args.fit, rear_row=args.rear_row, map_labels=not args.no_map,
            grouping=args.grouping, line_width=args.line_width, resolution=args.resolution, speed=args.speed,
            max_fps=args.max_fps,
            codec=args.codec, quality=args.quality, bitrate=args.bitrate, max_bitrate=args.max_bitrate,
            encoder=args.encoder, device=args.device, gpus=args.gpus,
            procs_per_gpu=args.procs_per_gpu, workers=args.workers, per_scene=args.per_scene,
            concat=not args.no_concat, progress=not args.quiet)
    except (ValueError, OSError, RuntimeError, KITScenesError) as err:
        print(f"kitscenes video: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("kitscenes video: interrupted", file=sys.stderr)
        return 130
    if args.quiet:
        print(result.path or result.scene_videos[0].parent)
    return 0


def _dry_run(args, where) -> int:
    import kitscenes

    ds = kitscenes.open(args.root)
    play = playback(args.speed, args.frame_step, args.max_fps)
    ids = select_scenes(ds, splits=args.splits, scene_ids=args.scenes, city=args.cities, lidar=args.lidar,
                        where=where, order=args.order, max_scenes=args.max_scenes)
    frames = parse_frames(args.frames)
    total = 0
    for scene_id in ids:
        n = len(frame_indices(len(ds.scene(scene_id)), frames, play.step))
        total += n
        print(f"  {ds.split_of(scene_id):<28} {scene_id}  {n} frames")
    size = parse_resolution(args.resolution)
    print(f"{len(ids)} scene(s), {total} frames, {float(play.fps):g} fps, {total / float(play.fps):.1f} s of video, "
          f"{size[0]}x{size[1]} {args.layout}" + (f"; every {play.dropped}. frame (max {args.max_fps:g} fps)"
                                                   if play.dropped > 1 else ""))
    return 0
