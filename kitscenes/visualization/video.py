"""An MP4 from rendered frames (:func:`kitscenes.visualization.render_scene`): camera grid, top-down view beside it."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from kitscenes.visualization import _deps


def rendered_frames(render_dir: Path | str) -> list[int]:
    """Frame indices with a grid image in *render_dir*."""
    return sorted(int(p.stem) for p in (Path(render_dir) / "grid").glob("*.jpg") if p.stem.isdigit())


def composite(render_dir: Path | str, index: int, top_down: bool = True) -> Optional[np.ndarray]:
    """The grid of frame *index* with its top-down view appended on the right (BGR), or None."""
    cv2 = _deps.cv2()

    name = f"{index:010d}.jpg"
    grid = cv2.imread(str(Path(render_dir) / "grid" / name))
    if grid is None:
        return None
    side = cv2.imread(str(Path(render_dir) / "top_down" / name)) if top_down else None
    if side is not None:
        width = int(grid.shape[0] * side.shape[1] / side.shape[0])
        grid = cv2.hconcat([grid, cv2.resize(side, (width, grid.shape[0]), interpolation=cv2.INTER_AREA)])
    return grid


def write_video(
    render_dir: Path | str,
    out_path: Path | str,
    *,
    fps: int = 10,
    top_down: bool = True,
) -> Path:
    """Write the rendered frames of *render_dir* as an MP4 to *out_path*.

    The video is MPEG-4 Part 2 (``mp4v``), written with OpenCV: VLC, mpv and ffmpeg play it, browsers do
    not. ``kitscenes video`` writes HEVC or H.264.

    Raises:
        ValueError: If there are no rendered frames.
    """
    cv2 = _deps.cv2()

    out_path = _deps.guard_output_path(out_path)
    frames = rendered_frames(render_dir)
    if not frames:
        raise ValueError(f"no rendered frames in {Path(render_dir) / 'grid'}")
    first = composite(render_dir, frames[0], top_down)
    size = (first.shape[1], first.shape[0])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, True)
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {out_path}")
    try:
        with ThreadPoolExecutor(8) as pool:  # decoding in threads, writing in order
            for image in pool.map(lambda i: composite(render_dir, i, top_down), frames):
                if image is not None:
                    writer.write(cv2.resize(image, size) if image.shape[1::-1] != size else image)
    finally:
        writer.release()
    return out_path
