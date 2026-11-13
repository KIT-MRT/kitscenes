"""Visualization of the dataset: map labels from above, everything reprojected into the cameras, and videos.

Three tools (for everything else in 3D, the 3D viewer: ``kitscenes view``):

- :func:`plot_top_down` / :func:`render_top_down`: the map labels around a pose from above.
- :func:`render_frame`: a frame's camera images with the map labels, the lidar points and
  (optionally) the radar detections drawn in; :func:`render_scene` and :func:`write_video` for whole
  scenes (``examples/render_reprojection.py``).
- :func:`render_video` (``kitscenes video``): MP4s of whole selections, the ring cameras with the
  labels and a BEV panel, drawn and encoded on the GPU (or the CPU).

Needs ``kitscenes[viz]`` (matplotlib, OpenCV; :func:`render_video` also torch and the ffmpeg binary:
``kitscenes[video]``) and, for map labels, a lanelet2 build with ``lanelet2.ml_converter``. The traffic
element icons ship with kitscenes (``$KITSCENES_TE_ICON_PATH`` overrides them).
"""

from kitscenes.visualization.reprojection import (
    GRID_CAMERAS,
    MapLayer,
    RenderOptions,
    camera_grid,
    render_frame,
    render_scene,
)
from kitscenes.visualization.topdown import plot_top_down, render_top_down
from kitscenes.visualization.video import write_video

__all__ = [
    "plot_top_down",
    "render_top_down",
    "MapLayer",
    "RenderOptions",
    "render_frame",
    "render_scene",
    "camera_grid",
    "write_video",
    "render_video",
    "GRID_CAMERAS",
]


def __getattr__(name: str):
    """``render_video`` on first use, so that importing the package does not import torch."""
    if name == "render_video":
        from kitscenes.visualization.gpu_video import render_video

        return render_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
