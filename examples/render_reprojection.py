"""Render a scene's reprojection: the ring cameras with map labels, lidar and radar drawn in, as JPEGs and an MP4.

Uses ``kitscenes.visualization.render_scene`` (one JPEG per camera, grid and top-down view per frame, in worker
processes) and ``write_video`` (an MP4 of grid + top-down). The map layer needs ``kitscenes[map,viz]``
and a scene with a map; scenes of the test splits are rendered without it. The output must lie outside the dataset.
For videos of whole splits with the map labels and a BEV panel, see ``kitscenes video``.

Run::

    KITSCENES_ROOT=/data/kitscenes python examples/render_reprojection.py --split val --out /tmp/render --frames 0:100:2
    python examples/render_reprojection.py --root /data/kitscenes --scene <scene uuid> --out /tmp/render --radar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kitscenes
from kitscenes.visualization import RenderOptions, render_scene, write_video
from kitscenes.visualization.gpu_video import parse_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="dataset root (default: $KITSCENES_ROOT)")
    parser.add_argument("--split", help="split to take the scene from")
    parser.add_argument("--scene", help="scene ID (default: the first scene)")
    parser.add_argument("--out", type=Path, required=True, help="output folder (outside the dataset)")
    parser.add_argument("--frames", default="0::2", metavar="START:STOP[:STEP]",
                        help="frames as a Python slice, or one index (default every other frame)")
    parser.add_argument("--reduce", type=int, default=4, choices=[1, 2, 4, 8], help="render at 1/reduce size")
    parser.add_argument("--radar", action="store_true", help="also draw the radar detections")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ds = kitscenes.open(args.root, splits=args.split)
    scene = ds[args.scene] if args.scene else ds[0]
    has_map = scene.availability("map") is kitscenes.Availability.PRESENT
    options = RenderOptions(map=has_map, top_down=has_map, lidar=True, radar=args.radar, reduce=args.reduce,
                            grouping="m3tr")
    frames = range(len(scene))[parse_frames(args.frames)]
    render_dir = render_scene(scene, args.out / scene.id, frames=frames, options=options, workers=args.workers)
    video = write_video(render_dir, args.out / f"{scene.id}.mp4", top_down=has_map)
    print(f"{len(frames)} frames rendered into {render_dir}\nvideo: {video}")


if __name__ == "__main__":
    main()
