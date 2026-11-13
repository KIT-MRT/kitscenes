"""Export a scene to common formats: point clouds as PCD/PLY, images as JPEG, poses as TUM, calibration as JSON.

For tools that do not read KITScenes: CloudCompare, Open3D, PCL, COLMAP-style pipelines, a SLAM evaluation. Per frame
it writes the lidar sweep(s) and the camera JPEGs (the stored bytes, not re-encoded), plus ``poses_tum.txt``
(``t_s x y z qx qy qz qw`` of the frames, world frame) and ``calibration.json`` (K, image size and ``T_base_sensor`` of
every sensor). The output must lie outside the dataset root.

Run::

    KITSCENES_ROOT=/data/kitscenes python examples/export_scene.py --split val --out /tmp/export --frames 0:50:10
    python examples/export_scene.py --root /data/kitscenes --scene <scene uuid> --out /tmp/export \\
        --lidars lidar_top lidar_front --cameras camera_ring_front --frame world --format ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import kitscenes
from kitscenes.export import write_pcd, write_ply


def frame_range(spec: str, count: int) -> range:
    """``"start:stop:step"`` (a Python slice) or one index -> frame indices."""
    if ":" not in spec:
        return range(int(spec), int(spec) + 1)
    return range(count)[slice(*(int(p) if p else None for p in spec.split(":")))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="dataset root (default: $KITSCENES_ROOT)")
    parser.add_argument("--split", help="split to take the scene from")
    parser.add_argument("--scene", help="scene ID (default: the first scene)")
    parser.add_argument("--out", type=Path, required=True, help="output folder (outside the dataset)")
    parser.add_argument("--frames", default="0:20:5", help="start:stop:step or one index (default 0:20:5)")
    parser.add_argument("--lidars", nargs="*", default=["lidar_top"], help="fused into one cloud; none: skip")
    parser.add_argument("--cameras", nargs="*", default=["camera_ring_front"], help="none: skip")
    parser.add_argument("--frame", choices=["base", "world"], default="base", help="coordinate frame of the clouds")
    parser.add_argument("--format", choices=["pcd", "ply"], default="pcd")
    args = parser.parse_args()

    ds = kitscenes.open(args.root, splits=args.split)
    scene = ds[args.scene] if args.scene else ds[0]
    out = args.out.expanduser().resolve()
    if out == ds.root.resolve() or ds.root.resolve() in out.parents:
        raise SystemExit(f"--out must lie outside the dataset root {ds.root}")
    out = out / scene.id
    out.mkdir(parents=True, exist_ok=True)

    rig = scene.rig
    calibration = {name: {"T_base_sensor": sensor.T_base_sensor.tolist()}
                   for name, sensor in {**rig.cameras, **rig.lidars, **rig.radars}.items()}
    for name, camera in rig.cameras.items():
        calibration[name].update(K=camera.K.tolist(), image_size=list(camera.image_size))
    (out / "calibration.json").write_text(json.dumps(calibration, indent=1))

    write = write_pcd if args.format == "pcd" else write_ply
    poses = []
    for index in frame_range(args.frames, len(scene)):
        frame = scene[index]
        name = f"{index:06d}"
        lidars = [lidar for lidar in args.lidars if frame.availability(lidar) is kitscenes.Availability.PRESENT]
        if lidars:
            cloud = frame.fused_lidar(lidars, frame=args.frame)  # fused, deskewed
            (out / "lidar").mkdir(exist_ok=True)
            write(out / "lidar" / f"{name}.{args.format}", cloud)
        for camera in args.cameras:
            jpeg = frame.image_bytes(camera, missing="none")
            if jpeg is not None:
                (out / camera).mkdir(exist_ok=True)
                (out / camera / f"{name}.jpg").write_bytes(jpeg)
        pose = frame.pose
        poses.append(f"{frame.t_s:.6f} " + " ".join(f"{v:.6f}" for v in (*pose.xyz, *pose.quat)))
        print(f"frame {index}: {', '.join(lidars) or 'no lidar'}, {len(args.cameras)} camera(s)")
    (out / "poses_tum.txt").write_text("# t_s x y z qx qy qz qw (world frame: " f"{scene.pose_frame})\n"
                                       + "\n".join(poses) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
