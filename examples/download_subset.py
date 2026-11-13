"""Pick scenes from the scene table shipped with the package and download only those, with only some sensors.

The table (``kitscenes.scene_table()``) knows split, city, lidar generation, frame count and recording metadata
(weather, light, road types, ...) of every released scene, so a subset can be chosen before anything is downloaded. This script selects scenes and prints the
``kitscenes download`` command for them; ``--run`` runs it (``--dry-run`` first to see the size). Downloading needs
the dataset terms accepted on HuggingFace and ``hf auth login``. The DOP20 aerial tiles of the scenes' areas come along
(``--no-aerial`` leaves them out); ``--aerial-features`` adds the DINOv3 aerial features of those areas (up to ~390 GB
per area).

Run::

    python examples/download_subset.py /data/kitscenes --split val --city Karlsruhe --lidar hesai --count 5
    python examples/download_subset.py /data/kitscenes --split train --weather rain --light night twilight --count 10
    python examples/download_subset.py /data/kitscenes --split train --count 20 --components ring-cameras lidar-top --run
    python examples/download_subset.py /data/kitscenes --split val --count 5 --aerial-features --dry-run --run
"""

from __future__ import annotations

import argparse
import random
import shlex
import subprocess
import sys

import kitscenes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="dataset root to download into")
    parser.add_argument("--split", default="val")
    parser.add_argument("--city", help="Karlsruhe, Frankfurt or Sindelfingen")
    parser.add_argument("--lidar", choices=["velodyne", "hesai"])
    parser.add_argument("--weather", nargs="+", help="clear, partly_cloudy, overcast, rain, fog, snow, unknown")
    parser.add_argument("--light", nargs="+", help="day, low_sun, twilight, night")
    parser.add_argument("--count", type=int, default=5, help="number of scenes (picked at random, --seed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--components", nargs="+", default=["all"],
                        help="cameras and lidars: names or presets (all, cameras, ring-cameras, base-cameras, "
                             "front-camera, lidars, lidar-top, none); metadata, maps, labels and radar always come")
    parser.add_argument("--aerial-features", action="store_true",
                        help="with the DINOv3 aerial feature tiles of the scenes' areas (up to ~390 GB per area)")
    parser.add_argument("--no-aerial", action="store_true", help="without the DOP20 aerial tiles of the scenes' areas")
    parser.add_argument("--run", action="store_true", help="run the download (else only print the command)")
    parser.add_argument("--dry-run", action="store_true", help="with --run: list what would be fetched")
    args = parser.parse_args()

    split = kitscenes.splits.canonical_split_name(args.split)
    chosen = [info for info in kitscenes.find_scenes(city=args.city, lidar=args.lidar, weather=args.weather,
                                                     light=args.light)
              if info.split == split]
    if not chosen:
        print("no scene matches", file=sys.stderr)
        return 1
    chosen = sorted(random.Random(args.seed).sample(chosen, min(args.count, len(chosen))), key=lambda i: i.scene_id)
    frames = sum(info.num_frames for info in chosen)
    print(f"{len(chosen)} scenes, {frames} frames ({frames / 10 / 60:.1f} min of driving):")
    for info in chosen:
        print(f"  {info.scene_id}  {info.city:12s} {info.lidar:8s} {info.num_frames:4d} frames  {info.weather}, {info.light}")

    command = ["kitscenes", "download", args.root, "--splits", split, "--scenes", *(i.scene_id for i in chosen),
               "--components", *args.components] + (["--dry-run"] if args.dry_run else [])
    command += (["--no-aerial"] if args.no_aerial else []) + (["--aerial-features"] if args.aerial_features else [])
    print("\n" + shlex.join(command))
    return subprocess.call(command) if args.run else 0


if __name__ == "__main__":
    sys.exit(main())
