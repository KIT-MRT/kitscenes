"""``kitscenes aerial`` (also ``python -m kitscenes.aerial``): aerial tiles, tile stores and crop caches.

    kitscenes aerial list ROOT                                   # the tiles in ROOT/aerial and the crop caches
    kitscenes aerial check ROOT --splits train --roi 60 30       # which frames the tiles cover
    kitscenes aerial store /scratch/tile_store --tiles ROOT      # tiles -> .npy tile store (once, ~10 GB)
    kitscenes aerial prepare ROOT --splits train --roi 60 30 --cache-dir /scratch/aerial   # a crop cache (PNG)

Crops need no preparation: ds.use_aerial(roi=(60, 30)) cuts them on the
fly (from a tile store: tiles=<store>). The raw tiles come with
'kitscenes download ROOT' (the areas of the selected scenes, unless
--no-aerial); --tiles reads other tile folders (e.g. one per state), only
reading.

Exit codes: 0 done, 1 an error, 2 done but some frames get no crop (no raw tiles there).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def build_parser(prog: str = "kitscenes aerial") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("root", metavar="ROOT", help="dataset root (holds data/<split>/<scene>/)")
        p.add_argument("--splits", nargs="+", metavar="SPLIT", help="split folders (default: all)")
        p.add_argument("--scenes", nargs="+", metavar="ID", help="scene IDs")
        p.add_argument("--roi", type=float, nargs=2, required=True, metavar=("LONGITUDINAL_M", "LATERAL_M"))
        p.add_argument("--tiles", nargs="+", metavar="DIR",
                       help="folders with raw tiles, only read (default: ROOT/aerial): dataset roots, aerial "
                            "folders (metadata/ + raw_tiles/) or tile stores")

    ls = sub.add_parser("list", help="list the raw tiles in ROOT/aerial and the crop caches")
    ls.add_argument("root", metavar="ROOT")
    ls.add_argument("--cache-dir", metavar="DIR", help="folder of the crop caches (default: ROOT/aerial)")

    c = sub.add_parser("check", help="which frames the raw tiles cover (exit code 2 if some are not)")
    selection(c)
    c.add_argument("--json", metavar="FILE", help="write the report, scene by scene, to FILE")

    s = sub.add_parser("store", help="convert raw tiles into a tile store (.npy) for fast crops on the fly")
    s.add_argument("out", metavar="OUT_DIR", help="the writable store folder")
    s.add_argument("--tiles", nargs="+", required=True, metavar="DIR",
                   help="the raw tiles, only read: a dataset root (its aerial/) or aerial folders")
    s.add_argument("--areas", nargs="+", metavar="AREA", help="only these areas (default: all)")
    s.add_argument("--jobs", type=int, default=None, help="conversion processes (default 4; ~130 MiB each)")
    s.add_argument("-v", "--verbose", action="store_true", help="a progress bar")

    p = sub.add_parser("prepare", help="cut one crop per frame into a crop cache (PNG; only what is missing)")
    selection(p)
    p.add_argument("--cache-dir", metavar="DIR", help="writable folder of the crop cache (default: ROOT/aerial)")
    p.add_argument("--version", default="v1", help="tag in the cache ID, to keep variants apart (default v1)")
    p.add_argument("--backend", default="auto", choices=["auto", "cv2", "scipy"], help="interpolation")
    p.add_argument("--jobs", type=int, default=None, help="crop processes, one scene at a time (default: CPU count, at most 8; "
                   "each ~150 MiB + --tile-cache-mb)")
    p.add_argument("--tile-cache-mb", type=float, default=256, help="decoded tiles kept per process (default 256)")
    p.add_argument("-v", "--verbose", action="store_true", help="progress bars")
    return parser


def _dataset(args):
    import kitscenes

    ds = kitscenes.open(args.root, splits=args.splits)
    return ds.filter(scene_ids=args.scenes) if args.scenes else ds


def main(argv: Optional[list[str]] = None, prog: str = "kitscenes aerial") -> int:
    from kitscenes.aerial.cache import AerialCache, aerial_root, cache_ids
    from kitscenes.errors import KITScenesError

    parser = build_parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            from kitscenes.aerial.tiles import AerialTiles

            folder = Path(args.cache_dir) if args.cache_dir else aerial_root(args.root)
            try:
                print(AerialTiles(args.root))
            except (FileNotFoundError, ValueError) as err:
                print(f"No raw tiles: {err}")
            ids = cache_ids(folder)
            for cache_id in ids:
                cache = AerialCache(folder, cache_id)
                scenes = cache.scene_ids
                print(f"crop cache {cache_id}: {sum(len(cache.frame_indices(s)) for s in scenes)} crops "
                      f"in {len(scenes)} scenes")
            if not ids:
                print(f"No crop caches in {folder}.")
            return 0
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if args.command == "store":
            from kitscenes.aerial import build_tile_store

            store = build_tile_store(args.out, args.tiles, areas=args.areas, jobs=args.jobs, verbose=args.verbose)
            print(f"Tile store ready: {store}")
            return 0
        ds = _dataset(args)
        if args.command == "check":
            from kitscenes.aerial import AerialCrops

            report = AerialCrops(args.tiles or args.root, tuple(args.roi)).coverage(ds)
            print(report.summary())
            for scene_id, s in sorted(report.incomplete().items()):
                why = s.reason or f"{len(s.uncovered)} of {s.num_frames} frames lack raw tiles"
                print(f"  {scene_id} ({s.split}): {why}")
            if args.json:
                report.to_json(args.json)
            return 0 if report.complete else 2
        cache = ds.prepare_aerials(
            tuple(args.roi), tiles=args.tiles, cache_dir=args.cache_dir, version=args.version,
            backend=args.backend, verbose=args.verbose, workers=args.jobs, tile_cache_mb=args.tile_cache_mb,
        )
    except (ValueError, LookupError, OSError, KITScenesError) as err:
        # bad selection, no georeferenced scenes, no tiles, I/O errors
        print(f"{prog} {args.command}: {err}", file=sys.stderr)
        return 1
    print(f"Aerial cache ready: cache_id={cache.cache_id} ({cache.root})")
    print(cache.coverage.summary())
    return 0 if cache.coverage.complete else 2


if __name__ == "__main__":
    sys.exit(main(prog="python -m kitscenes.aerial"))
