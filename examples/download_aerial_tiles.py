"""Fetch orthophoto tiles around chosen scenes from a WMS service into a tile folder of your own.

The release ships the DOP20 tiles of every scene's area, so aerial crops need no download. This script is for
imagery the release does not ship: a newer or older flight, or another source. One use is the map_change scenes,
where the road changed after it was mapped: imagery of another date next to the map and the recorded drive.

A service serves its current mosaic unless a layer or a parameter says otherwise, so the result depends on the day
the script is run: keep the folder with your results. The terms of the service apply to the tiles (see the
``SOURCE.md`` the script writes); the release's ``aerial/ATTRIBUTION.md`` covers only the released tiles.

The dataset root is only read (poses and map origins). Tiles go to ``--out``, in the layout and on the pixel grid of
the release tiles, one request at a time; a rerun fetches only what is missing. Needs ``pip install requests``.

Run::

    python examples/download_aerial_tiles.py /data/kitscenes --out /data/aerial_today --splits map_change_val --dry-run
    python examples/download_aerial_tiles.py /data/kitscenes --out /data/aerial_today --splits map_change_val
    python examples/download_aerial_tiles.py /data/kitscenes --out /data/aerial_ka --cities Karlsruhe --where weather=rain
    python examples/download_aerial_tiles.py /data/kitscenes --out /data/aerial_other --scenes SCENE_ID \\
        --wms-url https://example.org/wms --layer ortho_2019 --param TIME=2019 --name ortho2019

Without ``--wms-url`` the tiles come from the DOP20 service of the scene's state (Hesse for Frankfurt,
Baden-Württemberg for Karlsruhe and Sindelfingen), the services the release imagery comes from.

Use the folder like the release tiles::

    ds = kitscenes.open("/data/kitscenes")
    crops = ds.use_aerial(roi=(60, 30), tiles="/data/aerial_today")
    crop = crops[frame]                       # (150, 300, 3) uint8, as from the release tiles

    kitscenes aerial check /data/kitscenes --roi 60 30 --tiles /data/aerial_today
    kitscenes aerial store /scratch/tile_store --tiles /data/aerial_today
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import kitscenes
from kitscenes.aerial import scene_poses
from kitscenes.cli import where_criterion

#: The DOP20 services of the two states (``--source``), and the state of each city.
SOURCES = {
    "hessen": ("https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows", "he_dop20_rgb"),
    "bw": ("https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C", "IMAGES_DOP_20_RGB"),
}
CITY_SOURCE = {"Frankfurt": "hessen", "Karlsruhe": "bw", "Sindelfingen": "bw"}
AREA_SIZE_M = 10_000.0  # scenes are grouped into 10 km areas of EPSG:25832, as in the release
SUFFIX = {"png": ".png", "jpeg": ".jpg"}
BYTES_PER_PIXEL = {"png": 2.2, "jpeg": 0.4}  # rough, for the estimate of --dry-run
ATTEMPTS = 4


def fetch_tile(url: str, layer: str, bbox: tuple, size_px: int, fmt: str, params: dict) -> bytes:
    """One WMS GetMap request: the image of the EPSG:25832 *bbox* (min_e, min_n, max_e, max_n), as sent."""
    try:
        import requests
    except ImportError:
        raise SystemExit("this script needs the requests package: pip install requests") from None
    query = {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": layer, "STYLES": "",
             "SRS": "EPSG:25832", "BBOX": ",".join(f"{v:.3f}" for v in bbox), "WIDTH": size_px, "HEIGHT": size_px,
             "FORMAT": f"image/{fmt}", "EXCEPTIONS": "application/vnd.ogc.se_xml", **params}
    response = requests.get(url, params=query, timeout=120)
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        raise RuntimeError(f"the service sent no image: {response.text[:300]}")
    return response.content


def checked_pixels(data: bytes, size_px: int) -> np.ndarray:
    """The (S, S, 3) pixels of a fetched tile; raises for another size or a blank image (one colour)."""
    pixels = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    if pixels.shape != (size_px, size_px, 3):
        raise RuntimeError(f"the service sent {pixels.shape[1]} x {pixels.shape[0]} px, not {size_px} x {size_px}")
    if (pixels == pixels[0, 0]).all():
        raise RuntimeError("the service sent a blank tile")
    return pixels


def area_of(lat: float, lon: float) -> str:
    """``area_<min_e>_<min_n>``: the 10 km area of a map origin."""
    from pyproj import Transformer

    e, n = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True).transform(lon, lat)
    return f"area_{int(e // AREA_SIZE_M * AREA_SIZE_M)}_{int(n // AREA_SIZE_M * AREA_SIZE_M)}"


def plan(ds, extent_m: float, margin_m: float, source: str, known: dict) -> tuple[dict, list[str]]:
    """``{area: {"origin", "tiles", "scenes", "source"}}`` for the scenes of *ds* that can be placed, and the
    scenes that cannot. *known* gives the grid origin of the areas the folder already has."""
    areas: dict[str, dict] = {}
    skipped = []
    for scene in ds:
        try:
            poses = scene_poses(scene)
        except (LookupError, ValueError):  # no map origin (test-e2e), no usable poses
            skipped.append(scene.id)
            continue
        area = areas.setdefault(area_of(poses.lat, poses.lon), {"points": [], "scenes": [], "sources": set()})
        area["points"].append(poses.xy)
        area["scenes"].append(scene.id)
        city = scene.info.city if scene.info is not None else None
        area["sources"].add(source if source != "auto" else CITY_SOURCE.get(city))
    for name, area in areas.items():
        points = np.concatenate(area.pop("points"))
        sources = area.pop("sources")
        if len(sources) != 1 or None in sources:
            raise SystemExit(f"{name}: the city of scene {area['scenes'][0]} names no service; pass --source or --wms-url")
        area["source"] = sources.pop()
        # A grid origin on a multiple of the tile size puts the tiles on the pixel grid of the release tiles.
        area["origin"] = known.get(name) or [float(v) for v in np.floor(points.min(axis=0) / extent_m) * extent_m]
        lo = np.floor((points - margin_m - area["origin"]) / extent_m).astype(int)
        hi = np.floor((points + margin_m - area["origin"]) / extent_m).astype(int)
        area["tiles"] = sorted({(i, j) for (i0, j0), (i1, j1) in zip(lo.tolist(), hi.tolist())
                                for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)})
    return areas, skipped


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", metavar="ROOT", help="dataset root, only read")
    parser.add_argument("--out", required=True, metavar="DIR", help="the tile folder to write (not inside a dataset root)")
    parser.add_argument("--splits", nargs="+", metavar="SPLIT")
    parser.add_argument("--scenes", nargs="+", metavar="ID")
    parser.add_argument("--cities", nargs="+", metavar="CITY")
    parser.add_argument("--where", action="append", type=where_criterion, default=[], metavar="COLUMN=VALUE",
                        help="a column of the scene table, as in 'kitscenes download' (repeatable)")
    parser.add_argument("--source", default="auto", choices=["auto", *SOURCES],
                        help="the state DOP20 service (default: by the scene's city)")
    parser.add_argument("--wms-url", metavar="URL", help="another WMS service (EPSG:25832), with --layer")
    parser.add_argument("--layer", metavar="NAME")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="an extra request parameter, passed on as it is (repeatable), e.g. TIME=2019")
    parser.add_argument("--format", default="png", choices=list(SUFFIX), help="image format to ask for (default png)")
    parser.add_argument("--name", default="wms", help="name of the imagery in tile and crop IDs (default wms)")
    parser.add_argument("--margin-m", type=float, default=60.0,
                        help="metres around the driven path (default 60: enough for crops up to 100 x 50 m)")
    parser.add_argument("--resolution", type=float, default=0.2, metavar="M_PER_PX", help="default 0.2, as the release")
    parser.add_argument("--tile-size-px", type=int, default=2048, help="default 2048, as the release")
    parser.add_argument("--pause-s", type=float, default=0.5, help="pause between requests (default 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="print areas, tiles and size; fetch nothing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.wms_url) != bool(args.layer):
        raise SystemExit("--wms-url and --layer go together")
    out = Path(args.out).resolve()
    for root in {Path(args.root).resolve(), Path(os.environ.get("KITSCENES_ROOT") or args.root).resolve()}:
        if out == root or root in out.parents:
            raise SystemExit(f"--out {out} is inside the dataset root {root}: pick a folder of your own")
    extra = dict(p.split("=", 1) for p in args.param)
    if args.wms_url:
        SOURCES["custom"] = (args.wms_url, args.layer)
        args.source = "custom"

    ds = kitscenes.open(args.root, splits=args.splits)
    ds = ds.filter(scene_ids=args.scenes, city=args.cities, **dict(args.where))
    raw_id = f"{args.name}_res{round(args.resolution * 100):03d}_tile{args.tile_size_px}"
    extent = args.resolution * args.tile_size_px
    manifest_path = out / "metadata" / f"raw_tile_manifest_{raw_id}.json"
    manifest = read_json(manifest_path, {
        "raw_cache_id": raw_id, "provider": args.name, "resolution_m_per_px": args.resolution,
        "tile_size_px": args.tile_size_px, "effective_tile_size_px": args.tile_size_px, "tile_extent_m": extent,
        "areas": {}})
    index = read_json(out / "metadata" / "area_index.json",
                      {"strategy": "epsg25832_bucket", "bucket_size_m": AREA_SIZE_M, "scene_to_area": {}, "areas": {}})
    areas, skipped = plan(ds, extent, args.margin_m, args.source,
                          {a: entry["origin_xy"] for a, entry in manifest["areas"].items()})
    if not areas:
        raise SystemExit(f"none of the {len(ds)} selected scenes can be placed on imagery (no map origin)")

    tiles_dir = out / "raw_tiles" / raw_id
    todo = {}
    for name, area in areas.items():
        have = {(t["i"], t["j"]) for t in manifest["areas"].get(name, {}).get("tiles", [])
                if (out / t["path"]).is_file()}
        todo[name] = [ij for ij in area["tiles"] if ij not in have]
        url, layer = SOURCES[area["source"]]
        print(f"{name}: {len(area['scenes'])} scenes, {len(area['tiles'])} tiles, {len(todo[name])} to fetch "
              f"from {url} ({layer})")
    missing = sum(len(t) for t in todo.values())
    size_gb = missing * args.tile_size_px ** 2 * BYTES_PER_PIXEL[args.format] / 1e9
    print(f"{missing} tiles to fetch, about {size_gb:.2f} GB, into {out}"
          + (f"; {len(skipped)} scenes without a map origin left out" if skipped else ""))
    if args.dry_run or not missing:
        return 0

    fetched, failed = 0, []
    try:
        for name, area in areas.items():
            url, layer = SOURCES[area["source"]]
            entry = manifest["areas"].setdefault(name, {"origin_xy": area["origin"], "crs": "EPSG:25832", "tiles": []})
            listed = {(t["i"], t["j"]) for t in entry["tiles"]}
            for i, j in todo[name]:
                e, n = area["origin"][0] + i * extent, area["origin"][1] + j * extent
                rel = f"raw_tiles/{raw_id}/{name}/tile_{i}_{j}{SUFFIX[args.format]}"
                for attempt in range(ATTEMPTS):
                    try:
                        data = fetch_tile(url, layer, (e, n, e + extent, n + extent), args.tile_size_px, args.format, extra)
                        checked_pixels(data, args.tile_size_px)
                        break
                    except (OSError, RuntimeError) as err:  # network errors, error reports, blank or odd images
                        problem = err
                        time.sleep(args.pause_s * 4 * 2 ** attempt)
                else:
                    failed.append(f"{name} tile {i} {j}: {problem}")
                    continue
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".partial")
                partial.write_bytes(data)
                os.replace(partial, path)
                if (i, j) not in listed:
                    entry["tiles"].append({"i": i, "j": j, "path": rel})
                fetched += 1
                print(f"\r{fetched + len(failed)} / {missing} tiles", end="", file=sys.stderr, flush=True)
                time.sleep(args.pause_s)
    finally:  # also after Ctrl-C: what was fetched is listed, the next run goes on from there
        print(file=sys.stderr)
        if fetched:
            write_metadata(out, manifest_path, manifest, index, areas, args, extra)
    total = sum(len(a["tiles"]) for a in areas.values())
    print(f"fetched {fetched}, skipped {total - missing} (already there), failed {len(failed)}")
    for line in failed:
        print(f"  {line}", file=sys.stderr)
    return 2 if failed else 0


def write_metadata(out: Path, manifest_path: Path, manifest: dict, index: dict, areas: dict, args, extra: dict) -> None:
    """The tile manifest, the area index, the services and ``SOURCE.md`` (one entry per run)."""
    meta = out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    providers = read_json(meta / "provider_metadata.json", {})
    used = {}
    for name, area in areas.items():
        if name not in manifest["areas"]:
            continue
        entry = index["areas"].setdefault(name, {"scene_ids": []})
        entry["scene_ids"] = sorted(set(entry["scene_ids"]) | set(area["scenes"]))
        index["scene_to_area"].update({scene: name for scene in area["scenes"]})
        url, layer = SOURCES[area["source"]]
        used[area["source"]] = providers[area["source"]] = {"wms_url": url, "layer": layer, **extra}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    (meta / "area_index.json").write_text(json.dumps(index, indent=1))
    (meta / "provider_metadata.json").write_text(json.dumps(providers, indent=1))
    lines = [] if (out / "SOURCE.md").is_file() else [
        "# Source of these tiles", "",
        "Fetched with `examples/download_aerial_tiles.py` of the kitscenes API. The terms of the service named "
        "below apply to these tiles; `aerial/ATTRIBUTION.md` of the KITScenes release covers only the released tiles.",
        ""]
    lines.append(f"## Download of {datetime.date.today().isoformat()}")
    for source, entry in used.items():
        params = ", ".join(f"{k}={v}" for k, v in extra.items()) or "none"
        lines.append(f"- {source}: {entry['wms_url']}, layer `{entry['layer']}`, format {args.format}, "
                     f"extra parameters: {params}")
    with open(out / "SOURCE.md", "a") as fh:
        fh.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    sys.exit(main())
