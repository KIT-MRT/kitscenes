"""``kitscenes-viewer <scene | dataset root>``: the viewer page and the scene server in one command.

    kitscenes-viewer /data/kitscenes/data/train/<scene-uuid>     # one scene
    kitscenes-viewer /data/kitscenes                             # every scene of a city
    kitscenes-viewer /data/kitscenes --map frankfurt.osm --city-name frankfurt

A folder holding ``poses.txt`` is one scene; anything else is searched for
scenes and served as a city. The viewer server (``server.py``) runs as a
child process and opens a browser tab. When a viewer already answers on the
port (JOSM's 3D viewer, or another ``kitscenes-viewer``), the scenes go into
that one instead: the scene server adopts the map already shown there.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOST = "127.0.0.1"  # both servers listen on this machine only
CITIES = ("karlsruhe", "frankfurt", "sindelfingen")
ATTRIBUTES = ("auto", "geometry", "colour", "all")


def viewer_running(host, port, timeout=1.0):
    """Does a viewer server answer on ``host:port``?"""
    try:
        with urllib.request.urlopen("http://%s:%d/healthz" % (host, port), timeout=timeout) as response:
            return bool(json.loads(response.read(4096)).get("ok"))
    except Exception:
        return False


def start_viewer(host, http_port, ingest_port, browser, tile_proxy_hosts=(), wait_s=15.0):
    """The viewer server as a child process, once it answers."""
    command = [sys.executable, str(HERE / "server.py"), "--host", host, "--http-port", str(http_port),
               "--ingest-port", str(ingest_port), "--standalone"]
    for name in tile_proxy_hosts:
        command += ["--tile-proxy-host", name]
    if not browser:
        command.append("--no-browser")
    process = subprocess.Popen(command)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if viewer_running(host, http_port):
            return process
        if process.poll() is not None:
            raise RuntimeError("the viewer server exited (exit %s); is port %d or %d taken?"
                               % (process.returncode, http_port, ingest_port))
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError("the viewer server did not answer on %s:%d within %.0f s" % (host, http_port, wait_s))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="kitscenes-viewer", description="Show KITScenes scenes in the 3D viewer (browser).")
    parser.add_argument("path", type=Path, help="a scene folder (holds poses.txt) or a dataset root")
    parser.add_argument("--city-name", choices=CITIES, default=None,
                        help="dataset root: the city to show (default: the one with the most scenes)")
    parser.add_argument("--map", type=Path, default=None,
                        help="Lanelet2 map to show instead of the scene's own, e.g. a city-wide map "
                             "(default: the scene's map, or its city's map for a test scene; none for a city)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where point colours, labels and whole-scene clouds are cached "
                             "(default: ~/.cache/kitscenes-viewer); never the dataset")
    parser.add_argument("--no-cache", action="store_true", help="cache nothing on disk")
    parser.add_argument("--aggregate-attributes", choices=ATTRIBUTES, default="auto",
                        help="what a whole-scene cloud carries besides geometry: camera colour (colour), "
                             "and seamseg classes (all). auto: all with a GPU, geometry without one")
    parser.add_argument("--build-aggregates", action="store_true",
                        help="dataset root: build every scene's whole-scene cloud into the cache, then exit "
                             "(no page is served)")
    parser.add_argument("--port", type=int, default=8765,
                        help="the page's port (default 8765); the viewer also uses the next one")
    parser.add_argument("--data-port", type=int, default=8767, help="the scene data's port (default 8767)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--tile-proxy-host", action="append", default=[], metavar="HOST",
                        help="let the page fetch background map tiles from this host (host or host:port) "
                             "through the viewer, for tile servers that do not allow cross-origin use; "
                             "repeatable")
    return parser


def scene_arguments(args):
    """The scene server's command line for the parsed options."""
    mode = "--scene" if (args.path / "poses.txt").is_file() else "--city"
    out = [mode, str(args.path), "--viewer-host", HOST, "--viewer-port", str(args.port),
           "--ingest-port", str(args.port + 1), "--http-port", str(args.data_port),
           "--aggregate-attributes", args.aggregate_attributes]
    if args.city_name:
        out += ["--city-name", args.city_name]
    if args.map:
        out += ["--map", str(args.map)]
    if args.cache_dir:
        out += ["--cache-dir", str(args.cache_dir)]
    if args.no_cache:
        out.append("--no-cache")
    if args.build_aggregates:
        out.append("--build-aggregates")
    return out


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.path.is_dir():
        parser.error("no such folder: %s" % args.path)
    if args.map and not args.map.is_file():
        parser.error("no such map file: %s" % args.map)
    is_scene = (args.path / "poses.txt").is_file()
    if is_scene and args.city_name:
        parser.error("--city-name is for a dataset root; %s is one scene" % args.path)
    if is_scene and args.build_aggregates:
        parser.error("--build-aggregates is for a dataset root; %s is one scene" % args.path)
    if args.build_aggregates and args.no_cache:
        parser.error("--build-aggregates needs the cache (drop --no-cache)")

    from kitscenes_viewer.scene.__main__ import main as scene_main
    if args.build_aggregates:
        try:
            return scene_main(scene_arguments(args))
        except KeyboardInterrupt:
            return 1

    viewer = None
    if viewer_running(HOST, args.port):
        print("viewer: using the one already running at http://%s:%d" % (HOST, args.port), flush=True)
        if args.tile_proxy_host:
            print("viewer: --tile-proxy-host is not applied to a viewer that is already running", flush=True)
    else:
        try:
            viewer = start_viewer(HOST, args.port, args.port + 1, browser=not args.no_browser,
                                  tile_proxy_hosts=args.tile_proxy_host)
        except RuntimeError as exc:
            print("viewer: %s" % exc, file=sys.stderr)
            return 1
        print("viewer: http://%s:%d" % (HOST, args.port), flush=True)

    try:
        return scene_main(scene_arguments(args))
    except KeyboardInterrupt:
        return 0
    finally:
        if viewer is not None:
            viewer.terminate()
            try:
                viewer.wait(5)
            except subprocess.TimeoutExpired:
                viewer.kill()


if __name__ == "__main__":
    sys.exit(main())
