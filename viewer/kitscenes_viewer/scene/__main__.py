"""The scene server: feed a KITScenes scene, or a whole city of them, into a running viewer.

    # `kitscenes-viewer <path>` starts the viewer server and this; by hand:
    python -m kitscenes_viewer.scene --scene /path/to/<scene-uuid>
    python -m kitscenes_viewer.scene --city $KITSCENES_ROOT --map /path/to/city/lanelet2_map.osm
    python -m kitscenes_viewer.scene --city $KITSCENES_ROOT --build-aggregates   # batch, then exit

The scene server connects to the viewer's ingest port like the JOSM bridge
does. If a bridge is already streaming a map (JOSM), its anchor is adopted and
the map is left alone; otherwise the scene's own `maps/map.osm` is sent.
"""

import argparse
import re
import sys
import time
import urllib.request

from .aggregate import ATTRIBUTES, SceneAggregator, aggregate_cache_name, resolve_attributes
from .bridge import ViewerBridge
from .cache import DEFAULT_LIMIT_GB, FrameDiskCache, default_cache_dir
from .describe import make_command_handler
from .http import CityHttpServer, SceneHttpServer
from .scene import MapFile, SceneSource

ANCHOR_RE = re.compile(rb'"anchor":\s*\{"lat":\s*([-\d.eE+]+),\s*"lon":\s*([-\d.eE+]+)\}')


def probe_viewer_anchor(host, port, timeout=2.0):
    """The anchor a bridge has already established, or None.

    Only the head of `/state` is read: the whole scene can be tens of
    megabytes, and `anchor` and the start of `features` are in the first
    bytes of it.
    """
    url = "http://%s:%d/state" % (host, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            head = response.read(8192)
    except Exception:
        return None
    match = ANCHOR_RE.search(head)
    if not match:
        return None
    tail = head[match.end():]
    has_features = b'"features":[{' in tail.replace(b" ", b"")
    if not has_features:
        return None
    return float(match.group(1)), float(match.group(2))


def wait_for_viewer_anchor(host, port, seconds, log, probe=probe_viewer_anchor, sleep=time.sleep,
                           clock=time.monotonic):
    """`probe_viewer_anchor`, retried for up to `seconds` while JOSM's map is on its way.

    Started from JOSM, this server comes up while JOSM may still be building
    its first snapshot (seconds for a city). Adopting the anchor is what puts
    the scene on JOSM's map, so it is worth waiting for, once, at the start.
    """
    anchor = probe(host, port)
    if anchor or seconds <= 0:
        return anchor
    log("waiting up to %.0fs for JOSM's map in the viewer" % seconds)
    deadline = clock() + seconds
    while clock() < deadline:
        sleep(0.5)
        anchor = probe(host, port)
        if anchor:
            return anchor
    log("no map from JOSM after %.0fs; going on without it" % seconds)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m kitscenes_viewer.scene",
        description="Serve a KITScenes scene, or every scene of a city, to a running KITScenes viewer.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--scene", help="scene directory (the one holding poses.txt)")
    which.add_argument("--city", metavar="ROOT",
                       help="every scene of one city: KITSCENES_ROOT, its data/ or one split")
    parser.add_argument("--city-name", default=None,
                        help="karlsruhe, frankfurt or sindelfingen (default: the one with "
                             "the most scenes under --city)")
    parser.add_argument("--map", default=None,
                        help="Lanelet2 map to send instead of the scene's own crop -- the "
                             "city-wide map, typically. Without it, a scene with no map of "
                             "its own (test splits) gets its city's map from the dataset's "
                             "lanelet2_hd_maps/. Ignored while JOSM streams one")
    parser.add_argument("--rescan", action="store_true",
                        help="city: read every scene from the dataset instead of the local "
                             "index cache (new and removed scenes are noticed anyway; this "
                             "is for files changed inside a scene)")
    parser.add_argument("--build-aggregates", action="store_true",
                        help="city: build every scene's whole-scene cloud into the cache, "
                             "one after the other, then exit. Cold builds take minutes a "
                             "scene; browsing the city afterwards only reads the cache")
    parser.add_argument("--wait-for-map", type=float, default=0.0, metavar="SECONDS",
                        help="wait this long for a bridge (JOSM) to stream its map before "
                             "starting, so its anchor is adopted (default: do not wait)")
    parser.add_argument("--viewer-host", default="127.0.0.1")
    parser.add_argument("--ingest-port", type=int, default=8766, help="the viewer's TCP ingest port")
    parser.add_argument("--viewer-port", type=int, default=8765, help="the viewer's HTTP port")
    parser.add_argument("--http-port", type=int, default=8767, help="port for this server's data routes")
    parser.add_argument("--no-map", action="store_true", help="never send the map (JOSM owns it)")
    parser.add_argument("--cache-dir", default=None,
                        help="where point colours, labels and whole-scene clouds are cached "
                             "(default: %s). Never the dataset, which stays read-only." % default_cache_dir())
    parser.add_argument("--no-cache", action="store_true", help="cache nothing on disk")
    parser.add_argument("--cache-limit-gb", type=float, default=DEFAULT_LIMIT_GB,
                        help="trim the cache to this size, least recently used first")
    parser.add_argument("--aggregate-attributes", default="auto", choices=("auto",) + ATTRIBUTES,
                        help="what the whole-scene cloud carries besides geometry: camera colour "
                             "(colour), and seamseg classes (all). auto: all with a GPU, geometry "
                             "without one, where colour and classes double the time")
    args = parser.parse_args(argv)
    if args.build_aggregates and not args.city:
        parser.error("--build-aggregates needs --city")
    if args.build_aggregates and args.no_cache:
        parser.error("--build-aggregates needs the cache (drop --no-cache)")
    args.aggregate_attributes = resolve_attributes(args.aggregate_attributes)

    log = lambda msg: print(msg, flush=True)  # noqa: E731
    if args.city:
        return run_city(args, log)
    source = SceneSource(args.scene, map_path=args.map)
    log("scene: %s, %d frames, lidars=%s, cameras=%d"
        % (source.scene_id, source.frame_count, ",".join(source.lidars), len(source.cameras)))
    send_map = not args.no_map and source.map_path is not None
    for note in source.notes:
        log("scene: %s" % note)

    existing = wait_for_viewer_anchor(args.viewer_host, args.viewer_port, args.wait_for_map, log)
    if existing and not source.placed:
        log("scene: a bridge is streaming a map, but this scene cannot be placed on it")
    elif existing:
        source.set_anchor(existing)
        send_map = False
        log("scene: a bridge is already streaming a map; adopting its anchor %.7f, %.7f" % existing)
    if source.placed:
        log("scene: anchor %.7f, %.7f" % source.anchor)

    disk = FrameDiskCache(source.scene_id, cache_root(args), args.cache_limit_gb, log=log)
    log("scene: %s" % disk.describe())
    http = SceneHttpServer(source, host=args.viewer_host, port=args.http_port,
                           allow_origin=viewer_origin(args), log=log, disk_cache=disk,
                           aggregator=make_aggregator(args, source, disk, log))
    http.start()
    return serve(args, log, "scene", http, lambda: source.describe(http.base_url), existing,
                 anchor=source.anchor, features=source.map_features if send_map else None,
                 lanelet_map=lambda: source.lanelet_map)


def cache_root(args):
    """The cache directory, or None with `--no-cache`."""
    return None if args.no_cache else (args.cache_dir or default_cache_dir())


def viewer_origin(args):
    """The origin of the viewer page, the only one allowed to fetch from this server."""
    return "http://%s:%d" % (args.viewer_host, args.viewer_port)


def make_aggregator(args, source, disk, log):
    return SceneAggregator(source, disk_cache=disk, attributes=args.aggregate_attributes, log=log)


def serve(args, log, label, http, announce, existing, anchor=None, features=None, lanelet_map=None):
    """Connect to the viewer, hand it the map and the scenes, and stay until interrupted.

    Args:
        label: "scene" or "city", the prefix of the log lines.
        http: the started data server; stopped at the end.
        announce: () -> the scene or city message for the viewer.
        existing: the anchor of a map a bridge (JOSM) already streams, or None.
        anchor: the ENU anchor (lat, lon) the map is sent about.
        features: () -> the map's features, or None when this server sends no map.
        lanelet_map: () -> the lanelet2 map behind `features`.
    """
    def on_connect():
        if features:
            t0 = time.perf_counter()
            found = features()
            bridge.send({"type": "snapshot", "source": "scene_server",
                         "anchor": {"lat": anchor[0], "lon": anchor[1]}, "features": found})
            log("%s: sent %d map features in %.2fs" % (label, len(found), time.perf_counter() - t0))
        bridge.send(announce())

    # The viewer's Selection tab asks for every tag of what is selected. Ours
    # to answer only when the map in the viewer is ours; with JOSM streaming
    # it, JOSM answers and this server stays silent.
    def map_owner():
        if features:
            return lanelet_map()
        return False if existing else None

    bridge = ViewerBridge(args.viewer_host, args.ingest_port, log=log,
                          on_command=make_command_handler(map_owner, log=log))
    bridge.start(on_connect=on_connect)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("%s: stopping" % label)
    finally:
        bridge.stop()
        http.stop()
    return 0


def open_city_index(args, anchor, cache_dir, log):
    """The city index: the named city, else the one JOSM's map (`anchor`) is in.

    A map somewhere this dataset has no scenes falls back to the city with
    the most scenes; a city named on the command line does not.
    """
    from .city import CityIndex, city_of

    city_name = args.city_name or (city_of(*anchor) if anchor else None)

    def make(city):
        return CityIndex(args.city, city=city, log=log, cache_dir=cache_dir, rescan=args.rescan)
    try:
        return make(city_name)
    except ValueError:
        if args.city_name or not city_name:
            raise
        log("city: no scenes near JOSM's map (%s); serving the city with the most" % city_name)
        return make(None)


def run_city(args, log):
    cache = cache_root(args)
    # JOSM's map, when it streams one, also says which city this is about.
    existing = None if args.build_aggregates else wait_for_viewer_anchor(
        args.viewer_host, args.viewer_port, args.wait_for_map, log)
    t0 = time.perf_counter()
    index = open_city_index(args, existing, cache, log)
    log("city: %s (indexed in %.1fs: %d scenes read, %d from the local index cache)"
        % (index.describe_log(), time.perf_counter() - t0, index.read_count, index.cached_count))
    origin = index.default_anchor()
    name = aggregate_cache_name(attributes=args.aggregate_attributes)

    def aggregate_file(scene):
        return FrameDiskCache.scene_dir(cache, scene.id) / name if cache else None

    def aggregate_cached(scene):
        path = aggregate_file(scene)
        return path is not None and path.is_file()

    # Once, behind the start: does the cache still match the dataset?
    index.check_in_background()

    if args.build_aggregates:
        return build_all(index, origin, cache, aggregate_cached, args, log)

    map_file = MapFile(args.map, origin) if args.map else None
    anchor = origin
    send_map = bool(map_file) and not args.no_map
    if existing:
        anchor = existing
        send_map = False
        log("city: a bridge is already streaming a map; adopting its anchor %.7f, %.7f" % existing)
    elif map_file:
        t1 = time.perf_counter()
        anchor = map_file.bbox_centre()
        log("city: loaded %s in %.1fs" % (map_file.path.name, time.perf_counter() - t1))
    elif not args.no_map:
        log("city: no --map and no JOSM: the trajectories come without a map")
    log("city: anchor %.7f, %.7f" % anchor)

    def open_scene(entry, base_url, gate, build_gate):
        source = SceneSource(entry.dir, anchor=anchor)
        disk = FrameDiskCache(source.scene_id, cache, args.cache_limit_gb, log=log)
        return SceneHttpServer(source, allow_origin=viewer_origin(args), log=log, disk_cache=disk,
                               aggregator=make_aggregator(args, source, disk, log), listen=False,
                               base_url=base_url, gate=gate, build_gate=build_gate)

    http = CityHttpServer(index, open_scene, anchor, host=args.viewer_host,
                          port=args.http_port, allow_origin=viewer_origin(args), log=log,
                          aggregate_cached=aggregate_cached,
                          aggregate_file=aggregate_file if cache else None)
    http.start()

    def announce():
        return {"type": "city", "city": {"name": index.city, "base": http.base_url,
                                         "scenes": len(index.scenes),
                                         "anchor": {"lat": anchor[0], "lon": anchor[1]}}}

    return serve(args, log, "city", http, announce, existing, anchor=anchor,
                 features=(lambda: map_file.features(anchor)) if send_map else None,
                 lanelet_map=lambda: map_file.lanelet_map)


def build_all(index, origin, cache_dir, aggregate_cached, args, log):
    """`--build-aggregates`: every scene's static cloud into the cache, then exit."""
    todo = [s for s in index.scenes if not aggregate_cached(s)]
    log("city: %d of %d aggregates already cached, %d to build"
        % (len(index.scenes) - len(todo), len(index.scenes), len(todo)))
    t0 = time.perf_counter()
    failed = 0
    for n, entry in enumerate(todo, 1):
        t1 = time.perf_counter()
        try:
            source = SceneSource(entry.dir, anchor=origin)
            disk = FrameDiskCache(source.scene_id, cache_dir, args.cache_limit_gb, log=log)
            aggregator = make_aggregator(args, source, disk, log)
            aggregate = aggregator.build()
            aggregator.save_cached(aggregate)
        except KeyboardInterrupt:
            log("city: stopped; %d built, the rest next time" % (n - 1))
            return 1
        except Exception as exc:
            failed += 1
            log("city: %s failed (%r)" % (entry.id, exc))
            continue
        spent = time.perf_counter() - t0
        log("city: %d/%d %s: %d points in %.0fs; about %.0f min to go"
            % (n, len(todo), entry.id, len(aggregate), time.perf_counter() - t1,
               spent / n * (len(todo) - n) / 60))
    log("city: done, %d built, %d failed, %.0f min" % (len(todo) - failed, failed,
                                                       (time.perf_counter() - t0) / 60))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
