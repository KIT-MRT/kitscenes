#!/usr/bin/env python3
"""
The viewer server: the page and the live scene (separate process, stdlib only).

    bridge (scene server, JOSM)  --TCP newline JSON-->  THIS SERVER  --SSE-->  browser tab (Three.js)

Two listeners run in this one process:

  * TCP ingest (default 127.0.0.1:8766): a bridge connects and streams
    newline-delimited JSON messages (snapshot / patch / clear / scene / city).
  * HTTP + SSE (default 127.0.0.1:8765): serves the static Three.js page and a
    Server-Sent-Events stream (/events) that pushes the live scene to every
    open browser tab. A late-joining tab immediately receives a full snapshot.

SSE is one-directional (server -> browser), which is all the viewer needs, and
it works with nothing but the Python standard library.

Run:
    python3 server.py                 # starts both servers, opens a browser tab
    python3 server.py --no-browser    # don't auto-open the browser
    python3 server.py --http-port 8765 --ingest-port 8766

The protocol (see the PROTOCOL section below) is tiny and additive.
"""

import argparse
import json
import os
import queue
import re
import socketserver
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import parse_qs, unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
VIEWER_DIR = Path(__file__).resolve().parent

# Content hash of the shipped viewer, written next to this file by the plugin
# when it extracts the jar. /healthz reports it, so a JOSM running a different
# plugin build recycles this process instead of adopting it: an adopted server
# keeps its own routes and may serve a different extract. None when run from
# a source tree.
BUILD_FILE = VIEWER_DIR / ".shipped_version"
BUILD_ID = None

# --standalone: the page is the KITScenes viewer from its first paint, without
# the Edit mode and the JOSM controls. JOSM starts the server without it.
STANDALONE = False
_STANDALONE_PAGE = (
    ("<title>Lanelet2 3D Viewer</title>", "<title>KITScenes viewer</title>"),
    ("<body>", '<body class="standalone">'),
    ('class="brandName">Lanelet2 3D</span>', 'class="brandName">KITScenes viewer</span>'),
)


def index_page(standalone):
    """index.html as bytes; with `standalone` in its KITScenes look (see `setStandalone` in app.js)."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    if standalone:
        for old, new in _STANDALONE_PAGE:
            html = html.replace(old, new, 1)
    return html.encode("utf-8")


# Populated in main(). Directory of the traffic-element icons, served at
# /style_images/<filename>. None if there is no such directory.
ICONS_DIR = None

# Basename-only icon names (German StVO files like 274-30.png, 264-2.3.png).
_ICON_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(png|svg|jpe?g|gif|webp)$", re.IGNORECASE)
_ICON_CONTENT_TYPES = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def resolve_icons_dir(explicit=None):
    """`explicit`, else the package's own style_images directory; None when it is not a directory."""
    path = Path(os.path.expanduser(explicit)).resolve() if explicit else VIEWER_DIR / "style_images"
    return path if path.is_dir() else None


# Profiling is opt-in: --profile flag or VIEWER3D_PROFILE=1. When on, the server
# prints a one-line timing breakdown for ingest parse, scene apply, SSE fan-out,
# per-client serialization, and command forwarding, so large-map slowness can be
# attributed to a stage rather than guessed at.
PROFILE = bool(os.environ.get("VIEWER3D_PROFILE"))


def _plog(msg):
    print("[viewer][perf] " + msg, flush=True)


def _sse_encode(msg):
    """Serialize a message to its full SSE wire form, once, for all clients."""
    payload = json.dumps(msg, separators=(",", ":"))
    return ("data: %s\n\n" % payload).encode("utf-8")

# ---------------------------------------------------------------------------
# PROTOCOL
# ---------------------------------------------------------------------------
# Messages are JSON objects. Over TCP they are newline-delimited (one per line).
# Over SSE each message is sent as a single `data:` event.
#
#   snapshot : full replacement of the scene
#     {"type":"snapshot",
#      "anchor":{"lat":48.13,"lon":11.57},
#      "features":[<feature>, ...]}
#     An optional "source":"scene_server" says the map is the scene server's
#     read-only copy, so the browser offers no editing; it is kept and sent
#     with the bootstrap snapshot until the next snapshot replaces it.
#
#   patch    : incremental change
#     {"type":"patch","ops":[
#        {"op":"upsert","feature":<feature>},
#        {"op":"remove","id":"way/-123"},
#        {"op":"anchor","lat":48.13,"lon":11.57}]}
#
#   clear    : empty the scene
#     {"type":"clear"}
#
#   selection: JOSM's selection (kept, and sent with the bootstrap snapshot)
#     {"type":"selection","nodes":[17,-5],"ways":["way/3"],"truncated":false}
#
#   scene    : a sensor scene offered by a second bridge (the scene server),
#              kept and sent with the bootstrap snapshot like the selection.
#              The browser fetches the heavy data from `scene.base` itself.
#     {"type":"scene","scene":{"id":..,"base":"http://127.0.0.1:8767",
#                              "frames":220,"cameras":{..},"ego":{..}}}
#
#   city     : every scene of a city, offered by the scene server in city
#              mode; kept and replayed the same way. Trajectories and each
#              scene's data come from `city.base`.
#     {"type":"city","city":{"name":"frankfurt","base":"http://127.0.0.1:8767",
#                            "scenes":117,"anchor":{"lat":..,"lon":..}}}
#
#   command_result: JOSM's answer to a browser command that had an "id"
#     {"type":"command_result","id":"c12","ok":false,"message":"..."}
#     The answer to a `describe` op adds "details": [{"id":"way/3",
#     "tags":{..},"nodes":5,"memberships":[{"id":"relation/7","role":"left",
#     "tags":{..}}]}, ...] (relayed untouched, like the rest). A membership
#     through a lanelet (its traffic light, for one of its bounds) adds
#     "via":"relation/7"; a relation's entry adds "members" in the same shape.
#
#   feature  : one renderable object (protocol v2: flat arrays)
#     {"id":"way/-123",
#      "kind":"line",
#      "tags":{"type":"line_thin","subtype":"dashed"},
#      "pts":[x,y,z, x,y,z, ...],       # local ENU metres relative to anchor
#      "nodes":[-5, 17, ...],           # JOSM node unique id per vertex
#      "color":"#ffffff"}               # optional; browser falls back to tags
#     The server never looks inside a feature.
#
# The browser overlays style_images icons for traffic_sign / traffic_light /
# arrow / symbol from `tags` (see static/app.js). The server just serves the
# files at GET /style_images/<filename>.
# ---------------------------------------------------------------------------


class SceneHub:
    """Authoritative in-memory scene + fan-out to SSE subscribers.

    The hub keeps the latest full scene so any newly connected browser can be
    bootstrapped with a snapshot, then receives subsequent messages live.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._anchor = None                 # {"lat":..,"lon":..} or None
        self._source = None                 # the snapshot's "source", if any
        self._features = {}                 # id -> feature dict
        self._selection = None              # last JOSM selection message
        self._scene = None                  # last sensor-scene announcement
        self._city = None                   # last city announcement
        self._seq = 0                       # monotonically increasing
        self._subscribers = set()           # set[queue.Queue]
        self._bridges = set()               # set[_BridgeConn] (command back-channel)

    # -- subscription (SSE side) -------------------------------------------
    def subscribe(self):
        q = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.add(q)
            bootstrap = self._snapshot_message_locked()
        # Hand the new client a full snapshot first (pre-encoded SSE bytes).
        q.put(_sse_encode(bootstrap))
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subscribers.discard(q)

    # -- command back-channel (browser -> JOSM bridge) ---------------------
    def add_bridge(self, conn):
        with self._lock:
            self._bridges.add(conn)

    def remove_bridge(self, conn):
        with self._lock:
            self._bridges.discard(conn)

    def send_to_bridges(self, line):
        """Forward a command line to every connected bridge. Returns count."""
        with self._lock:
            bridges = list(self._bridges)
        delivered = 0
        for conn in bridges:
            if conn.send(line):
                delivered += 1
            else:
                self.remove_bridge(conn)
        return delivered

    # -- ingest (TCP side) -------------------------------------------------
    def handle_incoming(self, msg):
        """Apply an inbound message to the scene and broadcast it."""
        t0 = time.perf_counter() if PROFILE else 0.0
        with self._lock:
            self._apply_locked(msg)
            self._seq += 1
            msg = dict(msg)
            msg["seq"] = self._seq
            subscribers = list(self._subscribers)
            feat_count = len(self._features)
        t1 = time.perf_counter() if PROFILE else 0.0
        # Serialize ONCE for every subscriber instead of per client.
        data = _sse_encode(msg)
        t2 = time.perf_counter() if PROFILE else 0.0
        self._broadcast(subscribers, data)
        if PROFILE:
            mtype = msg.get("type")
            extra = ""
            if mtype == "patch":
                extra = " ops=%d" % len(msg.get("ops", []))
            _plog("ingest type=%s seq=%d feats=%d subs=%d%s bytes=%d apply=%.2fms "
                  "serialize=%.2fms fanout=%.2fms"
                  % (mtype, self._seq, feat_count, len(subscribers), extra,
                     len(data), (t1 - t0) * 1000.0, (t2 - t1) * 1000.0,
                     (time.perf_counter() - t2) * 1000.0))

    def _apply_locked(self, msg):
        mtype = msg.get("type")
        if mtype == "snapshot":
            self._anchor = msg.get("anchor")
            # Who sent the map: "scene_server" marks a map nobody can edit.
            # JOSM sends no source, which clears an earlier one.
            source = msg.get("source")
            self._source = source if isinstance(source, str) else None
            self._features = {
                f["id"]: f for f in msg.get("features", []) if "id" in f
            }
        elif mtype == "patch":
            for op in msg.get("ops", []):
                kind = op.get("op")
                if kind == "upsert":
                    feat = op.get("feature")
                    if feat and "id" in feat:
                        self._features[feat["id"]] = feat
                elif kind == "remove":
                    self._features.pop(op.get("id"), None)
                elif kind == "anchor":
                    self._anchor = {"lat": op.get("lat"), "lon": op.get("lon")}
        elif mtype == "clear":
            self._features = {}
        elif mtype == "selection":
            # Kept so a browser that (re)connects starts with JOSM's selection.
            self._selection = {"nodes": msg.get("nodes", []), "ways": msg.get("ways", [])}
        elif mtype == "scene":
            # Same reason: a reloaded tab must find the scene server again.
            if isinstance(msg.get("scene"), dict):
                self._scene = msg["scene"]
        elif mtype == "city":
            if isinstance(msg.get("city"), dict):
                self._city = msg["city"]

    def _snapshot_message_locked(self):
        msg = {
            "type": "snapshot",
            "seq": self._seq,
            "anchor": self._anchor,
            "features": list(self._features.values()),
        }
        if self._source is not None:
            msg["source"] = self._source
        if self._selection is not None:
            msg["selection"] = self._selection
        if self._scene is not None:
            msg["scene"] = self._scene
        if self._city is not None:
            msg["city"] = self._city
        return msg

    def snapshot_message(self):
        with self._lock:
            return self._snapshot_message_locked()

    def _broadcast(self, subscribers, data):
        for q in subscribers:
            try:
                q.put_nowait(data)
            except queue.Full:
                # Slow client: drop it rather than stall ingest.
                self.unsubscribe(q)


HUB = SceneHub()


class _BridgeConn:
    """Thread-safe writer for the bridge TCP socket (command back-channel).

    The bridge connection is full-duplex: the bridge streams telemetry on the
    read side while the server writes newline-delimited command messages back
    on the write side. Writes may originate from any HTTP worker thread, so
    they are serialized with a lock.
    """

    def __init__(self, wfile):
        self._wfile = wfile
        self._lock = threading.Lock()
        self._alive = True

    def send(self, line):
        with self._lock:
            if not self._alive:
                return False
            try:
                self._wfile.write((line + "\n").encode("utf-8"))
                self._wfile.flush()
                return True
            except OSError:
                self._alive = False
                return False


# ---------------------------------------------------------------------------
# TCP ingest: newline-delimited JSON from the JOSM bridge
# ---------------------------------------------------------------------------
# A browser can POST to the ingest port too (no-cors): its request line and
# headers are not JSON, but the body would be. Bridges never speak HTTP.
_HTTP_REQUEST_LINE = re.compile(rb"^[A-Z]+ \S+ HTTP/\d")


def _reject_constant(name):
    # NaN/Infinity: Python's json reads them, the browser's JSON.parse does
    # not -- one stored in the bootstrap snapshot would blank every new tab.
    raise ValueError("non-JSON number %s" % name)


class IngestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        peer = self.client_address
        print("[ingest] connected: %s:%s" % peer, flush=True)
        conn = _BridgeConn(self.wfile)
        HUB.add_bridge(conn)
        try:
            for raw in self.rfile:  # iterates by line; blocks until disconnect
                line = raw.strip()
                if not line:
                    continue
                if _HTTP_REQUEST_LINE.match(line):
                    print("[ingest] an HTTP request on the ingest port; closing", flush=True)
                    return
                t0 = time.perf_counter() if PROFILE else 0.0
                try:
                    msg = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                except (ValueError, UnicodeDecodeError) as exc:
                    print("[ingest] bad JSON line ignored: %s" % exc, flush=True)
                    continue
                if not isinstance(msg, dict):
                    print("[ingest] non-object message ignored", flush=True)
                    continue
                if PROFILE:
                    _plog("recv bytes=%d parse=%.2fms"
                          % (len(line), (time.perf_counter() - t0) * 1000.0))
                HUB.handle_incoming(msg)
        except (ConnectionError, OSError) as exc:
            print("[ingest] connection error: %s" % exc, flush=True)
        finally:
            HUB.remove_bridge(conn)
            print("[ingest] disconnected: %s:%s" % peer, flush=True)


class ThreadingTCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# HTTP + SSE: static files and the live event stream to browsers
# ---------------------------------------------------------------------------
SSE_HEARTBEAT_SECONDS = 15.0


def _shutdown_soon():
    """Flush the /shutdown response, then exit hard.

    The accept loops sit on daemon threads with no clean stop signal and this
    process owns nothing that needs unwinding.
    """
    time.sleep(0.2)
    os._exit(0)


# --- tile proxy -------------------------------------------------------------------
# WebGL can only texture with images whose server allows cross-origin use
# (CORS). OpenStreetMap does; some tile and WMS servers do not. For those the
# page can fetch tiles through this server instead -- but only from hosts the
# user named (--tile-proxy-host, or LL2_TILE_PROXY_HOSTS for a server JOSM
# starts), never anything else, and without following redirects elsewhere.
TILE_PROXY_HOSTS = set()
TILE_PROXY_MAX_BYTES = 8 * 1024 * 1024
TILE_PROXY_TIMEOUT_S = 20        # per socket read
TILE_PROXY_DEADLINE_S = 45       # for the whole tile
# Raster images only: SVG is an image/ type that can carry script, and the
# proxy answers from this viewer's own origin.
TILE_PROXY_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
# A page firing hundreds of proxy requests must not tie up every thread.
_TILE_SLOTS = threading.BoundedSemaphore(8)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # a redirect could lead off the allowed host


_TILE_OPENER = urllib.request.build_opener(_NoRedirect)


def tile_proxy_allowed(url):
    """True only for http(s) URLs on a host the user allowed.

    An entry `host:port` allows that port; a bare `host` only the scheme's
    default port, so allowing a host does not open every service on it.
    URLs with user info are refused outright.
    """
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return False
    host = parsed.hostname.lower()
    default = 80 if parsed.scheme == "http" else 443
    if port is not None and "%s:%d" % (host, port) in TILE_PROXY_HOSTS:
        return True
    return host in TILE_PROXY_HOSTS and port in (None, default)


def _clean(text, limit=300):
    """Upstream text for an error body: printable, short."""
    return "".join(c if c.isprintable() else " " for c in str(text))[:limit]


def parse_proxy_hosts(values):
    hosts = set()
    for value in values:
        for part in (value or "").replace(";", ",").split(","):
            part = part.strip().lower()
            if part:
                hosts.add(part)
    return hosts


# Who may talk to this server. It listens on loopback, but any web page the
# user opens can send it requests, and one that rebinds its own DNS name to
# 127.0.0.1 could even read the answers. So: the Host must be a name this
# server is reached by (DNS rebinding), and requests a browser marks as
# cross-site may not act or read data (CSRF). JOSM's own HTTP client sends
# neither header and is unaffected.
LOCAL_NAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}
BIND_HOST = "127.0.0.1"
_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "[::]"}
_GUARDED_PATHS = ("/state", "/events", "/tileproxy", "/command", "/shutdown")


def host_header_ok(value):
    if BIND_HOST in _WILDCARD_HOSTS:
        return True  # deliberately exposed to the network; any name may reach it
    if not value:
        return True  # HTTP/1.0 clients
    host = value.strip().lower()
    if host.startswith("["):
        host = host[:host.find("]") + 1]
    else:
        host = host.rsplit(":", 1)[0]
    return host in LOCAL_NAMES or host == BIND_HOST.lower()


class HttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _allowed(self, path):
        """Refuse foreign Host names everywhere, cross-site requests on data routes."""
        if not host_header_ok(self.headers.get("Host")):
            self.send_error(403, "Unknown host name")
            return False
        if path.startswith(_GUARDED_PATHS):
            if (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
                self.send_error(403, "Cross-site request refused")
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "null":
                try:
                    o = urlparse(origin)
                    same = host_header_ok(o.netloc) and (o.port or 80) == self.server.server_address[1]
                except ValueError:
                    same = False
                if not same:
                    self.send_error(403, "Cross-origin request refused")
                    return False
        return True

    def log_message(self, fmt, *args):  # quieter default logging
        if self.path != "/events":
            super().log_message(fmt, *args)

    def do_GET(self):
        path = urlparse(self.path).path
        if not self._allowed(path):
            return
        if path == "/events":
            self._serve_sse()
        elif path in ("/", "/index.html"):
            self._serve_static("index.html", "text/html; charset=utf-8",
                               read=lambda: index_page(STANDALONE))
        elif path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
        elif path == "/app.css":
            self._serve_static("app.css", "text/css; charset=utf-8")
        elif path.startswith("/vendor/"):
            self._serve_vendor(unquote(path[len("/vendor/"):]))
        elif path.startswith("/js/"):
            self._serve_module(unquote(path[len("/js/"):]))
        elif path == "/state":
            self._serve_json(HUB.snapshot_message())
        elif path == "/healthz":
            self._serve_json({"ok": True, "build": BUILD_ID})
        elif path.startswith("/style_images/"):
            name = unquote(path[len("/style_images/"):])
            self._serve_icon(name)
        elif path == "/tileproxy/hosts":
            self._serve_json({"hosts": sorted(TILE_PROXY_HOSTS)})
        elif path == "/tileproxy":
            self._serve_tile_proxy()
        else:
            self.send_error(404, "Not found")

    def _serve_tile_proxy(self):
        query = parse_qs(urlparse(self.path).query)
        url = (query.get("url") or [""])[0]
        if not tile_proxy_allowed(url):
            self.send_error(403, "Host not allowed for the tile proxy",
                            "Start the viewer with --tile-proxy-host <host> to allow one.")
            return
        if not _TILE_SLOTS.acquire(timeout=TILE_PROXY_TIMEOUT_S):
            self.send_error(503, "Tile proxy busy")
            return
        try:
            self._fetch_tile(url)
        finally:
            _TILE_SLOTS.release()

    def _fetch_tile(self, url):
        request = urllib.request.Request(url, headers={"User-Agent": "lanelet2-josm-viewer3d"})
        deadline = time.monotonic() + TILE_PROXY_DEADLINE_S
        try:
            with _TILE_OPENER.open(request, timeout=TILE_PROXY_TIMEOUT_S) as response:
                ctype = (response.headers.get("Content-Type", "").split(";")[0].strip().lower())
                expected = response.headers.get("Content-Length")
                parts = []
                size = 0
                while True:
                    if time.monotonic() > deadline:
                        self.send_error(504, "Tile server too slow")
                        return
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > TILE_PROXY_MAX_BYTES:
                        self.send_error(502, "Tile too large")
                        return
                    parts.append(chunk)
                body = b"".join(parts)
        except urllib.error.HTTPError as exc:
            self.send_error(502 if exc.code < 400 else exc.code, "Tile server error",
                            _clean(exc.reason))
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.send_error(502, "Tile server unreachable", _clean(exc))
            return
        if expected is not None and expected.isdigit() and int(expected) != len(body):
            self.send_error(502, "Tile cut short")  # not worth caching a broken tile
            return
        if ctype not in TILE_PROXY_TYPES:
            # A WMS answers errors with an XML ServiceException; its text goes
            # into the error body so the browser console says what was wrong.
            self.send_error(502, "Not a raster image",
                            "%s: %s" % (_clean(ctype, 60), _clean(body[:300].decode("utf-8", "replace"))))
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; sandbox")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._allowed(urlparse(self.path).path):
            return
        if self.path == "/command":
            self._handle_command()
        elif self.path == "/shutdown":
            self._handle_shutdown()
        else:
            self.send_error(404, "Not found")

    def _handle_shutdown(self):
        """Stop this process; lets a JOSM that did not start it free the port."""
        self._serve_json({"ok": True})
        threading.Thread(target=_shutdown_soon, daemon=True).start()

    def _handle_command(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""
        try:
            msg = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Bad JSON")
            return
        if not isinstance(msg, dict) or msg.get("type") != "command":
            self.send_error(400, "Expected {\"type\":\"command\",...}")
            return
        t0 = time.perf_counter() if PROFILE else 0.0
        line = json.dumps(msg, separators=(",", ":"))
        delivered = HUB.send_to_bridges(line)
        self._serve_json({"ok": delivered > 0, "delivered": delivered})
        if PROFILE:
            _plog("command ops=%d bytes=%d delivered=%d forward=%.2fms"
                  % (len(msg.get("ops", [])), len(line), delivered,
                     (time.perf_counter() - t0) * 1000.0))

    # -- helpers -----------------------------------------------------------
    def _serve_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_vendor(self, rel):
        # Version-pinned third-party ES modules (three.js and its controls),
        # shipped so the viewer works without network access.
        self._serve_contained(STATIC_DIR / "vendor", rel,
                              "public, max-age=31536000, immutable")

    def _serve_module(self, rel):
        # The viewer's own ES modules (static/js/). Uncached like app.js so a
        # restarted server never pairs a new page with old modules.
        if not rel.endswith(".js"):
            self.send_error(404, "Not found")
            return
        self._serve_contained(STATIC_DIR / "js", rel, "no-store")

    def _serve_contained(self, base_dir, rel, cache_control):
        """Serve `rel` under `base_dir`, refusing anything that resolves outside."""
        base = base_dir.resolve()
        try:
            target = (base / rel).resolve()
        except OSError:
            self.send_error(404, "Not found")
            return
        if target != base and base not in target.parents:
            self.send_error(403, "Forbidden")
            return
        try:
            body = target.read_bytes()
        except OSError:
            self.send_error(404, "Missing file: %s" % rel)
            return
        ctype = ("application/javascript; charset=utf-8"
                 if target.suffix == ".js" else "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, name, content_type, read=None):
        path = STATIC_DIR / name
        try:
            body = read() if read else path.read_bytes()
        except OSError:
            self.send_error(404, "Missing static file: %s" % name)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_icon(self, name):
        """Serve a basename from ICONS_DIR. Rejects path traversal."""
        if ICONS_DIR is None:
            self.send_error(404, "No style_images directory")
            return
        if not name or not _ICON_NAME_RE.match(name):
            self.send_error(400, "Bad icon name")
            return
        candidate = ICONS_DIR / name
        try:
            resolved = candidate.resolve()
            resolved.relative_to(ICONS_DIR)
        except (OSError, ValueError):
            self.send_error(404, "Not found")
            return
        if not resolved.is_file():
            self.send_error(404, "Not found")
            return
        content_type = _ICON_CONTENT_TYPES.get(
            resolved.suffix.lower(), "application/octet-stream")
        try:
            body = resolved.read_bytes()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        q = HUB.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                try:
                    data = q.get(timeout=SSE_HEARTBEAT_SECONDS)
                except queue.Empty:
                    # Comment line keeps the connection (and proxies) alive.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                # `data` is already-encoded SSE bytes (serialized once by the
                # hub for all clients); this thread only writes to its socket.
                t0 = time.perf_counter() if PROFILE else 0.0
                self.wfile.write(data)
                self.wfile.flush()
                if PROFILE:
                    _plog("sse client=%s bytes=%d write=%.2fms"
                          % (self.client_address[1], len(data),
                             (time.perf_counter() - t0) * 1000.0))
        except (ConnectionError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="The viewer server: the 3D page and its live scene")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; keep local)")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--ingest-port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not auto-open a browser tab")
    parser.add_argument("--standalone", action="store_true",
                        help="serve the page as the KITScenes viewer, without the JOSM controls")
    parser.add_argument("--profile", action="store_true",
                        help="print per-stage timing logs (or set VIEWER3D_PROFILE=1)")
    parser.add_argument("--icons-dir", default=None,
                        help="directory of style_images to serve at /style_images/ "
                             "(default: the package's own)")
    parser.add_argument("--tile-proxy-host", action="append", default=[],
                        help="let the page fetch map tiles from this host (host or "
                             "host:port) through this server, for tile servers that do "
                             "not allow cross-origin use. Repeatable; also read from "
                             "LL2_TILE_PROXY_HOSTS (comma separated)")
    args = parser.parse_args(argv)

    global PROFILE, ICONS_DIR, BUILD_ID, TILE_PROXY_HOSTS, BIND_HOST, STANDALONE
    BIND_HOST = args.host
    STANDALONE = args.standalone
    TILE_PROXY_HOSTS = parse_proxy_hosts(
        list(args.tile_proxy_host) + [os.environ.get("LL2_TILE_PROXY_HOSTS", "")])
    PROFILE = PROFILE or args.profile
    try:
        BUILD_ID = BUILD_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        BUILD_ID = None
    if PROFILE:
        print("[viewer] profiling ON", flush=True)

    ICONS_DIR = resolve_icons_dir(args.icons_dir)
    if args.icons_dir and ICONS_DIR is None:
        print("[viewer] --icons-dir is not a directory: %s" % args.icons_dir,
              flush=True)
        return 2

    ingest = ThreadingTCPServer((args.host, args.ingest_port), IngestHandler)
    http = ThreadingHTTPServer((args.host, args.http_port), HttpHandler)

    threading.Thread(target=ingest.serve_forever, name="ingest",
                     daemon=True).start()
    threading.Thread(target=http.serve_forever, name="http",
                     daemon=True).start()

    url = "http://%s:%d/" % (args.host, args.http_port)
    print("[viewer] HTTP/SSE : %s" % url, flush=True)
    print("[viewer] ingest   : tcp://%s:%d (bridges connect here)"
          % (args.host, args.ingest_port), flush=True)
    if TILE_PROXY_HOSTS:
        print("[viewer] tiles    : proxying %s" % ", ".join(sorted(TILE_PROXY_HOSTS)), flush=True)
    if ICONS_DIR is not None:
        print("[viewer] icons    : %s" % ICONS_DIR, flush=True)
    else:
        print("[viewer] icons    : none (traffic-element icons disabled)",
              flush=True)

    if not args.no_browser:
        # Open after a short delay so the server is definitely accepting.
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[viewer] shutting down", flush=True)
        ingest.shutdown()
        http.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
