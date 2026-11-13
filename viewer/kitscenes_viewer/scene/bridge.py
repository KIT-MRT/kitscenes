"""Ingest-socket client: the scene server as just another viewer bridge.

The viewer server (`server.py`) accepts newline-delimited JSON on its ingest
port and fans it out to every browser tab; it also forwards browser commands
back to every connected bridge. JOSM is one such bridge, this is another, so
the browser cannot tell them apart and nothing in the viewer had to learn
about scenes.

Commands that would edit the map are answered with a refusal, which the viewer
already knows how to show: it puts the moved nodes back and prints the reason.
Editing needs JOSM.
"""

import json
import socket
import threading
import time

READ_ONLY_MESSAGE = ("The 3D scene server is read-only; "
                     "connect JOSM to edit this map")
# Commands that only move the camera or ask for a selection are harmless.
HARMLESS_OPS = {"set_view", "select"}


class ViewerBridge:
    """Keeps a connection to the viewer's ingest port, reconnecting as needed."""

    def __init__(self, host="127.0.0.1", port=8766, on_command=None, log=print):
        self.host = host
        self.port = port
        self.on_command = on_command
        self.log = log
        self._sock = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = None
        self._on_connect = None

    def start(self, on_connect=None):
        """Connect (retrying) and serve the command back-channel in a thread."""
        self._on_connect = on_connect
        self._reader = threading.Thread(target=self._run, name="ll2-scene-bridge", daemon=True)
        self._reader.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def send(self, message):
        """One message, newline-delimited. False when not connected."""
        line = json.dumps(message, separators=(",", ":")) + "\n"
        with self._lock:
            if not self._sock:
                return False
            try:
                self._sock.sendall(line.encode("utf-8"))
                return True
            except OSError as exc:
                self.log("scene: lost the viewer connection (%s)" % exc)
                self._sock = None
                return False

    def connected(self):
        with self._lock:
            return self._sock is not None

    def _run(self):
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5)
            except OSError:
                time.sleep(1.0)
                continue
            sock.settimeout(None)
            with self._lock:
                self._sock = sock
            self.log("scene: connected to the viewer at %s:%d" % (self.host, self.port))
            if self._on_connect:
                try:
                    self._on_connect()
                except Exception as exc:  # a bad snapshot must not kill the bridge
                    self.log("scene: sending the scene failed: %r" % exc)
            self._read_until_closed(sock)
            with self._lock:
                self._sock = None
            if not self._stop.is_set():
                self.log("scene: viewer connection closed, retrying")
                time.sleep(1.0)

    def _read_until_closed(self, sock):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._handle(line)

    def _handle(self, line):
        try:
            command = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        # Anything that can POST /command to the viewer reaches this line; a
        # malformed one must not end the reader (and with it the reconnects).
        if not isinstance(command, dict):
            return
        reply = None
        if self.on_command:
            try:
                reply = self.on_command(command)
            except Exception as exc:
                reply = {"ok": False, "message": "scene server error: %r" % exc}
        if reply is None:
            reply = default_reply(command)
        if reply is not False and command.get("id"):
            self.send(dict(reply, type="command_result", id=command["id"]))


def default_reply(command):
    """Accept camera/selection traffic, refuse anything that would edit."""
    ops = command.get("ops") or []
    if not isinstance(ops, list):
        return {"ok": False, "message": READ_ONLY_MESSAGE}
    if ops and all(isinstance(op, dict) and op.get("op") in HARMLESS_OPS for op in ops):
        return {"ok": True, "message": "ok"}
    return {"ok": False, "message": READ_ONLY_MESSAGE}
