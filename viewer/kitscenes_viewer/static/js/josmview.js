// Keeping the JOSM map view and the 3D camera together.
//
// follow: the viewport feature carries JOSM's view centre; when its
//         `follow_camera` tag is set, JOSM pans translate the 3D camera.
// drive:  once the point the camera looks at (focusPoint) has travelled far
//         enough, post `set_view` so JOSM (and its cull box) recentre on it.
//         Turning the view at street level therefore pans JOSM too.
import * as THREE from "three";
import { now, round3 } from "./util.js";
import { store, VIEWPORT_ID } from "./store.js";
import { camera, camRefs, translateCamera } from "./camera.js";
import { sendCommand } from "./net.js";
import { setJosmViewStatus } from "./hud.js";

const VIEW_SYNC_MIN_M = 30;      // pan JOSM after the focus point moved this far (metres)
const VIEW_SYNC_MIN_MS = 200;
const VIEW_SYNC_TELEPORT_M = 80; // ignore F/B/frame jumps; do not pan JOSM
const FOCUS_MAX_M = 150;         // looking toward the horizon: focus at most this far ahead

let lastFollowCenter = null;
let lastViewSyncXY = null;   // last focus point XY that we posted (or adopted)
let ignoreFollowUntil = 0;   // skip JOSM→3D follow after we drove the view
let lastViewSyncT = 0;
let isBusy = () => false;    // a drag owns the camera

export function installJosmView({ busy }) {
  isBusy = busy;
  store.on("upsert", (f) => {
    if (f.id !== VIEWPORT_ID) return;
    applyViewportCameraFollow(f);
    noteJosmViewport(f);
  });
  // A snapshot clears the store before its features (and viewport) arrive.
  store.on("clear", reset);
}

const _dir = new THREE.Vector3();

/**
 * XY of the map point the camera looks at: where the view axis meets the map
 * height near the cursor (camRefs.groundZ), at most FOCUS_MAX_M ahead. Looking
 * straight down that is the camera's own XY; looking level or up, FOCUS_MAX_M
 * ahead.
 */
export function focusPoint() {
  camera.getWorldDirection(_dir);
  const o = camera.position;
  const h = Math.hypot(_dir.x, _dir.y);
  if (h < 1e-9) return [o.x, o.y];
  let reach = FOCUS_MAX_M;
  if (_dir.z < 0) reach = Math.min(FOCUS_MAX_M, Math.max(0, ((camRefs.groundZ - o.z) / _dir.z) * h));
  return [o.x + (_dir.x / h) * reach, o.y + (_dir.y / h) * reach];
}

function reset() {
  lastFollowCenter = null;
  lastViewSyncXY = null;
}

function applyViewportCameraFollow(feature) {
  const follow = feature.tags && feature.tags.follow_camera === "1";
  if (!feature.center || feature.center.length < 2) {
    lastFollowCenter = null;
    return;
  }
  const cx = feature.center[0];
  const cy = feature.center[1];
  const cz = feature.center.length > 2 ? feature.center[2] : 0;

  // Record the JOSM/cull centre even when we are not translating the 3D camera
  // (drive-view echo, follow off, or a drag in progress).
  if (!follow || isBusy() || now() < ignoreFollowUntil || lastFollowCenter === null) {
    lastFollowCenter = [cx, cy, cz];
    return;
  }
  const dx = cx - lastFollowCenter[0];
  const dy = cy - lastFollowCenter[1];
  const dz = cz - lastFollowCenter[2];
  lastFollowCenter = [cx, cy, cz];
  if (Math.abs(dx) < 1e-4 && Math.abs(dy) < 1e-4 && Math.abs(dz) < 1e-4) return;
  // Translate the camera only, so heading, pitch and height above the map stay.
  translateCamera(dx, dy, dz);
  lastViewSyncXY = focusPoint();
}

function noteJosmViewport(feature) {
  if (!feature.center || feature.center.length < 2) return;
  const cx = feature.center[0];
  const cy = feature.center[1];
  const [fx, fy] = focusPoint();
  const d = Math.hypot(cx - fx, cy - fy);
  setJosmViewStatus(`JOSM centre (${cx.toFixed(1)}, ${cy.toFixed(1)})  Δfocus ${d.toFixed(1)} m`);
}

/** Post `set_view` to centre JOSM on the focus point. */
export function requestJosmRecenter(reason, force) {
  const f = focusPoint();
  const x = round3(f[0]);
  const y = round3(f[1]);
  lastViewSyncXY = f;
  lastViewSyncT = now();
  lastFollowCenter = [x, y, lastFollowCenter ? lastFollowCenter[2] : 0];
  ignoreFollowUntil = now() + 2500;
  const op = { op: "set_view", x, y };
  if (force) op.force = true;
  setJosmViewStatus(`${reason}: POST (${x}, ${y})…`);
  return sendCommand([op]).then((info) => {
    if (!info || info.error) {
      setJosmViewStatus(`${reason}: POST failed (${info && info.error ? info.error : "no response"})`);
    } else if (!info.delivered) {
      setJosmViewStatus(`${reason}: server got it, delivered=0 (JOSM hook not connected)`);
    } else {
      setJosmViewStatus(`${reason}: delivered=${info.delivered} — watch JOSM map + gray rect`);
    }
    return info;
  });
}

/** Per frame: drive JOSM once the focus point travelled far enough. */
export function maybeSyncJosmView() {
  const [x, y] = focusPoint();
  if (lastViewSyncXY === null) {
    lastViewSyncXY = [x, y];
    return;
  }
  const dist = Math.hypot(x - lastViewSyncXY[0], y - lastViewSyncXY[1]);
  if (dist > VIEW_SYNC_TELEPORT_M) {
    lastViewSyncXY = [x, y];
    return;
  }
  if (dist < VIEW_SYNC_MIN_M) return;
  if (now() - lastViewSyncT < VIEW_SYNC_MIN_MS) return;
  requestJosmRecenter(`auto ${Math.round(dist)} m`, false);
}
