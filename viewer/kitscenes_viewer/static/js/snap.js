// Snapping for gizmo moves.
//
//   height:   the moved height sticks to the height of the nearest other node
//             (horizontally, within SNAP_NODE_RADIUS_M) or to the lanelet
//             surface under the point, when it comes within SNAP_PX on screen.
//   position: a single dragged node sticks onto another node it comes within
//             SNAP_PX of on screen (its x, y and z; the nodes stay separate).
//
// A marker shows what it snapped to.
import * as THREE from "three";
import { store, TILE_M } from "./store.js";
import { camera, pixelsToMetres } from "./camera.js";
import { nodesUnderCursor, screenProjector, projectToPage } from "./picking.js";
import { laneletLayer } from "./render/lanelets.js";
import { screenSphere } from "./render/overlays.js";

export const SNAP_PX = 10;
const SNAP_NODE_RADIUS_M = 10;
const SNAP_COLOR = 0xff5ec4;

export const snapMarker = screenSphere(SNAP_COLOR, 6, 1003, 0.95);
const down = new THREE.Raycaster();
const _o = new THREE.Vector3();
const _d = new THREE.Vector3(0, 0, -1);

/** Metres that SNAP_PX pixels span at the point. */
function tolerance(x, y, z) {
  return pixelsToMetres(SNAP_PX, Math.max(camera.near, camera.position.distanceTo(_o.set(x, y, z))));
}

/** The nearest node not in `exclude`, horizontally within SNAP_NODE_RADIUS_M of (x, y). */
function nearestNode(x, y, exclude) {
  const tx = Math.floor(x / TILE_M), ty = Math.floor(y / TILE_M);
  let best = null;
  let bestD2 = SNAP_NODE_RADIUS_M * SNAP_NODE_RADIUS_M;
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      const tile = store.tiles.get(`${tx + i},${ty + j}`);
      if (!tile) continue;
      for (const f of tile.features) {
        const n = Math.min(f.nodes.length, f.pos.length / 3);
        for (let k = 0; k < n; k++) {
          if (exclude.has(f.nodes[k])) continue;
          const dx = f.pos[k * 3] - x, dy = f.pos[k * 3 + 1] - y;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) {
            bestD2 = d2;
            best = { id: f.nodes[k], x: f.pos[k * 3], y: f.pos[k * 3 + 1], z: f.pos[k * 3 + 2] };
          }
        }
      }
    }
  }
  return best;
}

/** Height of the lanelet surface under (x, y) closest to z, or null. */
function surfaceZ(x, y, z) {
  const meshes = laneletLayer.surfaceMeshes();
  if (!meshes.length) return null;
  down.set(_o.set(x, y, z + 1000), _d);
  let best = null;
  for (const h of down.intersectObjects(meshes, false)) {
    if (best === null || Math.abs(h.point.z - z) < Math.abs(best - z)) best = h.point.z;
  }
  return best;
}

/**
 * A height to snap `z` at (x, y) to, or null: { z, what }.
 * `exclude` holds the node ids being moved.
 */
export function snapHeight(x, y, z, exclude) {
  const tol = tolerance(x, y, z);
  const candidates = [];
  const n = nearestNode(x, y, exclude);
  if (n) candidates.push({ z: n.z, what: `node/${n.id}` });
  const s = surfaceZ(x, y, z);
  if (s !== null) candidates.push({ z: s, what: "the lanelet surface" });
  let best = null;
  for (const c of candidates) {
    const dz = Math.abs(c.z - z);
    if (dz <= tol && (best === null || dz < Math.abs(best.z - z))) best = c;
  }
  return best;
}

/** Another node within SNAP_PX of point p on screen, not in `exclude`: { x, y, z, id } or null. */
export function snapToNode(p, exclude) {
  const pr = screenProjector();
  const s = projectToPage(pr, p[0], p[1], p[2]);
  if (!s) return null;
  for (const c of nodesUnderCursor(s[0], s[1], SNAP_PX)) {
    if (exclude.has(c.id)) continue;
    const rec = store.node(c.id);
    if (rec) return { x: rec.x, y: rec.y, z: rec.z, id: c.id };
  }
  return null;
}

export function showSnap(x, y, z) {
  snapMarker.position.set(x, y, z);
  snapMarker.visible = true;
}

export function hideSnap() {
  snapMarker.visible = false;
}
