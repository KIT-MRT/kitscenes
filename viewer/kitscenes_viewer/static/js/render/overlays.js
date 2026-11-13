// Things drawn over the map: the JOSM viewport rectangle and markers that keep
// a constant pixel size at any distance (orbit pivot, zoom target).
import * as THREE from "three";
import { store, VIEWPORT_ID } from "../store.js";
import { overlayRoot } from "../scene.js";
import { camera, pixelsToMetres } from "../camera.js";
import { VIEWPORT_COLOR, PIVOT_COLOR, ZOOM_COLOR } from "./style.js";

// --- JOSM viewport rectangle ------------------------------------------------------
// A gray closed loop on the floor plane, drawn on top of other geometry so it
// stays visible as an overlay.
const viewportMat = new THREE.LineBasicMaterial({
  color: VIEWPORT_COLOR, transparent: true, opacity: 0.85, depthTest: false,
});
let viewportLine = null;

function setViewport(f) {
  if (viewportLine) {
    overlayRoot.remove(viewportLine);
    viewportLine.geometry.dispose();
    viewportLine = null;
  }
  if (!f) return;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(f.pos), 3));
  viewportLine = new THREE.Line(geom, viewportMat);
  viewportLine.renderOrder = 999;
  overlayRoot.add(viewportLine);
}

store.on("upsert", (f) => { if (f.id === VIEWPORT_ID) setViewport(f); });
store.on("remove", (f) => { if (f.id === VIEWPORT_ID) setViewport(null); });
store.on("clear", () => setViewport(null));

// --- screen-sized markers ------------------------------------------------------------
const sized = [];

/** A unit sphere rescaled every frame to `radiusPx` on screen. */
export function screenSphere(color, radiusPx, renderOrder, opacity = 0.9) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity }),
  );
  mesh.visible = false;
  mesh.renderOrder = renderOrder;
  mesh.userData.radiusPx = radiusPx;
  overlayRoot.add(mesh);
  sized.push(mesh);
  return mesh;
}

/** Shows what an orbit turns around, for the duration of the drag. */
export const pivotMarker = screenSphere(PIVOT_COLOR, 5, 1002, 0.85);

/** Where the wheel zooms to (the point under the cursor), shown briefly. */
export const zoomMarker = screenSphere(ZOOM_COLOR, 4, 1002, 0.9);

export function updateScreenSizedMarkers() {
  for (const m of sized) {
    if (!m.visible) continue;
    const d = Math.max(camera.near, camera.position.distanceTo(m.position));
    m.scale.setScalar(pixelsToMetres(m.userData.radiusPx, d));
  }
}
