// "Street view" button: open the spot being edited in Mapillary, Google Street
// View or Apple Maps. Same URLs as the JOSM toolbar button (StreetImagery.kt).
//
// The spot is the selection's centre (a single node: the node), else the map
// point in the middle of the view, else the camera's ground position. Google
// also gets the camera heading.
import { store } from "./store.js";
import { selection } from "./selection.js";
import { camera, headingDeg } from "./camera.js";
import { canvas } from "./scene.js";
import { pickMapPoint } from "./picking.js";
import { toast } from "./hud.js";

const EARTH_R = 6378137.0;
const deg = (r) => (r * 180) / Math.PI;
const rad = (d) => (d * Math.PI) / 180;

/** Viewer ENU metres -> [lat, lon] (equirectangular about the anchor). */
export function enuToLatLon(x, y, anchor) {
  const cos0 = Math.max(1e-12, Math.abs(Math.cos(rad(anchor.lat))));
  return [anchor.lat + deg(y / EARTH_R), anchor.lon + deg(x / (EARTH_R * cos0))];
}

const f7 = (v) => v.toFixed(7);

export const PROVIDERS = {
  mapillary: {
    label: "Mapillary",
    url: (lat, lon) => `https://www.mapillary.com/app/?lat=${f7(lat)}&lng=${f7(lon)}&z=17&menu=false`,
  },
  google: {
    label: "Google Street View",
    // Maps URLs API: Street View at the panorama nearest the viewpoint.
    url: (lat, lon, heading) => `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${f7(lat)},${f7(lon)}` +
      (heading === null ? "" : `&heading=${Math.round(((heading % 360) + 360) % 360)}`),
  },
  apple: {
    label: "Apple Maps",
    url: (lat, lon) => `https://maps.apple.com/frame?center=${f7(lat)}%2C${f7(lon)}&span=0.000545%2C0.000709`,
  },
};

/** ENU [x, y] of the spot to look at, and what it is. */
export function imagerySpot() {
  if (!selection.isEmpty()) {
    let n = 0, x = 0, y = 0;
    for (const id of selection.effectiveNodeIds()) {
      const rec = store.node(id);
      if (!rec) continue;
      x += rec.x;
      y += rec.y;
      n++;
    }
    if (n) return { x: x / n, y: y / n, what: n === 1 ? "the selected node" : "the selection" };
  }
  const r = canvas.getBoundingClientRect();
  const hit = pickMapPoint(r.left + r.width / 2, r.top + r.height / 2);
  if (hit) return { x: hit.point.x, y: hit.point.y, what: "the middle of the view" };
  return { x: camera.position.x, y: camera.position.y, what: "the camera position" };
}

export function openImagery(key) {
  const p = PROVIDERS[key];
  if (!p) return null;
  if (!store.anchor) {
    toast("Street view: no map anchor yet (is JOSM connected?)", "warn");
    return null;
  }
  const spot = imagerySpot();
  const [lat, lon] = enuToLatLon(spot.x, spot.y, store.anchor);
  const url = p.url(lat, lon, headingDeg());
  window.open(url, "_blank", "noopener");
  toast(`${p.label} at ${spot.what}`, "info", 2000);
  return url;
}

/** The toolbar button and its provider menu. */
export function installImagery() {
  const btn = document.getElementById("imageryBtn");
  const menu = document.getElementById("imageryMenu");
  if (!btn || !menu) return;
  for (const [key, p] of Object.entries(PROVIDERS)) {
    const item = document.createElement("button");
    item.type = "button";
    item.textContent = p.label;
    item.setAttribute("data-provider", key);
    item.addEventListener("click", () => {
      menu.hidden = true;
      openImagery(key);
    });
    menu.appendChild(item);
  }
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
  });
  document.addEventListener("pointerdown", (e) => {
    if (!menu.hidden && !menu.contains(e.target) && e.target !== btn) menu.hidden = true;
  });
}
