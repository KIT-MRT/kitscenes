// Web map tiles under the map: XYZ, TMS or WMS, drawn inside the three.js
// scene rather than by a separate map library.
//
// Leaflet and friends draw tiles as DOM images in Web Mercator; stacked under
// this canvas they would have to follow every camera move, could not tilt
// or turn with it, and would drift against the viewer's frame, which is a
// local equirectangular ENU about the anchor, not Mercator -- metres off
// across a city. Here each tile is a small textured grid whose vertices are
// put through the same lat/lon -> ENU formula as the map itself, so tiles
// line up with the map wherever they are, in the orthographic map view and
// in perspective alike. The grid matters: Mercator -> ENU is not linear in y.
//
// Tiles draw first, without depth, so everything else lies on top of them --
// and *opaque*: three draws every opaque object before any transparent one,
// whatever their renderOrder, so a translucent tile would land on top of the
// map. Opacity is therefore a blend toward the background colour in the
// shader, and a transparent WMS layer's alpha blends the same way.
import * as THREE from "three";
import { camera, camRefs, ortho, pixelsToMetres } from "../camera.js";
import { renderer, scene } from "../scene.js";
import { store } from "../store.js";

const EARTH_R = 6378137.0;           // the viewer's ENU radius, and Web Mercator's sphere
const MERC_MAX = Math.PI * EARTH_R;  // 20037508.34 m
const GRID = 4;                      // quads per tile edge
const MAX_TILES = 96;                // per view; coarser zoom beyond that
const MAX_RANGE_M = 3000;            // perspective: how far out the ground is tiled
const MAX_IN_FLIGHT = 8;
const CACHE_TILES = 384;
const ANCESTOR_LEVELS = 4;           // coarser stand-ins while a tile loads

const TILE_VERTEX = /* glsl */`
varying vec2 vUv;
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  #include <logdepthbuf_vertex>
}
`;

const TILE_FRAGMENT = /* glsl */`
uniform sampler2D uMap;
uniform float uOpacity;
uniform vec3 uBackground;
varying vec2 vUv;
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  #include <logdepthbuf_fragment>
  vec4 t = texture2D(uMap, vUv);
  gl_FragColor = vec4(mix(uBackground, t.rgb, t.a * uOpacity), 1.0);
  #include <colorspace_fragment>
}
`;

export const TILE_PRESETS = {
  osm: {
    label: "OpenStreetMap",
    type: "xyz",
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
    attributionUrl: "https://www.openstreetmap.org/copyright",
  },
  // 20 cm aerial images of Hesse (where the Frankfurt scenes are). Serves
  // EPSG:3857 and allows cross-origin use, so it needs no proxy.
  hessen_dop20: {
    label: "Hessen DOP20 (aerial)",
    type: "wms",
    url: "https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows",
    layers: "he_dop20_rgb",
    format: "image/jpeg",
    version: "1.3.0",
    maxZoom: 20,
    attribution: "© HVBG, dl-de/by-2-0",
    attributionUrl: "https://www.govdata.de/dl-de/by-2-0",
  },
  // 20 cm aerial images of Baden-Wuerttemberg (Karlsruhe, Sindelfingen), in
  // EPSG:25832 as the LGL serves them. If tiles do not appear, start the
  // viewer with --tile-proxy-host owsproxy.lgl-bw.de.
  bw_dop20: {
    label: "Baden-Württemberg DOP20 (aerial)",
    type: "wms",
    url: "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C",
    layers: "IMAGES_DOP_20_RGB",
    format: "image/png",
    version: "1.1.1",
    crs: "EPSG:25832",
    maxZoom: 20,
    attribution: "© LGL-BW, www.lgl-bw.de",
    attributionUrl: "https://www.lgl-bw.de",
  },
};

const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;

/** Viewer ENU metres of a lat/lon (equirectangular about the anchor). */
export function latLonToEnu(lat, lon, anchor) {
  return [rad(lon - anchor.lon) * Math.cos(rad(anchor.lat)) * EARTH_R,
          rad(lat - anchor.lat) * EARTH_R];
}

export function enuToLatLon(x, y, anchor) {
  const cos0 = Math.max(1e-12, Math.abs(Math.cos(rad(anchor.lat))));
  return [anchor.lat + deg(y / EARTH_R), anchor.lon + deg(x / (EARTH_R * cos0))];
}

/**
 * UTM zone of an EPSG code: ETRS89 / UTM (258NN, what German state servers
 * serve, e.g. 25832) or WGS 84 / UTM north (326NN). null for anything else.
 * ETRS89 is taken as WGS 84 -- the same null transformation pyproj applies.
 */
export function utmZoneOf(crs) {
  const m = /^EPSG:(258(2[89]|3[0-8])|326(0[1-9]|[1-5][0-9]|60))$/.exec(String(crs || "").toUpperCase());
  return m ? Number(m[1].slice(3)) : null;
}

// Transverse Mercator by Krueger's series to n^4 (GRS80; WGS 84 differs by
// 0.1 mm here): sub-millimetre over a whole zone and well beyond it.
const UTM_A = 6378137.0;
const UTM_F = 1 / 298.257222101;
const UTM_N = UTM_F / (2 - UTM_F);
const UTM_AA = (UTM_A / (1 + UTM_N)) * (1 + UTM_N ** 2 / 4 + UTM_N ** 4 / 64);
const UTM_ALPHA = [
  UTM_N / 2 - (2 * UTM_N ** 2) / 3 + (5 * UTM_N ** 3) / 16 + (41 * UTM_N ** 4) / 180,
  (13 * UTM_N ** 2) / 48 - (3 * UTM_N ** 3) / 5 + (557 * UTM_N ** 4) / 1440,
  (61 * UTM_N ** 3) / 240 - (103 * UTM_N ** 4) / 140,
  (49561 * UTM_N ** 4) / 161280,
];
const UTM_E2N = (2 * Math.sqrt(UTM_N)) / (1 + UTM_N);

/** [easting, northing] of a lat/lon in UTM zone `zone` (north). */
export function toUtm(lat, lon, zone) {
  const phi = rad(lat);
  const dl = rad(lon - (zone * 6 - 183));
  const sinPhi = Math.sin(phi);
  const t = Math.sinh(Math.atanh(sinPhi) - UTM_E2N * Math.atanh(UTM_E2N * sinPhi));
  const xi = Math.atan2(t, Math.cos(dl));
  const eta = Math.atanh(Math.sin(dl) / Math.sqrt(1 + t * t));
  let e = eta;
  let n = xi;
  for (let j = 1; j <= 4; j++) {
    e += UTM_ALPHA[j - 1] * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta);
    n += UTM_ALPHA[j - 1] * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta);
  }
  return [500000 + 0.9996 * UTM_AA * e, 0.9996 * UTM_AA * n];
}

const mercX = (lon) => rad(lon) * EARTH_R;
const mercY = (lat) => EARTH_R * Math.log(Math.tan(Math.PI / 4 + rad(Math.max(-85.0511, Math.min(85.0511, lat))) / 2));
const lonOfMercX = (x) => deg(x / EARTH_R);
const latOfMercY = (y) => deg(2 * Math.atan(Math.exp(y / EARTH_R)) - Math.PI / 2);
const tileSize = (z) => (2 * MERC_MAX) / 2 ** z;

/** Mercator bounds of tile (z, x, y), y counted from the top as XYZ does. */
export function tileBounds(z, x, y) {
  const s = tileSize(z);
  return { minx: -MERC_MAX + x * s, maxx: -MERC_MAX + (x + 1) * s,
           maxy: MERC_MAX - y * s, miny: MERC_MAX - (y + 1) * s };
}

/** A tile's sample points: its corners and edge midpoints, as lat/lon. */
function tileOutline(z, x, y) {
  const b = tileBounds(z, x, y);
  const out = [];
  for (const u of [0, 0.5, 1]) {
    for (const v of [0, 0.5, 1]) {
      out.push([latOfMercY(b.miny + v * (b.maxy - b.miny)), lonOfMercX(b.minx + u * (b.maxx - b.minx))]);
    }
  }
  return out;
}

/**
 * What a WMS is asked for, for tile (z, x, y): the CRS and the box in it. In
 * EPSG:3857 that is the tile itself. In UTM the tile is turned by the grid
 * convergence (~0.3 degrees in Frankfurt), so the box encloses it, and each
 * vertex of the tile mesh then finds its own place in the image.
 */
export function wmsBox(source, z, x, y) {
  const zone = utmZoneOf(source.crs);
  if (!zone) {
    const b = tileBounds(z, x, y);
    return { crs: "EPSG:3857", zone: null, ...b };
  }
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [lat, lon] of tileOutline(z, x, y)) {
    const [e, n] = toUtm(lat, lon, zone);
    minx = Math.min(minx, e); maxx = Math.max(maxx, e);
    miny = Math.min(miny, n); maxy = Math.max(maxy, n);
  }
  // Rounded outward to the millimetre the URL carries, so the image the
  // server draws and the texture coordinates describe exactly the same box,
  // and it still encloses the tile.
  const down = (v) => Math.floor(v * 1000) / 1000;
  const up = (v) => Math.ceil(v * 1000) / 1000;
  return { crs: String(source.crs).toUpperCase(), zone,
           minx: down(minx), miny: down(miny), maxx: up(maxx), maxy: up(maxy) };
}

/** The URL of one tile for a source, or null when the source cannot serve it. */
export function tileUrl(source, z, x, y) {
  if (source.type === "wms") {
    const b = wmsBox(source, z, x, y);
    const v = source.version || "1.3.0";
    const params = [
      "SERVICE=WMS", "REQUEST=GetMap", `VERSION=${encodeURIComponent(v)}`,
      `LAYERS=${encodeURIComponent(source.layers || "")}`,
      `STYLES=${encodeURIComponent(source.styles || "")}`,
      `FORMAT=${encodeURIComponent(source.format || "image/png")}`,
      `TRANSPARENT=${source.transparent ? "TRUE" : "FALSE"}`,
      `${v === "1.3.0" ? "CRS" : "SRS"}=${b.crs}`,
      "WIDTH=256", "HEIGHT=256",
      // EPSG:3857 and the UTM codes are easting/northing in both versions;
      // only geographic CRSs like 4326 swap the axes in 1.3.0.
      `BBOX=${b.minx.toFixed(3)},${b.miny.toFixed(3)},${b.maxx.toFixed(3)},${b.maxy.toFixed(3)}`,
    ];
    const base = source.url || "";
    return base + (base.includes("?") ? (/[?&]$/.test(base) ? "" : "&") : "?") + params.join("&");
  }
  const n = 2 ** z;
  const yy = source.type === "tms" ? n - 1 - y : y;
  const subs = source.subdomains || "abc";
  return (source.url || "")
    .replace("{z}", String(z))
    .replace("{x}", String(x))
    .replace("{-y}", String(n - 1 - y))
    .replace("{y}", String(yy))
    .replace("{s}", subs[(x + y) % subs.length]);
}

/** Problems with a source a user typed, as a list of strings (empty = fine). */
export function checkSource(source) {
  const problems = [];
  if (!/^https?:\/\//i.test(source.url || "")) problems.push("the URL must start with http:// or https://");
  if (source.type === "wms") {
    if (!source.layers) problems.push("a WMS needs at least one layer name");
    if (source.crs && source.crs !== "EPSG:3857" && !utmZoneOf(source.crs)) {
      problems.push("the CRS must be EPSG:3857 or a UTM zone (EPSG:258NN / 326NN)");
    }
  } else if (!/\{z\}/.test(source.url || "") || !/\{x\}/.test(source.url || "")
             || !/\{-?y\}/.test(source.url || "")) {
    problems.push("an XYZ/TMS URL needs {z}, {x} and {y} (or {-y})");
  }
  return problems;
}

class TileLayer {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = "tiles";
    this.group.renderOrder = -1000;
    this.on = false;
    this.source = { ...TILE_PRESETS.osm };
    this.opacity = 0.85;
    this.proxyHosts = new Set();
    this.tiles = new Map();     // "z/x/y" -> { key, z, x, y, state, mesh, texture, used }
    this.inFlight = 0;
    this.queue = [];
    this.zoom = null;
    this.wanted = 0;
    this.failed = 0;
    this.loadedCount = 0;
    this.lastKey = "";
    this.generation = 0;
    this.onProblem = null;      // (failed, url) => void, for the HUD
    // Without a map there is no store anchor; the city's anchor stands in.
    this.fallbackAnchor = null;
    this.loader = new THREE.ImageLoader();
    this.loader.setCrossOrigin("anonymous");
  }

  attach() {
    scene.add(this.group);
  }

  setOn(on) {
    this.on = !!on;
    this.group.visible = this.on;
    this.lastKey = "";
  }

  setOpacity(value) {
    this.opacity = Math.max(0.05, Math.min(1, value));
    for (const t of this.tiles.values()) if (t.mesh) t.mesh.material.uniforms.uOpacity.value = this.opacity;
  }

  /** Switch source: everything cached from the old one goes. */
  setSource(source) {
    this.source = { ...source };
    this.generation++;
    for (const t of this.tiles.values()) this._dispose(t);
    this.tiles.clear();
    this.queue.length = 0;
    this.failed = 0;
    this.loadedCount = 0;
    this.lastKey = "";
  }

  setProxyHosts(hosts) {
    this.proxyHosts = new Set((hosts || []).map((h) => h.toLowerCase()));
  }

  /** Through the viewer server when the user allowed that host there (no CORS needed). */
  _fetchUrl(url) {
    try {
      const u = new URL(url);
      const host = u.hostname.toLowerCase();
      if (this.proxyHosts.has(host) || this.proxyHosts.has(`${host}:${u.port}`)) {
        return `/tileproxy?url=${encodeURIComponent(url)}`;
      }
    } catch (_) { /* not a URL; let the loader fail on it */ }
    return url;
  }

  /** Ground plane the view is tiled on. */
  _groundZ() {
    return ortho.on ? ortho.groundZ : camRefs.groundZ;
  }

  /**
   * The patch of ground in view, as ENU points: in ortho the screen
   * rectangle; in perspective the screen's rays on the ground plane, cut off
   * at MAX_RANGE_M so a view toward the horizon does not ask for a continent.
   */
  _footprint() {
    const g = this._groundZ();
    const pts = [];
    const ndc = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const origin = camera.position;
    for (const [nx, ny] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]]) {
      if (ortho.on) {
        ndc.set(nx, ny, 0).unproject(camera);
        pts.push([ndc.x, ndc.y]);
        continue;
      }
      ndc.set(nx, ny, 0.5).unproject(camera);
      dir.subVectors(ndc, origin).normalize();
      let t = dir.z < -1e-6 ? (g - origin.z) / dir.z : Infinity;
      const horiz = Math.hypot(dir.x, dir.y) || 1e-9;
      t = Math.min(t, MAX_RANGE_M / horiz);
      if (!(t > 0)) t = MAX_RANGE_M / horiz;
      pts.push([origin.x + dir.x * t, origin.y + dir.y * t]);
    }
    return pts;
  }

  /** Per frame: decide which tiles the view needs and show what is there. */
  anchor() {
    return store.anchor || this.fallbackAnchor;
  }

  update() {
    const anchor = this.anchor();
    if (!this.on || !anchor) return;
    // Tiles bake ENU vertices about the anchor they were built with; a new
    // map (or JOSM's anchor replacing the city's) moves the frame under them.
    const anchorKey = `${anchor.lat},${anchor.lon}`;
    if (this.builtFor !== anchorKey) {
      if (this.builtFor !== undefined) this.setSource(this.source);
      this.builtFor = anchorKey;
    }
    camera.updateMatrixWorld();
    const key = `${camera.position.toArray().map((v) => v.toFixed(2))}|${camera.quaternion.toArray()
      .map((v) => v.toFixed(4))}|${camera.top}|${camera.fov}|${this.opacity}|${renderer.domElement.clientWidth}x${renderer.domElement.clientHeight}`;
    this.group.position.z = this._groundZ() - 0.05;
    if (key === this.lastKey) {
      this._pump();
      return;
    }
    this.lastKey = key;

    const foot = this._footprint();
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const [x, y] of foot) {
      const [lat, lon] = enuToLatLon(x, y, anchor);
      const mx = mercX(lon);
      const my = mercY(lat);
      minx = Math.min(minx, mx); maxx = Math.max(maxx, mx);
      miny = Math.min(miny, my); maxy = Math.max(maxy, my);
    }
    // Zoom: one tile pixel per screen pixel at the middle of the view.
    const centre = foot[4];
    const [clat] = enuToLatLon(centre[0], centre[1], anchor);
    const depth = Math.max(1, camera.position.distanceTo(new THREE.Vector3(centre[0], centre[1], this._groundZ())));
    const mpp = pixelsToMetres(1, depth);
    const maxZoom = Math.max(0, Math.min(22, Number(this.source.maxZoom) || 19));
    let z = Math.round(Math.log2((2 * MERC_MAX * Math.cos(rad(clat))) / (256 * Math.max(1e-3, mpp))));
    z = Math.max(0, Math.min(maxZoom, z));
    let range = this._range(z, minx, miny, maxx, maxy);
    while (z > 0 && range.count > MAX_TILES) {
      z--;
      range = this._range(z, minx, miny, maxx, maxy);
    }
    this.zoom = z;

    const want = [];
    for (let ty = range.y0; ty <= range.y1; ty++) {
      for (let tx = range.x0; tx <= range.x1; tx++) want.push([z, tx, ty]);
    }
    this.wanted = want.length;
    const cx = (range.x0 + range.x1) / 2;
    const cy = (range.y0 + range.y1) / 2;
    want.sort((a, b) => Math.hypot(a[1] - cx, a[2] - cy) - Math.hypot(b[1] - cx, b[2] - cy));

    const now = performance.now();
    const shown = new Set();
    this.queue = [];
    for (const [tz, tx, ty] of want) {
      const t = this._tile(tz, tx, ty);
      t.used = now;
      if (t.state === "ready") {
        shown.add(t);
        continue;
      }
      if (t.state === "new") this.queue.push(t);
      // Until it arrives, the nearest coarser tile that is there stands in.
      for (let up = 1; up <= ANCESTOR_LEVELS && tz - up >= 0; up++) {
        const a = this.tiles.get(`${tz - up}/${tx >> up}/${ty >> up}`);
        if (a && a.state === "ready") {
          a.used = now;
          shown.add(a);
          break;
        }
      }
    }
    for (const t of this.tiles.values()) if (t.mesh) t.mesh.visible = shown.has(t);
    this._trim();
    this._pump();
  }

  _range(z, minx, miny, maxx, maxy) {
    const s = tileSize(z);
    const n = 2 ** z;
    const clamp = (v) => Math.max(0, Math.min(n - 1, v));
    const x0 = clamp(Math.floor((minx + MERC_MAX) / s));
    const x1 = clamp(Math.floor((maxx + MERC_MAX) / s));
    const y0 = clamp(Math.floor((MERC_MAX - maxy) / s));
    const y1 = clamp(Math.floor((MERC_MAX - miny) / s));
    return { x0, x1, y0, y1, count: (x1 - x0 + 1) * (y1 - y0 + 1) };
  }

  _tile(z, x, y) {
    const key = `${z}/${x}/${y}`;
    let t = this.tiles.get(key);
    if (!t) {
      t = { key, z, x, y, state: "new", mesh: null, texture: null, used: 0 };
      this.tiles.set(key, t);
    }
    return t;
  }

  _pump() {
    while (this.inFlight < MAX_IN_FLIGHT && this.queue.length) {
      const t = this.queue.shift();
      if (t.state !== "new") continue;
      const url = tileUrl(this.source, t.z, t.x, t.y);
      if (!url) {
        t.state = "failed";
        continue;
      }
      t.state = "loading";
      this.inFlight++;
      const gen = this.generation;
      this.loader.load(this._fetchUrl(url), (image) => {
        this.inFlight--;
        if (gen !== this.generation || !this.tiles.has(t.key)) return;
        this._build(t, image);
        this.loadedCount++;
        this.lastKey = "";  // show it on the next frame
      }, undefined, () => {
        this.inFlight--;
        if (gen !== this.generation) return;
        t.state = "failed";
        this.failed++;
        if (this.onProblem && (this.failed === 1 || this.failed % 25 === 0)) {
          this.onProblem(this.failed, url);
        }
      });
    }
  }

  /**
   * A GRID x GRID patch with each vertex put through Mercator -> lat/lon ->
   * ENU. Its texture coordinate is where that lat/lon falls in the image:
   * the tile itself for Mercator sources, the requested UTM box otherwise.
   */
  _build(t, image) {
    const anchor = this.anchor();
    const b = tileBounds(t.z, t.x, t.y);
    const box = this.source.type === "wms" ? wmsBox(this.source, t.z, t.x, t.y) : null;
    const positions = [];
    const uvs = [];
    for (let j = 0; j <= GRID; j++) {
      for (let i = 0; i <= GRID; i++) {
        const u = i / GRID;
        const v = j / GRID;  // 0 at the bottom (south), as the texture is flipped
        const lat = latOfMercY(b.miny + v * (b.maxy - b.miny));
        const lon = lonOfMercX(b.minx + u * (b.maxx - b.minx));
        const [x, y] = latLonToEnu(lat, lon, anchor);
        positions.push(x, y, 0);
        if (box && box.zone) {
          const [e, n] = toUtm(lat, lon, box.zone);
          uvs.push((e - box.minx) / (box.maxx - box.minx), (n - box.miny) / (box.maxy - box.miny));
        } else {
          uvs.push(u, v);
        }
      }
    }
    const index = [];
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        const a = j * (GRID + 1) + i;
        index.push(a, a + 1, a + GRID + 2, a, a + GRID + 2, a + GRID + 1);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(index);
    const texture = new THREE.Texture(image);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    texture.needsUpdate = true;
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: texture },
        uOpacity: { value: this.opacity },
        // Linear, as three keeps colours; the shader converts on output.
        uBackground: { value: (scene.background && scene.background.isColor)
          ? scene.background.clone() : new THREE.Color(0x11151c) },
      },
      vertexShader: TILE_VERTEX,
      fragmentShader: TILE_FRAGMENT,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    // Coarse first, so a finer tile always lies over its stand-in.
    mesh.renderOrder = -1000 + t.z;
    mesh.frustumCulled = false;
    mesh.visible = false;
    t.mesh = mesh;
    t.texture = texture;
    t.state = "ready";
    this.group.add(mesh);
  }

  _dispose(t) {
    if (t.mesh) {
      this.group.remove(t.mesh);
      t.mesh.geometry.dispose();
      t.mesh.material.dispose();
    }
    if (t.texture) t.texture.dispose();
    t.mesh = null;
    t.texture = null;
  }

  _trim() {
    if (this.tiles.size <= CACHE_TILES) return;
    const old = [...this.tiles.values()].filter((t) => t.state !== "loading")
      .sort((a, b) => a.used - b.used);
    for (const t of old.slice(0, this.tiles.size - CACHE_TILES)) {
      this._dispose(t);
      this.tiles.delete(t.key);
    }
  }

  stats() {
    let ready = 0;
    let shown = 0;
    for (const t of this.tiles.values()) {
      if (t.state === "ready") ready++;
      if (t.mesh && t.mesh.visible) shown++;
    }
    return { on: this.on, zoom: this.zoom, wanted: this.wanted, ready, shown, failed: this.failed,
             inFlight: this.inFlight, source: this.source, opacity: this.opacity };
  }
}

export const tileLayer = new TileLayer();
