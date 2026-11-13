// The map view controls: orthographic on/off (V), and the web-map tiles
// under it -- which server, how opaque, and the attribution it requires.
//
// Tiles start off on every page load: turning them on sends the viewed area
// to the tile server, so that is the user's click to make. The chosen
// source and opacity are remembered per browser.
import { TEST_HOOKS } from "./util.js";
import { ortho, setOrtho } from "./camera.js";
import { rigActive, setLocked } from "./sensors/camrig.js";
import { grid } from "./scene.js";
import { toast } from "./hud.js";
import { tileLayer, TILE_PRESETS, checkSource, tileUrl, toUtm, wmsBox } from "./render/tiles.js";

const STORE_KEY = "ll2.viewer3d.tiles";
let els = {};
let warnedFor = null;

function load() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
}

function save() {
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify({
      source: tileLayer.source, opacity: tileLayer.opacity,
    }));
  } catch (_) { /* private window or storage off: nothing to remember then */ }
}

export function installBasemap() {
  els = {
    ortho: document.getElementById("orthoBtn"),
    tiles: document.getElementById("tilesBtn"),
    menu: document.getElementById("tilesMenuBtn"),
    panel: document.getElementById("tilePanel"),
    preset: document.getElementById("tilePreset"),
    url: document.getElementById("tileUrl"),
    wms: document.getElementById("tileWms"),
    layers: document.getElementById("tileLayers"),
    styles: document.getElementById("tileStyles"),
    format: document.getElementById("tileFormat"),
    version: document.getElementById("tileVersion"),
    crs: document.getElementById("tileCrs"),
    transparent: document.getElementById("tileTransparent"),
    maxZoom: document.getElementById("tileMaxZoom"),
    attribution: document.getElementById("tileAttributionText"),
    opacity: document.getElementById("tileOpacity"),
    apply: document.getElementById("tileApply"),
    status: document.getElementById("tileStatus"),
    credit: document.getElementById("tileAttribution"),
    sourceName: document.getElementById("tileSourceName"),
    note: document.getElementById("tilePresetNote"),
  };
  tileLayer.attach();
  tileLayer.onProblem = onTileProblem;
  const saved = load();
  if (saved && saved.source && !checkSource(saved.source).length) tileLayer.setSource(saved.source);
  if (saved && saved.opacity) tileLayer.setOpacity(saved.opacity);
  refreshProxyHosts();

  if (!els.ortho) return;
  els.ortho.addEventListener("click", () => toggleOrtho());
  window.addEventListener("ll2:ortho", () => els.ortho.classList.toggle("on", ortho.on));
  els.tiles.addEventListener("click", () => setTiles(!tileLayer.on));
  els.menu.addEventListener("click", () => {
    els.panel.hidden = !els.panel.hidden;
    els.menu.classList.toggle("on", !els.panel.hidden);
    if (!els.panel.hidden) {
      fillForm(tileLayer.source);
      refreshProxyHosts();
    }
  });
  els.preset.addEventListener("change", () => {
    const preset = TILE_PRESETS[els.preset.value];
    if (preset) fillForm(preset);
    else fillForm({ ...readForm(), type: els.preset.value });
  });
  els.opacity.addEventListener("input", () => {
    tileLayer.setOpacity(Number(els.opacity.value) / 100);
    save();
  });
  els.apply.addEventListener("click", () => applyForm());
  fillForm(tileLayer.source);
  refresh();
}

export function toggleOrtho(on) {
  const next = on === undefined ? !ortho.on : !!on;
  // The lock drives the camera with a recorded pinhole; a map view replaces it.
  if (next && rigActive()) setLocked(false);
  setOrtho(next);
  if (els.ortho) els.ortho.classList.toggle("on", ortho.on);
}

export function setTiles(on) {
  if (on) refreshProxyHosts();
  tileLayer.setOn(on);
  // The reference grid is noise on top of a map.
  grid.visible = !tileLayer.on;
  refresh();
}

/**
 * Ask the viewer server which hosts it fetches tiles from for us. Asked
 * again whenever tiles matter (the form opens, a source is applied, tiles go
 * on): JOSM restarts the server when its allowed hosts change, and this tab
 * may have been open all along.
 */
export function refreshProxyHosts() {
  return fetch("/tileproxy/hosts").then((r) => (r.ok ? r.json() : null))
    .then((j) => {
      if (j && j.hosts) {
        const before = [...(tileLayer.proxyHosts || [])].sort().join(",");
        tileLayer.setProxyHosts(j.hosts);
        // Tiles asked for meanwhile went the other way (direct or proxied): ask again.
        if (before !== [...tileLayer.proxyHosts].sort().join(",") && tileLayer.on) tileLayer.setSource(tileLayer.source);
      }
      showPresetNote();
    })
    .catch(() => {});
}

/** The chosen preset's note, if it has one. */
function showPresetNote() {
  if (!els || !els.note) return;
  const preset = TILE_PRESETS[els.preset.value];
  els.note.textContent = (preset && preset.note) || "";
  els.note.hidden = !els.note.textContent;
}

/** Put a source into the form; presets fill it, custom types keep what was typed. */
function fillForm(source) {
  const preset = Object.keys(TILE_PRESETS).find((k) => TILE_PRESETS[k].url === source.url);
  els.preset.value = preset || source.type || "xyz";
  els.url.value = source.url || "";
  els.layers.value = source.layers || "";
  els.styles.value = source.styles || "";
  els.format.value = source.format || "image/png";
  els.version.value = source.version || "1.3.0";
  els.crs.value = source.crs || "EPSG:3857";
  els.transparent.checked = !!source.transparent;
  els.maxZoom.value = String(source.maxZoom || 19);
  els.attribution.value = source.attribution || "";
  els.opacity.value = String(Math.round(tileLayer.opacity * 100));
  const preset2 = TILE_PRESETS[els.preset.value];
  els.wms.hidden = (preset2 ? preset2.type : els.preset.value) !== "wms";
  for (const el of [els.url, els.layers, els.styles, els.format, els.version, els.crs,
                    els.transparent, els.maxZoom, els.attribution]) el.disabled = !!preset2;
  showPresetNote();
}

function readForm() {
  const preset = TILE_PRESETS[els.preset.value];
  if (preset) return { ...preset };
  return {
    type: els.preset.value,
    url: els.url.value.trim(),
    layers: els.layers.value.trim(),
    styles: els.styles.value.trim(),
    format: els.format.value,
    version: els.version.value,
    crs: els.crs.value,
    transparent: els.transparent.checked,
    maxZoom: Number(els.maxZoom.value) || 19,
    attribution: els.attribution.value.trim(),
  };
}

/** Use what the form says; returns the problems, empty when applied. */
export function applyForm() {
  const source = readForm();
  const problems = checkSource(source);
  if (problems.length) {
    els.status.textContent = problems.join("; ");
    els.status.classList.add("bad");
    return problems;
  }
  els.status.classList.remove("bad");
  warnedFor = null;
  tileLayer.setSource(source);
  save();
  if (!tileLayer.on) setTiles(true);
  else refreshProxyHosts();
  refresh();
  return [];
}

function onTileProblem(failed, url) {
  const key = `${tileLayer.source.url}|${tileLayer.source.layers || ""}`;
  if (warnedFor === key) return;
  warnedFor = key;
  let host = "";
  try { host = new URL(url).host; } catch (_) { /* leave it out */ }
  toast(`Tiles from ${host || "the server"} do not load. Check the URL in the browser; `
    + "the server must also allow cross-origin use (CORS) -- or start the viewer with "
    + `--tile-proxy-host ${host || "<host>"}`, "error", 9000);
}

/** Buttons, attribution and the status line. */
export function refresh() {
  if (!els.tiles) return;
  els.tiles.classList.toggle("on", tileLayer.on);
  els.ortho.classList.toggle("on", ortho.on);
  const s = tileLayer.source;
  els.credit.hidden = !tileLayer.on || !s.attribution;
  els.credit.textContent = "";
  if (s.attribution) {
    if (s.attributionUrl) {
      const a = document.createElement("a");
      a.href = s.attributionUrl;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = s.attribution;
      els.credit.appendChild(a);
    } else {
      els.credit.textContent = s.attribution;
    }
  }
  if (els.sourceName) els.sourceName.textContent = sourceLabel(s);
  const st = tileLayer.stats();
  if (!els.status.classList.contains("bad")) {
    els.status.textContent = tileLayer.on
      ? `zoom ${st.zoom ?? "–"} · ${st.shown}/${st.wanted} tiles${st.failed ? ` · ${st.failed} failed` : ""}`
      : "off";
  }
}

/** A short name for the source: the preset's, else the server's host. */
function sourceLabel(source) {
  const key = Object.keys(TILE_PRESETS).find((k) => TILE_PRESETS[k].url === source.url);
  if (key && els.preset) {
    const option = els.preset.querySelector(`option[value="${key}"]`);
    if (option) return option.textContent;
  }
  try {
    return `${(source.type || "xyz").toUpperCase()} · ${new URL(source.url.replace(/\{[^}]*\}/g, "0")).host}`;
  } catch (_) {
    return source.url || "custom";
  }
}

let lastRefresh = 0;
/** Per frame. */
export function update() {
  tileLayer.update();
  const t = performance.now();
  if (t - lastRefresh > 500) {
    lastRefresh = t;
    refresh();
  }
}

if (TEST_HOOKS) {
  window.__ll2map = {
    ortho: () => ({ on: ortho.on, groundZ: ortho.groundZ }),
    setOrtho: (on) => toggleOrtho(on),
    tiles: () => tileLayer.stats(),
    setTiles: (on) => setTiles(on),
    setSource: (source) => {
      const problems = checkSource(source);
      if (!problems.length) {
        tileLayer.setSource(source);
        refresh();
      }
      return problems;
    },
    tileUrl: (z, x, y) => tileUrl(tileLayer.source, z, x, y),
    utm: (lat, lon, zone) => toUtm(lat, lon, zone),
    wmsBox: (z, x, y) => wmsBox(tileLayer.source, z, x, y),
    // A built tile's texture coordinates, vertex by vertex.
    tileUvs: (key) => {
      const t = tileLayer.tiles.get(key);
      return t && t.mesh ? Array.from(t.mesh.geometry.getAttribute("uv").array) : null;
    },
    gridVisible: () => grid.visible,
    readyKeys: () => [...tileLayer.tiles.values()].filter((t) => t.state === "ready").map((t) => t.key),
    // ENU of a built tile's south-west corner (its first vertex).
    tileCorner: (key) => {
      const t = tileLayer.tiles.get(key);
      if (!t || !t.mesh) return null;
      const p = t.mesh.geometry.getAttribute("position").array;
      return [p[0], p[1]];
    },
  };
}
