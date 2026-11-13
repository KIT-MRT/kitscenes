// Every recorded scene of one city, on the city map.
//
// The scene server in city mode announces itself with one `city` message;
// the trajectories come from its `/city.json`, and each scene's data from
// `/scene/<id>/...` -- the same routes a single scene has, so picking a
// trajectory turns it into "the" scene and every scene tool (frames, camera
// lock, overlay, whole-scene cloud) works unchanged.
//
// Point data around the camera is the static clouds of the nearby scenes,
// merged by the server into 128 m tiles (`/city/tiles.json`, `/city/tile/`):
// five drives down one street are one cloud, not five stacked copies. A tile
// is fetched again when one more of its scenes has been built (its `version`
// names them). Servers without merged tiles (no cache) send each scene's own
// chunks instead. Tiles are loaded within a radius and dropped again beyond
// 1.3x of it. By default that only happens on `Load here`: flying across a
// city costs nothing. `Auto` follows the camera but only ever loads clouds
// that are already built -- a cold build takes a while, and only an
// explicit click should start one.
import { TEST_HOOKS } from "../util.js";
import { renderer, canvas, mapRoot } from "../scene.js";
import * as THREE from "three";
import { camera, frameBox, setCameraView, lookAtPoint, view } from "../camera.js";
import { store } from "../store.js";
import { toast } from "../hud.js";
import { setJob, setTabAvailable, currentTab, panelOpen } from "../ui/shell.js";
import { edit, addViewClickFilter } from "../edit.js";
import { rigActive } from "./camrig.js";
import { trajectoryLayer } from "../render/trajectories.js";
import { pointLayer } from "../render/points.js";
import { tileLayer } from "../render/tiles.js";
import { onSceneMessage, clearScene, sceneState, showFrame, setCityActive, setClassTable,
         refreshPointUi, goToCar } from "./scene.js";

export const cityState = {
  info: null,       // the `city` message's payload
  name: null,
  scenes: [],       // /city.json's scenes
  loading: false,
  error: null,
  radius: 300,
  auto: false,
  opening: null,    // the scene being opened
};

const EVICT_FACTOR = 1.3;   // dropped beyond this many radii, so an edge does not flicker
const BUILD_POLL_MS = 2000;
const TICK_MS = 400;
const CLICK_SLOP_PX = 4;

// Per scene: { state, progress, chunks, error, timer }. state is one of
// idle, unbuilt (Auto skipped it), meta, queued, building, ready, error.
const sceneLoad = new Map();
const loaded = new Map();    // chunk key -> { id, centre, radius, version, ids }
const queue = [];            // chunk jobs, nearest first
let busy = false;
let generation = 0;          // bumped by Unload, so jobs in flight are dropped
let cityGen = 0;             // bumped by a new or reset city: its answers are stale
let lastFocus = null;
let lastTick = 0;
let status = {};             // /city/status.json
let mergedTiles = true;      // the server merges scenes into tiles; false: per-scene chunks
let els = {};

const byId = (id) => cityState.scenes.find((s) => s.id === id) || null;
const shortId = (id) => (id && id.length > 12 ? `${id.slice(0, 8)}…` : id || "");
const loadOf = (id) => {
  if (!sceneLoad.has(id)) sceneLoad.set(id, { state: "idle" });
  return sceneLoad.get(id);
};
/** Where loading is centred: under the camera. */
const focus = () => [camera.position.x, camera.position.y];

export function installCity() {
  els = {
    bar: document.getElementById("cityBar"),
    name: document.getElementById("cityName"),
    load: document.getElementById("cityLoad"),
    auto: document.getElementById("cityAuto"),
    radius: document.getElementById("cityRadius"),
    radiusLabel: document.getElementById("cityRadiusLabel"),
    unload: document.getElementById("cityUnload"),
    tracks: document.getElementById("cityTracks"),
    deselect: document.getElementById("cityDeselect"),
    status: document.getElementById("cityStatus"),
    hover: document.getElementById("cityHover"),
    list: document.getElementById("sceneList"),
    listCount: document.getElementById("sceneListCount"),
    search: document.getElementById("sceneSearch"),
  };
  if (!els.bar) return;
  if (els.search) els.search.addEventListener("input", () => renderSceneList(true));
  if (els.list) installSceneList();
  trajectoryLayer.attach(mapRoot);
  els.load.addEventListener("click", () => loadHere(true));
  els.auto.addEventListener("click", () => setAuto(!cityState.auto));
  els.radius.addEventListener("input", () => setRadius(Number(els.radius.value)));
  els.unload.addEventListener("click", () => unloadAll());
  els.tracks.addEventListener("click", () => setTracks(!trajectoryLayer.visible));
  els.deselect.addEventListener("click", () => clearScene());
  window.addEventListener("ll2:scene", onSceneChanged);
  installPointer();
  // A click on a trajectory opens its scene; it does not also select the map.
  addViewClickFilter((x, y) => pickable() && !!pickAt(x, y));
}

/** A `city` message: fetch the trajectories and draw them. */
export function onCityMessage(msg) {
  const info = msg.city;
  if (!info || !info.base || !els.bar) return Promise.resolve(false);
  if (cityState.info && cityState.info.base === info.base && cityState.scenes.length) {
    return Promise.resolve(true);  // a reloaded tab's bootstrap; nothing new
  }
  if (cityState.info) resetCity();
  cityState.info = info;
  cityState.name = info.name || null;
  cityState.loading = true;
  const gen = ++cityGen;
  els.bar.hidden = false;
  setCityActive(true);
  setTabAvailable("scene", true, { focus: true });
  refreshBar();
  return fetch(`${info.base}/city.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((city) => {
      if (gen !== cityGen) return false;  // a newer city arrived meanwhile
      cityState.scenes = city.scenes || [];
      cityState.name = city.city || cityState.name;
      cityState.error = null;
      if (city.classes) setClassTable(city.classes);
      if (city.anchor) tileLayer.fallbackAnchor = city.anchor;
      trajectoryLayer.setScenes(cityState.scenes);
      // With a map there is already something framed; without one the
      // trajectories are all there is to look at.
      if (!store.featureCount() && cityState.scenes.length) frameBox(trajectoryLayer.box());
      return true;
    })
    .catch((err) => {
      if (gen !== cityGen) return false;
      cityState.error = String(err.message || err);
      toast(`City: ${cityState.error}`, "error");
      return false;
    })
    .finally(() => {
      if (gen === cityGen) cityState.loading = false;
      refreshBar();
    });
}

function resetCity() {
  cityGen++;
  mergedTiles = true;  // a new server may merge where the last one did not
  setJob("city", null);
  cityState.opening = null;
  unloadAll();
  clearScene();
  trajectoryLayer.clear();
  for (const st of sceneLoad.values()) clearTimeout(st.timer);
  sceneLoad.clear();
  cityState.scenes = [];
}

// ---- picking a scene --------------------------------------------------------
const pickable = () => cityState.scenes.length > 0 && !edit.on && !rigActive()
  && trajectoryLayer.visible;

function pickAt(x, y) {
  return trajectoryLayer.pick(x, y, camera, canvas.getBoundingClientRect());
}

function installPointer() {
  let down = null;
  canvas.addEventListener("pointerdown", (e) => {
    down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
  });
  canvas.addEventListener("pointerup", (e) => {
    const d = down;
    down = null;
    if (!d || e.button !== 0 || !pickable()) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_SLOP_PX) return;  // an orbit
    const hit = pickAt(e.clientX, e.clientY);
    if (hit) openScene(hit.id, hit.frame);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons) hideHover();
    else scheduleHover(e.clientX, e.clientY);
  });
  canvas.addEventListener("pointerleave", hideHover);
}

let hoverAt = null;
let hoverFrame = 0;
let cursorSet = false;
let hoverCamKey = "";

function scheduleHover(x, y) {
  hoverAt = [x, y];
  if (hoverFrame) return;
  hoverFrame = requestAnimationFrame(() => {
    hoverFrame = 0;
    if (!hoverAt || !pickable()) {
      hideHover();
      return;
    }
    const hit = pickAt(hoverAt[0], hoverAt[1]);
    if (!hit) {
      hideHover();
      return;
    }
    const s = cityState.scenes[hit.scene];
    trajectoryLayer.setHovered(hit.scene);
    const current = sceneState.info && sceneState.info.id === s.id;
    els.hover.textContent = `${shortId(s.id)} · ${s.split} · frame ${hit.frame} / ${s.frames - 1}`
      + (current ? "" : " · click to open");
    els.hover.style.left = `${hoverAt[0] + 14}px`;
    els.hover.style.top = `${hoverAt[1] + 12}px`;
    els.hover.hidden = false;
    canvas.style.cursor = "pointer";
    cursorSet = true;
    hoverCamKey = camKey();
  });
}

const camKey = () => `${camera.position.toArray().join()},${camera.quaternion.toArray().join()}`;

function hideHover() {
  hoverAt = null;
  if (trajectoryLayer.hovered !== -1) trajectoryLayer.setHovered(-1);
  if (els.hover) els.hover.hidden = true;
  if (cursorSet) {
    canvas.style.cursor = "";
    cursorSet = false;
  }
}

/**
 * Make `id` the scene, at `frame`. The camera stays where it is: the frame
 * that was clicked is in view already, and K goes to the car if wanted.
 */
export function openScene(id, frame = 0) {
  const s = byId(id);
  if (!s) return Promise.resolve(false);
  if (sceneState.info && sceneState.info.id === id) {
    showFrame(frame);
    return Promise.resolve(true);
  }
  cityState.opening = id;
  const gen = cityGen;
  refreshBar();
  return fetch(`${s.base}/scene.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((info) => {
      // Another click won, Deselect came first, or the city changed.
      if (cityState.opening !== id || gen !== cityGen) return false;
      onSceneMessage({ scene: info }, frame);
      return true;
    })
    .catch((err) => {
      toast(`Scene ${shortId(id)}: ${err.message || err}`, "error");
      return false;
    })
    .finally(() => {
      if (cityState.opening === id) cityState.opening = null;
      refreshBar();
    });
}

function onSceneChanged(e) {
  const id = e.detail ? e.detail.id : null;
  hideHover();
  // Deselect also cancels a scene still being opened.
  if (!id) cityState.opening = null;
  trajectoryLayer.setSelected(id ? trajectoryLayer.indexOf(id) : -1);
  // While the picked scene shows its own whole-scene cloud, its city chunks
  // would be the same points twice. The layer remembers the hidden scene, so
  // chunks of it that arrive later come in hidden too.
  const own = id && e.detail.aggregate ? id : null;
  for (const s of [...(pointLayer.hiddenCityScenes || [])]) {
    if (s !== own) pointLayer.setCitySceneVisible(s, true);
  }
  if (own) pointLayer.setCitySceneVisible(own, false);
  refreshBar();
}

// ---- point clouds around the camera ------------------------------------------
/**
 * Load the static clouds of every scene within the radius. `build` (the
 * button) may start builds on the server; Auto passes false and skips scenes
 * whose cloud is not built yet.
 */
export function loadHere(build = true) {
  if (!cityState.scenes.length) return Promise.resolve(0);
  const [fx, fy] = focus();
  lastFocus = [fx, fy];
  const near = trajectoryLayer.scenesNear(fx, fy, cityState.radius);
  return refreshStatus().then(() => {
    let started = 0;
    for (const n of near) {
      const s = cityState.scenes[n.index];
      const st = loadOf(s.id);
      if (["ready", "meta", "queued", "building"].includes(st.state)) continue;
      const built = status[s.id] ? status[s.id].aggregate : s.aggregate;
      if (!build && !built) {
        st.state = "unbuilt";
        continue;
      }
      requestMeta(s);
      started++;
    }
    enqueueChunks();
    refreshBar();
    return near.length;
  });
}

function refreshStatus() {
  if (!cityState.info) return Promise.resolve();
  return fetch(`${cityState.info.base}/city/status.json`)
    .then((r) => (r.ok ? r.json() : {}))
    .then((s) => { status = s || {}; })
    .catch(() => {});
}

function stillNear(s) {
  const [fx, fy] = focus();
  const index = trajectoryLayer.indexOf(s.id);
  return trajectoryLayer.distanceTo(index, fx, fy) <= cityState.radius * EVICT_FACTOR;
}

function requestMeta(s) {
  const st = loadOf(s.id);
  if (st.state !== "building" && st.state !== "queued") st.state = "meta";
  clearTimeout(st.timer);
  const gen = generation;
  return fetch(`${s.base}/aggregate/meta.json`)
    .then((r) => r.json().then((meta) => ({ code: r.status, meta })))
    .then(({ code, meta }) => {
      if (gen !== generation) return;
      if (meta.error) throw new Error(meta.error);
      if (code === 202 || !meta.ready) {
        const p = meta.progress || {};
        st.state = p.state === "queued" ? "queued" : "building";
        st.progress = p;
        // Keep asking only while it is still wanted; the server finishes the
        // build either way and caches it.
        st.timer = setTimeout(() => {
          if (gen !== generation) return;
          if (stillNear(s)) requestMeta(s);
          else st.state = "idle";
          refreshBar();
        }, BUILD_POLL_MS);
        refreshBar();
        return;
      }
      st.state = "ready";
      st.progress = null;
      st.chunks = meta.chunks || [];
      enqueueChunks();
      refreshBar();
    })
    .catch((err) => {
      if (gen !== generation) return;  // unloaded meanwhile: not this scene's fault
      st.state = "error";
      st.error = String(err.message || err);
      refreshBar();
    });
}

/** What to fetch around the camera: merged tiles, or each scene's chunks from servers without them. */
function enqueueChunks() {
  if (mergedTiles) enqueueTiles();
  else enqueueSceneChunks();
}

let tilesAsked = 0;
function enqueueTiles() {
  if (!cityState.info) return;
  const [fx, fy] = focus();
  const r = cityState.radius;
  const gen = generation;
  const asked = ++tilesAsked;
  fetch(`${cityState.info.base}/city/tiles.json?x=${fx.toFixed(1)}&y=${fy.toFixed(1)}&r=${r}`)
    .then((resp) => {
      if (resp.status === 404) {  // no merged tiles here: fall back to per-scene chunks
        mergedTiles = false;
        enqueueSceneChunks();
        return null;
      }
      return resp.ok ? resp.json() : null;
    })
    .then((answer) => {
      // A newer question (the camera moved, a scene got built) supersedes this one.
      if (!answer || gen !== generation || asked !== tilesAsked) return;
      for (let i = queue.length - 1; i >= 0; i--) if (queue[i].tile) queue.splice(i, 1);
      for (const t of answer.tiles || []) {
        const key = `tile:${t.key}`;
        const have = loaded.get(key);
        if (have && have.version === t.version) continue;
        queue.push({ key, id: "tiles", tile: true, url: `${cityState.info.base}/city/tile/${t.key}.bin`,
                     centre: t.centre, radius: t.radius, version: t.version, ids: t.ids || [],
                     gen: generation });
      }
      queue.sort((a, b) => Math.hypot(a.centre[0] - fx, a.centre[1] - fy)
        - Math.hypot(b.centre[0] - fx, b.centre[1] - fy));
      pump();
    })
    .catch(() => {});
}

function enqueueSceneChunks() {
  const [fx, fy] = focus();
  const r = cityState.radius;
  const queued = new Set(queue.map((j) => j.key));
  for (const s of cityState.scenes) {
    const st = sceneLoad.get(s.id);
    if (!st || st.state !== "ready") continue;
    for (const c of st.chunks) {
      const key = `${s.id}/${c.index}`;
      if (loaded.has(key) || queued.has(key)) continue;
      if (Math.hypot(c.centre[0] - fx, c.centre[1] - fy) > r + c.radius) continue;
      queue.push({ key, id: s.id, url: `${s.base}/aggregate/${c.index}.bin`, centre: c.centre,
                   radius: c.radius, gen: generation });
    }
  }
  queue.sort((a, b) => Math.hypot(a.centre[0] - fx, a.centre[1] - fy)
    - Math.hypot(b.centre[0] - fx, b.centre[1] - fy));
  pump();
}

const wanted = (job) => {
  const [fx, fy] = focus();
  return job.gen === generation
    && Math.hypot(job.centre[0] - fx, job.centre[1] - fy) <= cityState.radius * EVICT_FACTOR + job.radius;
};

/** One chunk at a time, so the picked scene's frames are never stuck behind them. */
function pump() {
  if (busy) return;
  let job = queue.shift();
  while (job && !wanted(job)) job = queue.shift();
  if (!job) {
    refreshBar();
    return;
  }
  busy = true;
  fetch(job.url)
    .then((r) => {
      // A listed tile can hold no voxel after all: remember it, do not ask again.
      if (r.status === 404 && job.tile) return "empty";
      return r.ok ? r.arrayBuffer() : null;
    })
    .then((buf) => {
      if (!buf || !wanted(job)) return;
      const entry = { id: job.id, centre: job.centre, radius: job.radius, version: job.version, ids: job.ids };
      if (buf === "empty") {
        pointLayer.removeCityChunk(job.key);
      } else {
        pointLayer.addCityChunk(job.key, job.id, buf);  // replaces an older version of the tile
      }
      loaded.set(job.key, entry);
      refreshPointUi();
    })
    .catch(() => {})
    .finally(() => {
      busy = false;
      refreshBar();
      pump();
    });
}

/** Drop chunks beyond the radius (with some slack), wherever the camera went. */
function evict() {
  const [fx, fy] = focus();
  const limit = cityState.radius * EVICT_FACTOR;
  let dropped = 0;
  for (const [key, c] of [...loaded]) {
    if (Math.hypot(c.centre[0] - fx, c.centre[1] - fy) <= limit + c.radius) continue;
    pointLayer.removeCityChunk(key);
    loaded.delete(key);
    dropped++;
  }
  for (let i = queue.length - 1; i >= 0; i--) if (!wanted(queue[i])) queue.splice(i, 1);
  // A scene whose points are all gone may be asked for again later.
  if (dropped) {
    const still = new Set([...loaded.values()].flatMap((c) => (c.ids ? c.ids : [c.id])));
    for (const [id, st] of sceneLoad) {
      if (st.state === "ready" && !still.has(id) && !stillNear(byId(id))) st.state = "idle";
    }
    refreshBar();
  }
  return dropped;
}

export function unloadAll() {
  generation++;
  queue.length = 0;
  for (const key of [...loaded.keys()]) pointLayer.removeCityChunk(key);
  loaded.clear();
  for (const st of sceneLoad.values()) {
    clearTimeout(st.timer);
    if (st.state !== "error") st.state = "idle";
  }
  lastFocus = null;
  refreshBar();
}

export function setAuto(on) {
  cityState.auto = !!on;
  if (els.auto) els.auto.classList.toggle("on", cityState.auto);
  if (cityState.auto) loadHere(false);
  refreshBar();
}

export function setRadius(metres) {
  cityState.radius = Math.max(50, Math.min(2000, metres));
  if (els.radius) els.radius.value = String(cityState.radius);
  if (els.radiusLabel) els.radiusLabel.textContent = `${cityState.radius} m`;
  if (els.load) els.load.textContent = `Load ${cityState.radius} m`;
  evict();
  if (cityState.auto) loadHere(false);
  refreshBar();
}

export function setTracks(on) {
  trajectoryLayer.setVisible(on);
  if (els.tracks) els.tracks.classList.toggle("on", trajectoryLayer.visible);
  if (!on) hideHover();
}

/** Per animation frame: shader uniforms every frame, load/evict a few times a second. */
export function update() {
  trajectoryLayer.update(renderer, camera);
  // The label names what was under the pointer; once the view moves, it
  // would name something else.
  if (els.hover && !els.hover.hidden && camKey() !== hoverCamKey) hideHover();
  if (!cityState.scenes.length) return;
  const t = performance.now();
  if (t - lastTick < TICK_MS) return;
  lastTick = t;
  evict();
  if (cityState.auto) {
    const [fx, fy] = focus();
    if (!lastFocus || Math.hypot(fx - lastFocus[0], fy - lastFocus[1]) > cityState.radius / 4) {
      loadHere(false);
    }
  }
}

const countLabel = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M`
  : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n));

function refreshBar() {
  if (!els.bar) return;
  const n = cityState.scenes.length;
  els.name.textContent = cityState.loading ? `${cityState.name || "city"} · loading…`
    : `${cityState.name || "city"} · ${n} scene${n === 1 ? "" : "s"}`;
  els.deselect.disabled = !sceneState.info;
  els.load.textContent = `Load ${cityState.radius} m`;
  const parts = [];
  const c = pointLayer.cityStats();
  if (c.chunks) {
    const ids = new Set([...loaded.values()].flatMap((t) => (t.ids ? t.ids : [t.id])));
    parts.push(`${countLabel(c.points)} pts from ${ids.size} scene${ids.size === 1 ? "" : "s"}`
      + (mergedTiles ? ", merged" : ""));
  }
  const by = {};
  for (const st of sceneLoad.values()) by[st.state] = (by[st.state] || 0) + 1;
  const building = [...sceneLoad.values()].find((st) => st.state === "building");
  if (building) {
    const p = building.progress || {};
    parts.push(`building ${p.total ? `${p.done || 0}/${p.total}` : "…"}`
      + (by.building > 1 ? ` (+${by.building - 1})` : ""));
  }
  if (by.queued) parts.push(`${by.queued} queued`);
  if (queue.length || busy) parts.push(`${queue.length + (busy ? 1 : 0)} chunks to fetch`);
  if (by.unbuilt) parts.push(`${by.unbuilt} not built (Load builds)`);
  if (by.error) parts.push(`${by.error} failed`);
  if (cityState.opening) parts.push(`opening ${shortId(cityState.opening)}…`);
  els.status.textContent = parts.join(" · ");
  // Builds and fetches can take minutes: say so in the status bar too.
  const busyParts = parts.filter((p) => /^(building|\d+ queued|\d+ chunks|opening)/.test(p));
  setJob("city", busyParts.length ? `City: ${busyParts.join(" · ")}` : null);
  window.dispatchEvent(new CustomEvent("ll2:crumbs", { detail: { city: cityState.name || "city" } }));
  renderSceneList(false);
}

// ---- the scene list ------------------------------------------------------------
// Every scene of the city, nearest the camera first. Opening one from the
// list (rather than from its trajectory) also goes to the car: the camera
// may be anywhere.
const LIST_MAX = 150;
const LIST_REFRESH_MS = 1000;
let listKey = "";

function installSceneList() {
  els.list.addEventListener("click", (e) => {
    const item = e.target.closest("[data-scene]");
    if (!item) return;
    const id = item.dataset.scene;
    if (sceneState.info && sceneState.info.id === id) {
      goToCar();
      return;
    }
    openScene(id, 0).then((ok) => { if (ok) goToCar(); });
  });
  els.list.addEventListener("pointerover", (e) => {
    const item = e.target.closest("[data-scene]");
    trajectoryLayer.setHovered(item ? trajectoryLayer.indexOf(item.dataset.scene) : -1);
  });
  els.list.addEventListener("pointerleave", () => trajectoryLayer.setHovered(-1));
  // Distances change as the camera moves; only worth redoing while visible.
  setInterval(() => {
    if (panelOpen() && currentTab() === "scene" && cityState.scenes.length) renderSceneList(false);
  }, LIST_REFRESH_MS);
}

const fmtDist = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);

function renderSceneList(force) {
  if (!els.list) return;
  const q = els.search ? els.search.value.trim().toLowerCase() : "";
  const [fx, fy] = focus();
  const rows = cityState.scenes.map((s, index) => ({
    s, index, dist: trajectoryLayer.distanceTo(index, fx, fy),
  })).filter((r) => !q || r.s.id.toLowerCase().includes(q) || String(r.s.split || "").toLowerCase().includes(q));
  rows.sort((a, b) => a.dist - b.dist);
  const shown = rows.slice(0, LIST_MAX);
  const current = sceneState.info ? sceneState.info.id : null;
  // Rebuild only when something visible changed (distances to 10 m).
  const key = `${q}|${current}|${shown.map((r) => `${r.s.id}:${Math.round(r.dist / 10)}:${(sceneLoad.get(r.s.id) || {}).state}`).join()}`;
  if (!force && key === listKey) return;
  listKey = key;
  els.list.textContent = "";
  for (const r of shown) {
    const st = sceneLoad.get(r.s.id);
    const built = status[r.s.id] ? status[r.s.id].aggregate : r.s.aggregate;
    const item = document.createElement("button");
    item.type = "button";
    item.className = "sceneItem" + (r.s.id === current ? " current" : "");
    item.dataset.scene = r.s.id;
    item.setAttribute("role", "listitem");
    item.title = `${r.s.id}\nClick: open it and go to the car`;
    const sid = document.createElement("span");
    sid.className = "sid";
    sid.textContent = r.s.id;
    const dist = document.createElement("span");
    dist.className = "dist";
    dist.textContent = Number.isFinite(r.dist) ? fmtDist(r.dist) : "";
    const meta = document.createElement("span");
    meta.className = "meta";
    const bits = [r.s.split, r.s.frames ? `${r.s.frames} frames` : null].filter(Boolean);
    meta.textContent = bits.join(" · ");
    const state = r.s.id === current ? "open" : st && st.state === "ready" ? "cloud loaded"
      : st && (st.state === "building" || st.state === "queued") ? st.state : built ? "cloud built" : null;
    if (state) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = ` · ${state}`;
      meta.appendChild(tag);
    }
    item.append(sid, dist, meta);
    els.list.appendChild(item);
  }
  if (els.listCount) {
    els.listCount.textContent = rows.length === cityState.scenes.length
      ? `(${rows.length})` : `(${rows.length} of ${cityState.scenes.length})`;
  }
}

if (TEST_HOOKS) {
  window.__ll2city = {
    state: () => ({
      name: cityState.name, scenes: cityState.scenes.length, loading: cityState.loading,
      error: cityState.error, radius: cityState.radius, auto: cityState.auto,
      opening: cityState.opening, selected: sceneState.info ? sceneState.info.id : null,
      frame: sceneState.frame,
      load: Object.fromEntries([...sceneLoad].map(([id, st]) => [id, st.state])),
      loaded: [...loaded.keys()].sort(), queue: queue.length, busy,
      tracks: trajectoryLayer.stats(),
      points: pointLayer.cityStats(),
      hover: els.hover && !els.hover.hidden ? els.hover.textContent : null,
    }),
    ids: () => cityState.scenes.map((s) => s.id),
    // Where frame `frame` of scene `id` is drawn, in viewer ENU.
    nodePosition: (id, frame) => trajectoryLayer.nodePosition(trajectoryLayer.indexOf(id), frame),
    // Look down at a point from `dist` metres south and above it.
    lookAt: (p, dist) => {
      setCameraView(new THREE.Vector3(p[0], p[1] - dist * 0.6, p[2] + dist * 0.8), view.yaw, view.pitch);
      lookAtPoint(new THREE.Vector3(p[0], p[1], p[2]));
    },
    pick: (x, y) => pickAt(x, y),
    open: (id, frame) => openScene(id, frame),
    loadHere: (build = true) => loadHere(build),
    setAuto: (on) => setAuto(on),
    setRadius: (m) => setRadius(m),
    unload: () => unloadAll(),
    evict: () => evict(),
  };
}
