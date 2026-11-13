// Recorded sensor scenes (KITScenes) next to the map.
//
// The scene server announces itself with one `scene` message over the same
// SSE stream the map arrives on; everything heavy is then fetched straight
// from it as binary, because a frame is ~750 k points and has no business
// going through JSON. The viewer server only relays the announcement, so it
// keeps knowing nothing about sensors.
import * as THREE from "three";
import { TEST_HOOKS } from "../util.js";
import { camera, view, setCameraView, lookAtPoint } from "../camera.js";
import { pointLayer, setClassPalette, setHiddenClasses, SENSOR_COLORS } from "../render/points.js";
import { toast } from "../hud.js";
import { setIcon } from "../ui/icons.js";
import { setCrumbs, setJob, setTabAvailable } from "../ui/shell.js";
import { store } from "../store.js";
import { installCamRig, setSceneInfo, update as updateRig, rig, rigState, handleRigKey,
         projectToImage } from "./camrig.js";

/** The whole-scene aggregate: built on the server, streamed in chunks. */
export const aggregateState = {
  on: false, building: false, chunks: 0, loaded: 0, points: 0, error: null, stats: null,
  framesWere: undefined,
};

export const sceneState = {
  info: null,      // the scene message's payload
  frame: 0,
  loading: false,
  playing: false,
  follow: false,   // the camera rides along with the car
  error: null,
};

// Follow: the camera keeps its place relative to the car, so it is moved
// and turned with the car from the pose it was last placed against.
const FOLLOW_NEAR_M = 120;  // turning Follow on further away than this goes to the car first
let followFrom = null;      // frame whose car pose the camera currently sits against
const _zAxis = new THREE.Vector3(0, 0, 1);

const PLAY_MIN_MS = 60;       // never queue faster than the fetches come back
const PREFETCH_IDLE_MS = 400; // only look ahead once the user has settled
let els = {};
let playTimer = null;
let prefetchTimer = null;
// Exactly one request is ever in flight, and the newest wanted frame wins:
// whatever is in flight finishes, then the latest wanted frame is fetched, so
// with a held step key the frames stepped over are never asked for at all.
// The server decodes one frame at a time (0.1-0.3 s, more with colours and
// labels).
let pending = null;  // { index, controller, show }
let wanted = null;   // the frame the user last asked for
const cache = new Map(); // frame -> ArrayBuffer
const CACHE_MAX = 6;

/** Wire the scene bar. Safe to call when the page has no scene (yet). */
export function installScene() {
  els = {
    bar: document.getElementById("sceneBar"),
    name: document.getElementById("sceneName"),
    points: document.getElementById("scenePoints"),
    car: document.getElementById("sceneCar"),
    prev: document.getElementById("scenePrev"),
    play: document.getElementById("scenePlay"),
    next: document.getElementById("sceneNext"),
    frame: document.getElementById("sceneFrame"),
    label: document.getElementById("sceneFrameLabel"),
    mode: document.getElementById("sceneMode"),
    size: document.getElementById("sceneSize"),
    aggregate: document.getElementById("sceneAggregate"),
    tint: document.getElementById("sceneTint"),
    filterBtn: document.getElementById("sceneFilterBtn"),
    filters: document.getElementById("sceneFilters"),
    alpha: document.getElementById("pointAlpha"),
    aboveGround: document.getElementById("aboveGround"),
    aboveGroundLabel: document.getElementById("aboveGroundLabel"),
    intensityMin: document.getElementById("intensityMin"),
    intensityMinLabel: document.getElementById("intensityMinLabel"),
    chips: document.getElementById("classChips"),
    presetCars: document.getElementById("classPresetCars"),
    showAll: document.getElementById("classShowAll"),
    classMode: document.getElementById("classMode"),
    classSect: document.getElementById("classSect"),
    filterReset: document.getElementById("filterReset"),
    follow: document.getElementById("sceneFollow"),
    meta: document.getElementById("sceneMeta"),
    legend: document.getElementById("pointLegend"),
    timeline: document.getElementById("timeline"),
  };
  if (!els.bar) return;
  installFilters();
  els.aggregate.addEventListener("click", () => toggleAggregate());
  els.tint.addEventListener("click", () => toggleTint());
  els.points.classList.add("on");
  els.points.addEventListener("click", () => togglePoints());
  els.car.addEventListener("click", () => goToCar());
  if (els.follow) els.follow.addEventListener("click", () => setFollow(!sceneState.follow));
  els.prev.addEventListener("click", () => step(-1));
  els.next.addEventListener("click", () => step(1));
  els.play.addEventListener("click", () => setPlaying(!sceneState.playing));
  els.frame.addEventListener("input", () => showFrame(Number(els.frame.value)));
  els.mode.addEventListener("change", () => {
    pointLayer.setMode(els.mode.value);
    refreshLegend();
  });
  els.size.addEventListener("input", () => pointLayer.setSize(Number(els.size.value)));
  pointLayer.setSize(Number(els.size.value));
  refreshLegend();
  installCamRig({ urlFor: imageUrl, currentFrame: () => sceneState.frame });
}

// ---- filters -------------------------------------------------------------------
// Everything here is a shader uniform, so dragging a slider never refetches
// or re-uploads a point: filtered points are pushed outside the clip volume.
// `ticked` are the class chips that are ticked; `only` says what that means:
// hide those classes, or show nothing else.
export const filterState = { ticked: new Set(), only: false, aboveGround: 0, intensityMin: 0, alpha: 1 };

// What "Vehicles" hides: the Vistas thing classes that are traffic.
const VEHICLE_CLASSES = [55, 57, 58, 59, 60, 61, 62, 63, 64, 32, 33];

function installFilters() {
  if (!els.filters) return;
  els.filterBtn.addEventListener("click", () => {
    els.filters.hidden = !els.filters.hidden;
    els.filterBtn.parentElement.classList.toggle("closed", els.filters.hidden);
    refreshFilterButton();
  });
  els.filterBtn.parentElement.classList.toggle("closed", els.filters.hidden);
  if (els.filterReset) els.filterReset.addEventListener("click", () => resetFilters());
  els.alpha.addEventListener("input", () => {
    filterState.alpha = Number(els.alpha.value) / 100;
    pointLayer.setAlpha(filterState.alpha);
    refreshFilterButton();
  });
  els.aboveGround.addEventListener("input", () => {
    filterState.aboveGround = Number(els.aboveGround.value);
    els.aboveGroundLabel.textContent = filterState.aboveGround
      ? `${filterState.aboveGround.toFixed(1)} m` : "off";
    pointLayer.setAboveGround(filterState.aboveGround);
    refreshFilterButton();
  });
  els.intensityMin.addEventListener("input", () => {
    filterState.intensityMin = Number(els.intensityMin.value) / 100;
    els.intensityMinLabel.textContent = filterState.intensityMin.toFixed(2);
    pointLayer.setIntensityRange(filterState.intensityMin, 1);
    refreshFilterButton();
  });
  els.presetCars.addEventListener("click", () => hideClasses(VEHICLE_CLASSES));
  els.showAll.addEventListener("click", () => hideClasses([]));
  els.classMode.addEventListener("click", () => setClassMode(!filterState.only));
}

/**
 * The class ids that are not drawn. "Only" with nothing ticked filters
 * nothing: an empty view on the way to ticking the first class helps nobody.
 */
export function hiddenClasses() {
  if (!filterState.only) return new Set(filterState.ticked);
  if (!filterState.ticked.size) return new Set();
  const out = new Set();
  for (let id = 0; id < 256; id++) if (!filterState.ticked.has(id)) out.add(id);
  return out;
}

function applyClassFilter() {
  setHiddenClasses(hiddenClasses());
  for (const label of els.chips ? els.chips.querySelectorAll("label") : []) {
    const id = Number(label.dataset.classId);
    const box = label.querySelector("input");
    box.checked = filterState.ticked.has(id);
    label.classList.toggle("hidden", box.checked && !filterState.only);
    label.classList.toggle("only", box.checked && filterState.only);
  }
  if (els.classMode) {
    els.classMode.textContent = filterState.only ? "Only" : "Hide";
    els.classMode.classList.toggle("on", filterState.only);
  }
  refreshFilterButton();
}

/** Tick exactly these class ids (hidden, or the only ones shown, per the mode). */
export function hideClasses(ids) {
  filterState.ticked = new Set(ids);
  applyClassFilter();
}

/** Hide the ticked classes (false) or show only them (true). */
export function setClassMode(only) {
  filterState.only = !!only;
  applyClassFilter();
}

/** Every filter back to letting everything through. */
export function resetFilters() {
  filterState.alpha = 1;
  filterState.aboveGround = 0;
  filterState.intensityMin = 0;
  els.alpha.value = "100";
  els.aboveGround.value = "0";
  els.intensityMin.value = "0";
  for (const input of [els.alpha, els.aboveGround, els.intensityMin]) {
    input.dispatchEvent(new Event("input"));
  }
  filterState.only = false;
  hideClasses([]);
}

/** CSS colour of a class, from the palette the scene server sent. */
function classColour(id) {
  const rows = (sceneState.info && sceneState.info.classes && sceneState.info.classes.palette)
    || (classTable && classTable.palette);
  const c = rows && rows[id];
  return c ? `rgb(${c[0]}, ${c[1]}, ${c[2]})` : "#5b6576";
}

/** A chip per class actually present, commonest first. */
function refreshClassChips() {
  if (!els.chips) return;
  const histogram = pointLayer.classHistogram(true);
  if (!histogram) return;
  const present = Object.entries(histogram)
    .filter(([id]) => Number(id) !== 255)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 14);
  if (!present.length) return;
  const have = new Set(Array.from(els.chips.children).map((n) => n.dataset.classId));
  for (const [id] of present) {
    if (have.has(id)) continue;
    const label = document.createElement("label");
    label.dataset.classId = id;
    label.title = `${className(Number(id))} (class ${id})`;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = filterState.ticked.has(Number(id));
    box.addEventListener("change", () => {
      if (box.checked) filterState.ticked.add(Number(id));
      else filterState.ticked.delete(Number(id));
      applyClassFilter();
    });
    label.appendChild(box);
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = classColour(Number(id));
    label.appendChild(swatch);
    label.appendChild(document.createTextNode(className(Number(id))));
    label.classList.toggle("hidden", box.checked && !filterState.only);
    label.classList.toggle("only", box.checked && filterState.only);
    els.chips.appendChild(label);
  }
  if (els.classSect) els.classSect.hidden = els.chips.children.length === 0;
}

function refreshFilterButton() {
  if (!els.filterBtn) return;
  const active = filterState.ticked.size + (filterState.aboveGround ? 1 : 0)
    + (filterState.intensityMin > 0 ? 1 : 0) + (filterState.alpha < 1 ? 1 : 0);
  const label = els.filterBtn.querySelector(".lbl") || els.filterBtn;
  label.textContent = active
    ? `Filter · ${filterState.only && filterState.ticked.size ? "only " : ""}${active}` : "Filter";
  els.filterBtn.classList.toggle("on", active > 0);
  if (els.filterReset) els.filterReset.disabled = active === 0;
}

// Viridis, as the point shader computes it, for the legend's ramp.
const VIRIDIS_CSS = "linear-gradient(90deg, #440154, #482878, #3e4989, #31688e, #26828e, "
  + "#1f9e89, #35b779, #6ece58, #b5de2b, #fde725)";
// The dataset's sensor_id order (render/points.js).
const SENSOR_NAMES = ["top", "corner left", "corner right", "left", "right", "front", "rear"];

/** What the point colours mean, for the current colour mode. */
function refreshLegend() {
  const el = els.legend;
  if (!el) return;
  el.textContent = "";
  const mode = pointLayer.mode;
  const add = (cls, text) => {
    const d = document.createElement("div");
    d.className = cls;
    if (text) d.textContent = text;
    el.appendChild(d);
    return d;
  };
  if (mode === "height" || mode === "intensity") {
    add("ramp").style.background = VIRIDIS_CSS;
    const ends = add("ends");
    const [lo, hi] = mode === "height" ? ["low", "high"] : ["dim", "bright"];
    ends.innerHTML = `<span>${lo}</span><span>${hi}</span>`;
  } else if (mode === "sensor") {
    SENSOR_NAMES.forEach((name, i) => {
      const key = add("key");
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = `#${SENSOR_COLORS[i].toString(16).padStart(6, "0")}`;
      key.appendChild(sw);
      key.appendChild(document.createTextNode(name));
    });
  } else if (mode === "class") {
    add("", "Colours as in the class list below.");
  } else if (mode === "rgb") {
    add("", "Colour sampled from the ring camera images.");
  } else if (mode === "instance") {
    add("", "One colour per detected object, within this frame only; the whole-scene cloud keeps intensity.");
  }
}

/**
 * A `scene` message from the server, or a scene picked on the city map
 * (`startFrame` then says which frame was clicked).
 */
export function onSceneMessage(msg, startFrame) {
  const info = msg.scene;
  if (!info || !info.base) return;
  // A reloaded tab gets the same announcement again; only another scene
  // throws away what this one has loaded.
  const same = sceneState.info && sceneState.info.id === info.id
    && sceneState.info.base === info.base;
  // An EventSource reconnect replays the bootstrap snapshot, scene included.
  // Nothing about the scene changed: replacing its info object would orphan
  // the whole-scene load in flight and release the camera lock.
  if (same) {
    if (startFrame !== undefined) showFrame(startFrame);
    return;
  }
  if (sceneState.info) resetScene();
  sceneState.info = info;
  if (startFrame !== undefined) sceneState.frame = startFrame;
  cache.clear();
  setSceneInfo(info);
  // Classes come with their own Vistas palette and names, so the browser has
  // nothing hard-coded to drift from the server's table.
  if (info.classes && info.classes.palette) setClassTable(info.classes);
  if (els.bar) {
    refreshBar();
    els.name.textContent = info.id;
    els.name.title = `${info.id}\n${info.base}`;
    const nCams = Object.keys(info.cameras || {}).length;
    const duration = info.times && info.times.length ? info.times[info.times.length - 1] - info.times[0] : null;
    if (els.meta) {
      els.meta.textContent = `${info.frames} frames${duration ? ` · ${duration.toFixed(1)} s` : ""}`
        + ` · ${(info.lidars || []).length} LiDARs · ${nCams} cameras`;
    }
    els.frame.max = String(Math.max(0, info.frames - 1));
  }
  setTabAvailable("scene", true, { focus: true });
  refreshLegend();
  showFrame(Math.min(sceneState.frame, info.frames - 1));
  announce();
}

/** Tell the city layer (and anyone else) which scene is current. */
function announce() {
  window.dispatchEvent(new CustomEvent("ll2:scene", {
    detail: { id: sceneState.info ? sceneState.info.id : null, aggregate: aggregateState.on },
  }));
}

/** Drop everything the current scene has loaded, ready for another or none. */
function resetScene() {
  setPlaying(false);
  setFollow(false);
  if (pending) {
    pending.controller.abort();
    pending = null;
  }
  wanted = null;
  clearTimeout(prefetchTimer);
  clearTimeout(aggregateTimer);
  cache.clear();
  if (aggregateState.on) toggleAggregate(false);
  pointLayer.clear();
  pointLayer.clearAggregate();
  Object.assign(aggregateState, { on: false, building: false, chunks: 0, loaded: 0, points: 0,
                                  error: null, stats: null, framesWere: undefined });
  refreshAggregateButton();
  sceneState.loading = false;
  sceneState.error = null;
  sceneState.frame = 0;
  // What the old scene's data offered goes with it; whatever is still loaded
  // (the city's clouds) offers it again just below.
  for (const value of ["rgb", "instance"]) {
    const option = els.mode && els.mode.querySelector(`option[value="${value}"]`);
    if (option) option.remove();
  }
  if (els.chips) els.chips.textContent = "";
  refreshColourOptions();
  refreshClassChips();
  if (els.mode && !els.mode.querySelector(`option[value="${pointLayer.mode}"]`)) {
    els.mode.value = "intensity";
    pointLayer.setMode("intensity");
  }
}

/** No current scene any more: the city map is all that is left. */
export function clearScene() {
  if (!sceneState.info) return;
  resetScene();
  sceneState.info = null;
  setSceneInfo(null);  // releases the camera lock
  refreshBar();
  announce();
}

// In city mode the colour, size and filter controls act on the city's clouds
// too, so the bar stays up without a scene; only what needs one is hidden.
let cityActive = false;

export function setCityActive(on) {
  cityActive = !!on;
  refreshBar();
}

function refreshBar() {
  if (!els.bar) return;
  els.bar.hidden = !(sceneState.info || cityActive);
  els.bar.classList.toggle("noScene", !sceneState.info);
  for (const el of els.bar.querySelectorAll("[data-scene-only]")) el.hidden = !sceneState.info;
  if (els.timeline) els.timeline.hidden = !sceneState.info;
  document.body.classList.toggle("hasTimeline", !!sceneState.info);
  if (!sceneState.info) els.name.textContent = "no scene · click a trajectory";
  setTabAvailable("scene", !!(sceneState.info || cityActive));
  sceneCrumb();
}

/** The scene part of the breadcrumb; the city part is city.js's. */
function sceneCrumb() {
  const info = sceneState.info;
  window.dispatchEvent(new CustomEvent("ll2:crumbs", {
    detail: { scene: info ? shortId(info.id) : null, sceneTitle: info ? info.id : "" },
  }));
}

// The Vistas table: from the scene, or from the city before any scene is open.
let classTable = null;

/** Install class colours and names: `classes` is { palette, names, ... }. */
export function setClassTable(classes) {
  if (!classes || !classes.palette) return;
  classTable = classes;
  setClassPalette(classes.palette);
  addClassOption(classes);
}

/** After points arrived from elsewhere (the city): offer what they carry. */
export function refreshPointUi() {
  refreshColourOptions();
  refreshClassChips();
}

const shortId = (id) => (id && id.length > 12 ? `${id.slice(0, 8)}…` : id || "scene");

/** Add a colour mode to the menu once the data turns out to carry it. */
function addColourOption(value, title) {
  if (!els.mode || els.mode.querySelector(`option[value="${value}"]`)) return;
  const option = document.createElement("option");
  option.value = value;
  option.textContent = value;
  option.title = title;
  els.mode.appendChild(option);
}

function addClassOption(classes) {
  addColourOption("class", "Mapillary Vistas classes from the seamseg masks of the "
    + "front and rear cameras");
}

/** The camera colour only shows up once a cloud with it has been decoded. */
function refreshColourOptions() {
  if (pointLayer.hasRgb()) {
    addColourOption("rgb", "Colour sampled from the ring camera images");
  }
  if (pointLayer.hasInstances()) {
    addColourOption("instance", "A colour per detected object in this frame. "
      + "Nothing tracks an object between frames, so the numbers are good only "
      + "within one frame and the aggregate keeps its intensity colouring.");
  }
}

/** Name of a Vistas class id, for the HUD and the tests. */
export function className(id) {
  const info = sceneState.info;
  const names = (info && info.classes && info.classes.names) || (classTable && classTable.names);
  if (id === undefined || id === null) return null;
  if (names && names[String(id)]) return names[String(id)];
  return id === 255 ? "unlabelled" : `class ${id}`;
}

export function frameUrl(index) {
  return `${sceneState.info.base}/frame/${index}/points.bin`;
}

// ---- the whole-scene aggregate ------------------------------------------------
// Building it means decoding every frame, so the server does it once in the
// background and caches it; until it is ready the meta route answers 202 with
// progress, which the button shows.
const AGGREGATE_POLL_MS = 1000;
let aggregateTimer = null;

export function toggleAggregate(on) {
  const next = on === undefined ? !aggregateState.on : !!on;
  if (next === aggregateState.on) return;
  aggregateState.on = next;
  pointLayer.setAggregateVisible(next);
  // The point of the aggregate is to look at the scene without one frame's
  // clutter on top, so showing it puts the frame away -- P brings it back,
  // tinted, when the two are worth comparing. Turning the aggregate off
  // restores whatever the frame was doing before.
  if (next) {
    aggregateState.framesWere = pointLayer.visible;
    togglePoints(false);
  } else if (aggregateState.framesWere !== undefined) {
    togglePoints(aggregateState.framesWere);
    aggregateState.framesWere = undefined;
  }
  refreshAggregateButton();
  if (next && !aggregateState.loaded && !aggregateState.building) loadAggregate();
  announce();
}

/** Paint the single frame one bright colour while the aggregate is shown. */
export function toggleTint(on) {
  const next = on === undefined ? !pointLayer.tintFrame : !!on;
  pointLayer.setTintFrame(next);
  if (els.tint) els.tint.classList.toggle("on", next);
}

function loadAggregate() {
  const info = sceneState.info;
  if (!info) return Promise.resolve(null);
  aggregateState.building = true;
  aggregateState.error = null;
  refreshAggregateButton();
  return fetch(`${info.base}/aggregate/meta.json`)
    .then((r) => r.json().then((meta) => ({ status: r.status, meta })))
    .then(({ status, meta }) => {
      if (sceneState.info !== info) return null;  // another scene since
      if (meta.error) throw new Error(meta.error);
      if (status === 202 || !meta.ready) {
        aggregateState.stats = meta.progress || null;
        refreshAggregateButton();
        clearTimeout(aggregateTimer);
        aggregateTimer = setTimeout(loadAggregate, AGGREGATE_POLL_MS);
        return null;
      }
      aggregateState.stats = meta.stats || null;
      aggregateState.chunks = meta.chunks.length;
      aggregateState.loaded = 0;
      aggregateState.points = 0;
      pointLayer.clearAggregate();
      return fetchChunks(meta.chunks);
    })
    .catch((err) => {
      aggregateState.building = false;
      aggregateState.error = String(err.message || err);
      toast(`Whole scene: ${aggregateState.error}`, "error");
      refreshAggregateButton();
      return null;
    });
}

/** One chunk at a time, so a frame the user asks for is never stuck behind them. */
function fetchChunks(chunks) {
  const info = sceneState.info;
  const next = (i) => {
    if (i >= chunks.length || !sceneState.info || sceneState.info !== info) {
      aggregateState.building = false;
      refreshAggregateButton();
      return Promise.resolve(null);
    }
    return fetch(`${info.base}/aggregate/${chunks[i].index}.bin`)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((buf) => {
        if (buf && sceneState.info === info) {
          aggregateState.points += pointLayer.addAggregateChunk(buf);
          refreshColourOptions();
          refreshClassChips();
          aggregateState.loaded++;
          refreshAggregateButton();
        }
        return next(i + 1);
      });
  };
  return next(0);
}

const countLabel = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M`
  : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n));

function refreshAggregateButton() {
  if (!els.aggregate) return;
  els.aggregate.classList.toggle("on", aggregateState.on);
  const s = aggregateState;
  let label = "Whole scene";
  let job = null;
  if (s.error) label = "Whole scene ✕";
  else if (s.building && s.chunks) {
    label = `Whole scene ${s.loaded}/${s.chunks}`;
    job = `Loading the whole scene ${s.loaded}/${s.chunks}`;
  } else if (s.building) {
    const p = s.stats || {};
    label = p.total ? `Building ${p.done || 0}/${p.total}` : "Building…";
    job = p.total ? `Building the whole scene ${p.done || 0}/${p.total}` : "Building the whole scene…";
  } else if (s.points) label = `Whole scene · ${countLabel(s.points)} points`;
  const lbl = els.aggregate.querySelector(".lbl") || els.aggregate;
  lbl.textContent = label;
  els.aggregate.classList.toggle("busy", !!s.building);
  els.aggregate.classList.toggle("bad", !!s.error);
  if (s.error) els.aggregate.title = `Whole scene: ${s.error}`;
  setJob("aggregate", job);
}

/** Where a recorded camera frame lives; the camera lock blends it over the render. */
export function imageUrl(cameraName, index) {
  return `${sceneState.info.base}/frame/${index}/camera/${cameraName}.jpg`;
}

/** Load and show one frame; returns a promise for the tests. */
export function showFrame(index) {
  const info = sceneState.info;
  if (!info) return Promise.resolve(null);
  index = Math.max(0, Math.min(info.frames - 1, Math.round(index)));
  sceneState.frame = index;
  wanted = index;
  followTo(index);
  updateLabels();
  // The locked view rides along with the frame, image and all.
  if (rig.on) updateRig(index);
  const cached = cache.get(index);
  if (cached) {
    apply(index, cached);
    return Promise.resolve(cached);
  }
  if (pending) {
    if (pending.index === index) {
      // The look-ahead is now what the user waits for: let it say so, and
      // report a failure, instead of finishing silently.
      if (!pending.show) {
        pending.show = true;
        sceneState.loading = true;
        updateLabels();
      }
      return pending.promise;
    }
    // A look-ahead is worth dropping for a frame the user is waiting for; a
    // request already being decoded is not -- aborting it would send one
    // request per keypress anyway, and the server has started the work. Let
    // it finish, and its `finally` jumps straight to whatever is wanted by
    // then, so the frames stepped over are never asked for.
    if (!pending.show) {
      pending.controller.abort();
      return startFetch(index, true);
    }
    sceneState.loading = true;
    return pending.promise;
  }
  return startFetch(index, true);
}

/** Fetch one frame. `show` distinguishes a user step from a look-ahead. */
function startFetch(index, show) {
  const controller = new AbortController();
  // `job.show` may be raised later, when the user steps onto a look-ahead.
  const job = { index, controller, show, promise: null };
  if (show) {
    sceneState.loading = true;
    updateLabels();
  }
  const promise = fetch(frameUrl(index), { signal: controller.signal })
    .then((r) => {
      if (!r.ok) throw new Error(`frame ${index}: HTTP ${r.status}`);
      return r.arrayBuffer();
    })
    .then((buf) => {
      remember(index, buf);
      sceneState.error = null;
      if (job.show && sceneState.frame === index) apply(index, buf);
      return buf;
    })
    .catch((err) => {
      if (err && err.name === "AbortError") return null; // we moved on; normal
      if (job.show) {
        sceneState.error = String(err.message || err);
        setPlaying(false);
        toast(`Point cloud: ${sceneState.error}`, "error");
      }
      return null;
    })
    .finally(() => {
      // An aborted look-ahead settles *after* its replacement has taken the
      // slot; clearing it blindly would let two requests run at once.
      if (!pending || pending.controller !== controller) return;
      pending = null;
      sceneState.loading = false;
      if (wanted !== null && wanted !== index && !cache.has(wanted)) {
        startFetch(wanted, true);
        return;
      }
      if (wanted !== null && cache.has(wanted) && sceneState.frame === wanted) {
        apply(wanted, cache.get(wanted));
      }
      updateLabels();
      schedulePrefetch();
    });
  job.promise = promise;
  pending = job;
  return promise;
}

function apply(index, buffer) {
  try {
    pointLayer.setCloud(buffer);
    refreshColourOptions();
    refreshClassChips();
  } catch (err) {
    sceneState.error = String(err.message || err);
    toast(`Point cloud: ${sceneState.error}`, "error");
  }
  updateLabels();
  schedulePrefetch();
}

function remember(index, buffer) {
  cache.set(index, buffer);
  while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
}

/** Look one frame ahead, but only once nothing else is being waited for. */
function schedulePrefetch() {
  clearTimeout(prefetchTimer);
  prefetchTimer = setTimeout(() => {
    const info = sceneState.info;
    if (!info || pending) return;
    const next = sceneState.frame + 1;
    if (next < info.frames && !cache.has(next)) startFetch(next, false);
  }, PREFETCH_IDLE_MS);
}

export function step(delta) {
  if (!sceneState.info) return;
  showFrame(sceneState.frame + delta);
}

export function setPlaying(on) {
  sceneState.playing = !!on && !!sceneState.info;
  if (els.play) {
    setIcon(els.play.querySelector("[data-icon]"), sceneState.playing ? "pause" : "play");
    els.play.classList.toggle("on", sceneState.playing);
    els.play.dataset.tip = sceneState.playing ? "Pause" : "Play the scene";
  }
  clearTimeout(playTimer);
  if (sceneState.playing) queueNext();
}

function queueNext() {
  playTimer = setTimeout(() => {
    if (!sceneState.playing) return;
    const info = sceneState.info;
    const next = sceneState.frame + 1;
    if (next >= info.frames) {
      setPlaying(false);
      return;
    }
    const done = showFrame(next);
    Promise.resolve(done).then(() => { if (sceneState.playing) queueNext(); });
  }, PLAY_MIN_MS);
}

export function togglePoints(on) {
  const next = on === undefined ? !pointLayer.visible : !!on;
  pointLayer.setVisible(next);
  if (els.points) els.points.classList.toggle("on", next);
}

/** Camera behind and above the car at the current frame, looking where it drives. */
export function goToCar() {
  const info = sceneState.info;
  if (!info || !info.ego || !info.ego.pos[sceneState.frame]) return false;
  const pos = info.ego.pos[sceneState.frame];
  const quat = info.ego.quat[sceneState.frame];
  // The ego frame is x forward, y left, z up.
  const forward = new THREE.Vector3(1, 0, 0)
    .applyQuaternion(new THREE.Quaternion(quat[0], quat[1], quat[2], quat[3]));
  const horiz = Math.hypot(forward.x, forward.y) || 1;
  const car = new THREE.Vector3(pos[0], pos[1], pos[2]);
  const back = 14;
  const up = 7;
  // Stand behind and above the car, then aim at it. Pointing the camera with
  // `lookAtPoint` keeps this independent of the yaw convention, which is
  // atan2(-x, y) here and easy to get backwards.
  setCameraView(new THREE.Vector3(
    car.x - (forward.x / horiz) * back,
    car.y - (forward.y / horiz) * back,
    car.z + up), view.yaw, view.pitch);
  lookAtPoint(car);
  return true;
}

/** The car at frame `i`: position and heading (radians from east, counter-clockwise). */
function carPose(i) {
  const info = sceneState.info;
  if (!info || !info.ego || !info.ego.pos[i] || !info.ego.quat[i]) return null;
  const p = info.ego.pos[i];
  const q = info.ego.quat[i];
  const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(new THREE.Quaternion(q[0], q[1], q[2], q[3]));
  return { pos: new THREE.Vector3(p[0], p[1], p[2]), heading: Math.atan2(forward.y, forward.x) };
}

/** Follow the car: turning it on far from the car goes there first. */
export function setFollow(on) {
  sceneState.follow = !!on && !!sceneState.info;
  if (els.follow) els.follow.classList.toggle("on", sceneState.follow);
  followFrom = sceneState.info ? sceneState.frame : null;
  if (!sceneState.follow) return false;
  const car = carPose(sceneState.frame);
  if (car && camera.position.distanceTo(car.pos) > FOLLOW_NEAR_M) goToCar();
  return true;
}

/** Move and turn the camera with the car from `followFrom` to frame `index`. */
function followTo(index) {
  const from = followFrom;
  followFrom = index;
  // Under the camera lock the recorded camera drives the view.
  if (!sceneState.follow || rig.on || from === null || from === index) return;
  const a = carPose(from);
  const b = carPose(index);
  if (!a || !b) return;
  let turn = b.heading - a.heading;
  if (turn > Math.PI) turn -= 2 * Math.PI;
  if (turn < -Math.PI) turn += 2 * Math.PI;
  const offset = camera.position.clone().sub(a.pos).applyAxisAngle(_zAxis, turn);
  setCameraView(b.pos.clone().add(offset), view.yaw + turn, view.pitch);
}

function updateLabels() {
  const info = sceneState.info;
  if (!els.label || !info) return;
  const time = info.times && info.times[sceneState.frame];
  const suffix = sceneState.loading ? " …" : "";
  els.label.textContent = `${sceneState.frame} / ${info.frames - 1}`
    + (time === undefined ? "" : `  ${time.toFixed(1)}s`) + suffix;
  els.frame.value = String(sceneState.frame);
}

/** Scene keys; returns true when the event was consumed. */
export function handleSceneKey(event) {
  if (!sceneState.info) return false;
  if (event.key === ",") {
    step(-1);
    return true;
  }
  if (event.key === ".") {
    step(1);
    return true;
  }
  // Shift+P before P: `event.key` is already "P" when shift is held.
  if ((event.key === "p" || event.key === "P") && event.shiftKey) {
    toggleAggregate();
    return true;
  }
  if (event.key === "p" || event.key === "P") {
    togglePoints();
    return true;
  }
  if ((event.key === "k" || event.key === "K") && event.shiftKey) {
    setFollow(!sceneState.follow);
    return true;
  }
  if (event.key === "k" || event.key === "K") {
    goToCar();
    return true;
  }
  if (event.key === "u" || event.key === "U") {
    toggleTint();
    return true;
  }
  return handleRigKey(event);
}

/**
 * Height of the loaded cloud and of the map around a point, for the live
 * alignment check (`viewer/tests/browser/live_scene.mjs`). The cloud's 10th
 * percentile is the road surface: most returns near the car are ground, and
 * the percentile ignores the few that come off the bonnet.
 */
function heightsNear(xy, radius) {
  const out = { ground: { count: 0, p10: null }, map: { count: 0, median: null } };
  const cloud = pointLayer.cloud;
  if (cloud) {
    const zs = [];
    const { positions, origin, count } = cloud;
    for (let i = 0; i < count; i++) {
      const dx = origin[0] + positions[i * 3] - xy[0];
      const dy = origin[1] + positions[i * 3 + 1] - xy[1];
      if (dx * dx + dy * dy <= radius * radius) zs.push(origin[2] + positions[i * 3 + 2]);
    }
    zs.sort((a, b) => a - b);
    out.ground = { count: zs.length, p10: zs.length ? zs[Math.floor(zs.length * 0.1)] : null };
  }
  const zs = [];
  for (const f of store.features.values()) {
    if (f.kind !== "line") continue;
    for (let i = 0; i < f.pos.length; i += 3) {
      const dx = f.pos[i] - xy[0];
      const dy = f.pos[i + 1] - xy[1];
      if (dx * dx + dy * dy <= radius * radius) zs.push(f.pos[i + 2]);
    }
  }
  zs.sort((a, b) => a - b);
  out.map = { count: zs.length, median: zs.length ? zs[Math.floor(zs.length / 2)] : null };
  return out;
}

if (TEST_HOOKS) {
  window.__ll2sceneDebug = {
    cloudZNear: (xy, r) => heightsNear(xy, r).ground,
    mapZNear: (xy, r) => heightsNear(xy, r).map,
  };
  window.__ll2scene = {
    state: () => ({ frame: sceneState.frame, frames: sceneState.info ? sceneState.info.frames : 0,
                    loading: sceneState.loading, playing: sceneState.playing, error: sceneState.error,
                    follow: sceneState.follow,
                    base: sceneState.info ? sceneState.info.base : null }),
    points: () => pointLayer.stats(),
    rig: () => rigState(),
    aggregate: () => ({ ...aggregateState }),
    // `hidden` is what is not drawn, whatever the mode; `ticked` the chips.
    filters: () => ({ ...filterState, ticked: Array.from(filterState.ticked).sort((a, b) => a - b),
                      hidden: Array.from(hiddenClasses()).sort((a, b) => a - b) }),
    hideClasses: (ids) => hideClasses(ids),
    setClassMode: (only) => setClassMode(only),
    classes: () => {
      const info = sceneState.info;
      return {
        available: Boolean(info && info.classes),
        cameras: (info && info.classes && info.classes.cameras) || [],
        inData: pointLayer.hasClasses(),
        rgb: pointLayer.hasRgb(),
        instances: pointLayer.hasInstances() ? pointLayer.instanceCount() : 0,
        car: className(58),
        histogram: pointLayer.classHistogram(),
      };
    },
    showAggregate: (on) => toggleAggregate(on),
    project: (world) => projectToImage(world),
    show: (i) => showFrame(i),
    ego: (i) => (sceneState.info && sceneState.info.ego ? sceneState.info.ego.pos[i] : null),
  };
}
