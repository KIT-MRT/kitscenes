// Every action the viewer has, by name: the command palette (Ctrl+K) runs
// them and the keyboard sheet (?) lists them. Commands click the same
// buttons the panels have, so there is one code path per action and the
// buttons' own state stays right.
import { TEST_HOOKS } from "../util.js";
import { edit, editBlockedReason } from "../edit.js";
import { sceneState } from "../sensors/scene.js";
import { cityState } from "../sensors/city.js";
import { rig, handleRigKey } from "../sensors/camrig.js";
import { ortho } from "../camera.js";
import { tileLayer } from "../render/tiles.js";
import { laneletLayer } from "../render/lanelets.js";
import { lineLayer } from "../render/lines.js";
import { iconLayer } from "../render/icons.js";
import { pointLayer } from "../render/points.js";
import { trajectoryLayer } from "../render/trajectories.js";
import { PROVIDERS, openImagery } from "../imagery.js";
import { showModal, closeModal, modalOpen, onUiKey, togglePanel, showTab, reveal, prefs, isStandalone,
         togglePopover, escapeHtml } from "./shell.js";
import { faceNorth } from "./navgizmo.js";

const $ = (id) => document.getElementById(id);
const click = (id) => () => $(id)?.click();
const onOff = (on) => (on ? "on" : "off");
const hasScene = () => !!sceneState.info;
const hasCity = () => cityState.scenes.length > 0;
const hasCameras = () => !!(sceneState.info && Object.keys(sceneState.info.cameras || {}).length);

/**
 * { group, title, keys, run, when?, state? }. `keys` is what the help sheet
 * shows; the keyboard handling itself lives with each feature.
 */
export const COMMANDS = [
  // View
  { group: "View", title: "Frame the whole map", keys: ["F"], run: click("frameBtn") },
  { group: "View", title: "Top view, north up", keys: ["B"], run: click("bevBtn") },
  { group: "View", title: "Face north", run: () => faceNorth(), when: () => !rig.on },
  { group: "View", title: "Flat map view (2D)", keys: ["V"], run: click("orthoBtn"), state: () => onOff(ortho.on) },
  { group: "View", title: "FPS mouse look", run: click("fpsBtn"), state: () => onOff($("fpsBtn")?.classList.contains("on")) },
  { group: "View", title: "Centre JOSM on this view", run: click("recenterJosmBtn"), when: () => !isStandalone() },
  { group: "View", title: "Default BEV at the origin", run: click("defaultBevBtn") },
  ...Object.entries(PROVIDERS).map(([key, p]) => (
    { group: "View", title: `Open in ${p.label}`, run: () => openImagery(key) })),

  // Layers
  { group: "Layers", title: "Lanelet surfaces and arrows", keys: ["L"], run: click("lanesBtn"), state: () => onOff(laneletLayer.visible) },
  { group: "Layers", title: "Signs, lights and markings", keys: ["N"], run: click("iconsBtn"), state: () => onOff(iconLayer.visible) },
  { group: "Layers", title: "Styled lines (real widths, dashes)", run: () => (lineLayer.styled ? click("lineSimpleBtn") : click("lineStyledBtn"))(),
    state: () => onOff(lineLayer.styled) },
  { group: "Layers", title: "Curbs, rails and walls in 3D", run: click("raisedBtn"), when: () => lineLayer.styled,
    state: () => onOff(lineLayer.raised) },
  { group: "Layers", title: "Map tiles", run: click("tilesBtn"), state: () => onOff(tileLayer.on) },
  { group: "Layers", title: "Tile source…", run: () => {
    const panel = $("tilePanel");
    if (panel && panel.hidden) $("tilesMenuBtn")?.click();
    reveal(panel);
  } },

  // Edit
  { group: "Edit", title: "Edit mode", keys: ["E"], run: click("editBtn"), state: () => onOff(edit.on),
    when: () => edit.on || !editBlockedReason() },
  { group: "Edit", title: "Move tool", keys: ["G"], run: click_edit("translate"), when: () => edit.on },
  { group: "Edit", title: "Rotate tool", keys: ["R"], run: click_edit("rotate"), when: () => edit.on },
  { group: "Edit", title: "Height only", keys: ["H"], run: click_edit("height"), when: () => edit.on, state: () => onOff(edit.heightOnly) },
  { group: "Edit", title: "Interpolate heights along the way", keys: ["I"], run: click_edit("interpolate"), when: () => edit.on },
  { group: "Edit", title: "Snap", keys: ["M"], run: click_edit("snap"), when: () => edit.on, state: () => onOff(edit.snap) },
  { group: "Edit", title: "Keys move the selection", keys: ["T"], run: click_edit("keys"), when: () => edit.on,
    state: () => onOff(edit.keysMove === "selection") },
  { group: "Edit", title: "Type a height", keys: ["Z"], run: () => $("zField")?.focus(), when: () => edit.on && !$("zField")?.disabled },
  { group: "Edit", title: "Delete the selection", keys: ["Del"], run: click_edit("delete"), when: () => edit.on },
  { group: "Edit", title: "Undo", keys: ["Ctrl Z"], run: click_edit("undo"), when: () => edit.on },
  { group: "Edit", title: "Redo", keys: ["Ctrl Y", "Ctrl Shift Z"], run: click_edit("redo"), when: () => edit.on },

  // Scene
  { group: "Scene", title: "Play / pause", run: click("scenePlay"), when: hasScene, state: () => onOff(sceneState.playing) },
  { group: "Scene", title: "Next frame", keys: ["."], run: click("sceneNext"), when: hasScene },
  { group: "Scene", title: "Previous frame", keys: [","], run: click("scenePrev"), when: hasScene },
  { group: "Scene", title: "Go to the car", keys: ["K"], run: click("sceneCar"), when: hasScene },
  { group: "Scene", title: "Follow the car", keys: ["Shift K"], run: click("sceneFollow"), when: hasScene, state: () => onOff(sceneState.follow) },
  { group: "Scene", title: "Point cloud of this frame", keys: ["P"], run: click("scenePoints"), when: hasScene, state: () => onOff(pointLayer.visible) },
  { group: "Scene", title: "Whole scene as one cloud", keys: ["Shift P"], run: click("sceneAggregate"), when: hasScene,
    state: () => onOff($("sceneAggregate")?.classList.contains("on")) },
  { group: "Scene", title: "Tint this frame over the whole scene", keys: ["U"], run: click("sceneTint"), when: hasScene,
    state: () => onOff(pointLayer.tintFrame) },
  { group: "Scene", title: "Look through the ring camera", keys: ["Shift L"], run: click("sceneLock"), when: hasCameras, state: () => onOff(rig.on) },
  { group: "Scene", title: "Next ring camera", keys: ["]"], run: () => handleRigKey({ key: "]" }), when: () => rig.on },
  { group: "Scene", title: "Previous ring camera", keys: ["["], run: () => handleRigKey({ key: "[" }), when: () => rig.on },
  { group: "Scene", title: "Recorded image over the render", keys: ["O"], run: click("sceneOverlay"), when: () => rig.on, state: () => onOff(rig.overlay) },
  { group: "Scene", title: "Release the camera", keys: ["Esc"], run: () => handleRigKey({ key: "Escape" }), when: () => rig.on },
  { group: "Scene", title: "Point filters…", run: () => {
    if ($("sceneFilters")?.hidden) $("sceneFilterBtn")?.click();
    reveal($("sceneFilters"));
  }, when: () => !$("sceneBar")?.hidden },

  // City
  { group: "City", title: "Load point clouds around the camera", run: click("cityLoad"), when: hasCity },
  { group: "City", title: "Load clouds automatically", run: click("cityAuto"), when: hasCity, state: () => onOff(cityState.auto) },
  { group: "City", title: "Unload city clouds", run: click("cityUnload"), when: hasCity },
  { group: "City", title: "Trajectories", run: click("cityTracks"), when: hasCity, state: () => onOff(trajectoryLayer.visible) },
  { group: "City", title: "Back to the city (close the scene)", run: click("cityDeselect"), when: () => hasCity() && hasScene() },
  { group: "City", title: "Find a scene…", run: () => { reveal($("sceneSearch")); $("sceneSearch")?.focus(); }, when: hasCity },

  // Window
  { group: "Window", title: "Side panel", keys: ["\\"], run: () => togglePanel() },
  { group: "Window", title: "Show layers", run: () => showTab("layers") },
  { group: "Window", title: "Show the selection's tags", run: () => showTab("inspect") },
  { group: "Window", title: "Show sensor data", run: () => showTab("scene"), when: () => hasScene() || hasCity() },
  { group: "Window", title: "Show debug readouts", run: () => showTab("debug"), when: () => prefs.debug },
  { group: "Window", title: "Connection and map source", run: () => $("connBtn")?.click() },
  { group: "Window", title: "Keyboard and mouse", keys: ["?"], run: () => openHelp() },
  { group: "Window", title: "Settings", run: () => showModal("settingsDlg") },
  { group: "Window", title: "Search commands", keys: ["Ctrl K"], run: () => openPalette(), palette: false },
];

function click_edit(name) {
  return () => document.querySelector(`#editBar [data-edit="${name}"]`)?.click();
}

const enabled = (c) => !c.when || c.when();

// ---- palette ------------------------------------------------------------------------
let matches = [];
let active = 0;

/** Every word of the query must start a word of the title (or group), in any order. */
function score(cmd, query) {
  if (!query) return 1;
  const hay = `${cmd.title} ${cmd.group}`.toLowerCase();
  const words = hay.split(/[^a-z0-9]+/).filter(Boolean);
  let s = 0;
  for (const q of query.toLowerCase().split(/\s+/).filter(Boolean)) {
    const at = words.findIndex((w) => w.startsWith(q));
    if (at >= 0) s += 3 - Math.min(2, at);
    else if (hay.includes(q)) s += 1;
    else return 0;
  }
  return s;
}

function renderPalette() {
  const list = $("paletteList");
  const q = $("paletteInput").value.trim();
  matches = COMMANDS.filter((c) => c.palette !== false)
    .map((c, i) => ({ c, s: score(c, q), i, ok: enabled(c) }))
    .filter((m) => m.s > 0)
    .sort((a, b) => (b.ok - a.ok) || (b.s - a.s) || (a.i - b.i));
  active = Math.min(active, Math.max(0, matches.length - 1));
  if (matches[active] && !matches[active].ok) active = 0;
  list.textContent = "";
  if (!matches.length) {
    const empty = document.createElement("div");
    empty.className = "palEmpty";
    empty.textContent = "No command by that name.";
    list.appendChild(empty);
    return;
  }
  matches.forEach((m, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `palItem${i === active ? " active" : ""}${m.ok ? "" : " disabled"}`;
    row.setAttribute("role", "option");
    row.dataset.index = String(i);
    const st = m.ok && m.c.state ? m.c.state() : null;
    row.innerHTML = `<span class="grp">${escapeHtml(m.c.group)}</span><span class="ttl">${escapeHtml(m.c.title)}</span>`
      + (st ? `<span class="state">${st}</span>` : "")
      + (m.c.keys ? m.c.keys.slice(0, 1).map((k) => `<kbd>${escapeHtml(k)}</kbd>`).join("") : "");
    if (!m.ok) row.title = "Not available right now";
    list.appendChild(row);
  });
  list.querySelector(".active")?.scrollIntoView({ block: "nearest" });
}

export function openPalette() {
  if (modalOpen() && !$("palette").hidden) return;
  $("paletteInput").value = "";
  active = 0;
  showModal("palette");
  renderPalette();
}

function runMatch(i) {
  const m = matches[i];
  if (!m || !m.ok) return false;
  closeModal();
  m.c.run();
  return true;
}

// ---- help sheet -----------------------------------------------------------------------
const MOUSE = [
  ["Navigate", [
    [["Left-drag"], "Orbit the point under the cursor (over the sky: turn)"],
    [["Right-drag"], "Look around"],
    [["Middle-drag", "Shift drag"], "Pan"],
    [["Wheel"], "Move toward the cursor"],
    [["W A S D", "Arrows"], "Walk"],
    [["Space", "C"], "Up / down"],
    [["Alt ← →"], "Turn"],
    [["Ctrl arrows"], "Look"],
    [["Shift"], "Ten times faster"],
    [["2D view"], "Left-drag pans, right-drag turns"],
  ]],
  ["Select (View and Edit)", [
    [["Click"], "Edit: the node under the cursor, else the way. View: the way, else the lane (nodes too with the setting). Tags in the Selection tab"],
    [["Shift click"], "Add or remove"],
    [["Ctrl drag"], "Box select, Edit mode (Shift: add)"],
    [["Middle-click"], "Cycle through everything under the cursor"],
    [["Esc"], "Deselect"],
  ]],
  ["City", [
    [["Click a track"], "Open that scene at that frame"],
  ]],
];

function helpRow(keys, what) {
  return `<div class="helpRow"><span class="what">${escapeHtml(what)}</span><span class="keys">`
    + keys.map((k) => `<kbd>${escapeHtml(k)}</kbd>`).join("") + "</span></div>";
}

export function openHelp() {
  const groups = [];
  for (const [name, rows] of MOUSE) {
    groups.push(`<div class="helpGroup"><h3>${escapeHtml(name)}</h3>${rows.map(([k, w]) => helpRow(k, w)).join("")}</div>`);
  }
  const byGroup = new Map();
  for (const c of COMMANDS) {
    if (!c.keys || (c.group === "Edit" && isStandalone())) continue;
    if (!byGroup.has(c.group)) byGroup.set(c.group, []);
    byGroup.get(c.group).push(helpRow(c.keys, c.title));
  }
  for (const [name, rows] of byGroup) {
    groups.push(`<div class="helpGroup"><h3>${escapeHtml(name)}</h3>${rows.join("")}</div>`);
  }
  $("helpBody").innerHTML = groups.join("");
  showModal("helpSheet");
}

// ---- connection popover ----------------------------------------------------------------
let connInfo = () => [];

/** `fn()` returns [label, value, cls?] rows for the connection popover. */
export function setConnectionInfo(fn) {
  connInfo = fn;
}

function fillConnection(el) {
  el.innerHTML = "<h3>Connection</h3><dl>"
    + connInfo().map(([k, v, cls]) => `<dt>${escapeHtml(k)}</dt><dd class="${cls || ""}">${escapeHtml(v)}</dd>`).join("")
    + "</dl>";
}

export function installCommands() {
  $("paletteBtn")?.addEventListener("click", () => openPalette());
  $("helpBtn")?.addEventListener("click", () => openHelp());
  $("settingsBtn")?.addEventListener("click", () => showModal("settingsDlg"));
  $("connBtn")?.addEventListener("click", () => {
    const pop = $("connPop");
    togglePopover(pop, $("connBtn"), fillConnection);
  });
  $("paletteInput")?.addEventListener("input", () => {
    active = 0;
    renderPalette();
  });
  $("paletteList")?.addEventListener("click", (e) => {
    const row = e.target.closest("[data-index]");
    if (row) runMatch(Number(row.dataset.index));
  });
  onUiKey((e) => {
    const paletteOpen = !$("palette").hidden;
    if ((e.key === "k" || e.key === "K") && (e.ctrlKey || e.metaKey) && !e.altKey) {
      e.preventDefault();
      if (paletteOpen) closeModal();
      else openPalette();
      return true;
    }
    if (paletteOpen) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const n = matches.length;
        if (!n) return true;
        do active = (active + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
        while (!matches[active].ok && matches.some((m) => m.ok));
        renderPalette();
        return true;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        runMatch(active);
        return true;
      }
      return false;
    }
    if (modalOpen()) return false;
    const t = e.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
    if (e.key === "?" && !typing && !e.ctrlKey && !e.metaKey) {
      openHelp();
      return true;
    }
    return false;
  });
}

if (TEST_HOOKS) {
  window.__ll2commands = {
    list: () => COMMANDS.map((c) => ({ group: c.group, title: c.title, keys: c.keys || [], enabled: enabled(c),
                                       state: enabled(c) && c.state ? c.state() : null })),
    run: (title) => {
      const c = COMMANDS.find((x) => x.title === title);
      if (!c || !enabled(c)) return false;
      c.run();
      return true;
    },
  };
}
