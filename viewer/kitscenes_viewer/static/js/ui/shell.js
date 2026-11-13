// The frame around the 3D view: side panel and its tabs, collapsible
// sections, drop-down menus, dialogs, tooltips, the breadcrumb, the status
// bar and the remembered interface preferences.
//
// Nothing here knows about the map or the sensors; the modules that do push
// their state in (setCrumbs, setJob, setHints, setTabAvailable).
import { Emitter, TEST_HOOKS } from "../util.js";
import { applyIcons } from "./icons.js";

const PREFS_KEY = "ll2.viewer3d.ui";
const NARROW_PX = 1100;      // below this the panel starts closed
const TOOLTIP_DELAY_MS = 380;

const $ = (id) => document.getElementById(id);

export const ui = new Emitter(); // "panel" (open), "tab" (name), "prefs" (prefs), "resize"

const DEFAULT_PREFS = {
  panelOpen: null,   // null: decided by the window width on first load
  tab: "layers",
  closed: [],        // collapsed section keys
  hints: true,
  debug: false,
  tooltips: true,
  walkExp: 0,        // walk speed = 2^walkExp
  viewNodes: false,  // View-mode clicks select nodes too
  panelW: null,      // dragged panel width in px; null is the stylesheet's
};
const PANEL_MIN_PX = 240;
const PANEL_MAX_PX = 620;

function loadPrefs() {
  try {
    const raw = window.localStorage.getItem(PREFS_KEY);
    return { ...DEFAULT_PREFS, ...(raw ? JSON.parse(raw) : {}) };
  } catch (_) {
    return { ...DEFAULT_PREFS };
  }
}

export const prefs = loadPrefs();

export function savePrefs() {
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch (_) { /* storage blocked: the choice lasts for this page */ }
  ui.emit("prefs", prefs);
}

/** Back to the defaults (the panel follows the window width again). */
export function resetPrefs() {
  Object.assign(prefs, DEFAULT_PREFS, { closed: [] });
  savePrefs();
  setPanelWidth(null);
  for (const sect of document.querySelectorAll(".sect[data-sect]")) sect.classList.remove("closed");
  setPanelOpen(window.innerWidth >= NARROW_PX, false);
  showTab("layers", false);
  setTabAvailable("debug", false);
}

// ---- side panel -------------------------------------------------------------------
const available = { layers: true, inspect: true, scene: false, debug: false };
let activeTab = "layers";

export const panelOpen = () => !document.body.classList.contains("panelClosed");
export const currentTab = () => activeTab;
/** True while the page is the KITScenes viewer, without the JOSM controls (see app.js). */
export const isStandalone = () => document.body.classList.contains("standalone");

export function setPanelOpen(open, remember = true) {
  document.body.classList.toggle("panelClosed", !open);
  $("panelBtn")?.classList.toggle("on", !!open);
  if (remember) {
    prefs.panelOpen = !!open;
    savePrefs();
  }
  ui.emit("panel", !!open);
}

export const togglePanel = () => setPanelOpen(!panelOpen());

/** Show tab `name` (opening the panel). Unavailable tabs are refused. */
export function showTab(name, open = true) {
  if (!available[name]) return false;
  activeTab = name;
  for (const pane of document.querySelectorAll("#sidePanel .pane")) {
    pane.hidden = pane.dataset.pane !== name;
  }
  for (const tab of document.querySelectorAll("#sidePanel .tabs [data-tab]")) {
    tab.classList.toggle("active", tab.dataset.tab === name);
    tab.setAttribute("aria-selected", tab.dataset.tab === name ? "true" : "false");
    if (tab.dataset.tab === name) {
      const badge = tab.querySelector(".badge");
      if (badge) badge.hidden = true;
    }
  }
  if (open && !panelOpen()) setPanelOpen(true);
  if (prefs.tab !== name) {
    prefs.tab = name;
    savePrefs();
  }
  ui.emit("tab", name);
  return true;
}

/**
 * Offer or withdraw a tab. A tab that becomes available while another is
 * showing gets a dot instead of taking over, unless `focus` asks for it.
 */
export function setTabAvailable(name, on, { focus = false } = {}) {
  const was = available[name];
  available[name] = !!on;
  const tab = document.querySelector(`#sidePanel .tabs [data-tab="${name}"]`);
  if (tab) tab.hidden = !on;
  if (!on && activeTab === name) showTab("layers", false);
  if (on && (focus || (!was && prefs.tab === name))) showTab(name, false);
  else if (on && !was && activeTab !== name) {
    const badge = tab && tab.querySelector(".badge");
    if (badge) badge.hidden = false;
  }
}

/** Panel width in px, or null for the default. */
export function setPanelWidth(px) {
  const root = document.documentElement;
  if (px === null || !Number.isFinite(px)) root.style.removeProperty("--panel-w");
  else root.style.setProperty("--panel-w", `${Math.round(Math.max(PANEL_MIN_PX, Math.min(PANEL_MAX_PX, px)))}px`);
}

/** The panel's left edge drags its width; a double click puts it back. */
function installPanelResize() {
  const handle = $("panelResize");
  const body = document.querySelector("#sidePanel .panelBody");
  if (!handle || !body) return;
  let drag = null;
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, w: body.getBoundingClientRect().width };
    handle.classList.add("dragging");
    document.body.classList.add("resizingPanel");
  });
  handle.addEventListener("pointermove", (e) => {
    if (!drag) return;
    setPanelWidth(drag.w + (drag.x - e.clientX));
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    handle.classList.remove("dragging");
    document.body.classList.remove("resizingPanel");
    prefs.panelW = Math.round(body.getBoundingClientRect().width);
    savePrefs();
  };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  handle.addEventListener("dblclick", () => {
    prefs.panelW = null;
    savePrefs();
    setPanelWidth(null);
  });
  if (prefs.panelW) setPanelWidth(prefs.panelW);
}

function installPanel() {
  installPanelResize();
  for (const tab of document.querySelectorAll("#sidePanel .tabs [data-tab]")) {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      // The active tab's button folds the panel away, as in most editors.
      if (name === activeTab && panelOpen()) setPanelOpen(false);
      else showTab(name);
    });
  }
  $("panelBtn")?.addEventListener("click", togglePanel);
  const open = prefs.panelOpen === null ? window.innerWidth >= NARROW_PX : prefs.panelOpen;
  setPanelOpen(open, false);
  setTabAvailable("debug", prefs.debug);
  showTab(available[prefs.tab] ? prefs.tab : "layers", false);

  // Collapsible sections remember their state; a section key is its data-sect.
  const closed = new Set(prefs.closed || []);
  for (const head of document.querySelectorAll(".sect > .sectHead[data-collapse]")) {
    const sect = head.parentElement;
    const key = sect.dataset.sect;
    if (key && closed.has(key)) sect.classList.add("closed");
    head.setAttribute("aria-expanded", sect.classList.contains("closed") ? "false" : "true");
    head.addEventListener("click", () => {
      sect.classList.toggle("closed");
      head.setAttribute("aria-expanded", sect.classList.contains("closed") ? "false" : "true");
      if (!key) return;
      const set = new Set(prefs.closed || []);
      if (sect.classList.contains("closed")) set.add(key);
      else set.delete(key);
      prefs.closed = [...set];
      savePrefs();
    });
  }
}

/** Open the section holding `el` (and its tab), e.g. before focusing a field in it. */
export function reveal(el) {
  if (!el) return;
  const pane = el.closest(".pane");
  if (pane && pane.dataset.pane) showTab(pane.dataset.pane);
  const sect = el.closest(".sect.closed");
  if (sect) sect.querySelector(".sectHead[data-collapse]")?.click();
  el.scrollIntoView({ block: "nearest" });
}

/** A dot on a tab that is not showing: something new there. */
export function setTabBadge(name, on) {
  const tab = document.querySelector(`#sidePanel .tabs [data-tab="${name}"]`);
  const badge = tab && tab.querySelector(".badge");
  if (badge) badge.hidden = !on || (name === activeTab && panelOpen());
}

// ---- menus and popovers ----------------------------------------------------------
let openMenu = null; // { menu, button }

export function closeMenus() {
  if (!openMenu) return;
  openMenu.menu.hidden = true;
  openMenu.button.classList.remove("open");
  openMenu.button.setAttribute("aria-expanded", "false");
  openMenu = null;
}

function toggleMenu(button, menu) {
  const wasOpen = openMenu && openMenu.menu === menu;
  closeMenus();
  closePopover();
  if (wasOpen) return;
  menu.hidden = false;
  button.classList.add("open");
  button.setAttribute("aria-expanded", "true");
  openMenu = { menu, button };
}

let popover = null; // { el, anchor }

export function closePopover() {
  if (!popover) return;
  popover.el.hidden = true;
  popover = null;
}

/** Show popover `el` under `anchor`; `fill()` writes its content first. */
export function togglePopover(el, anchor, fill) {
  const wasOpen = popover && popover.el === el;
  closePopover();
  closeMenus();
  if (wasOpen) return;
  if (fill) fill(el);
  el.hidden = false;
  const r = anchor.getBoundingClientRect();
  el.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - el.offsetWidth - 8))}px`;
  el.style.top = `${r.bottom + 6}px`;
  popover = { el, anchor };
}

function installMenus() {
  for (const button of document.querySelectorAll("[data-menu]")) {
    const menu = $(button.dataset.menu);
    if (!menu) continue;
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleMenu(button, menu);
    });
    // A chosen item closes the menu; the item's own handler still runs.
    menu.addEventListener("click", (e) => {
      if (e.target.closest("button")) closeMenus();
    });
  }
  document.addEventListener("pointerdown", (e) => {
    if (openMenu && !openMenu.menu.contains(e.target) && !openMenu.button.contains(e.target)) closeMenus();
    if (popover && !popover.el.contains(e.target) && !popover.anchor.contains(e.target)) closePopover();
  });
  window.addEventListener("resize", () => { closeMenus(); closePopover(); });
}

// ---- dialogs ------------------------------------------------------------------------
let openModal = null;
let lastFocus = null;

export const modalOpen = () => openModal !== null;

export function showModal(id) {
  const el = $(id);
  if (!el) return;
  if (openModal && openModal !== el) closeModal();
  closeMenus();
  closePopover();
  lastFocus = document.activeElement;
  el.hidden = false;
  openModal = el;
  ui.emit("modal", id);
  const first = el.querySelector("input:not([type=checkbox]):not([type=range]), [data-close]");
  if (first) first.focus();
}

export function closeModal() {
  if (!openModal) return;
  openModal.hidden = true;
  openModal = null;
  if (lastFocus && lastFocus.focus) lastFocus.focus();
  lastFocus = null;
  ui.emit("modal", null);
}

function installModals() {
  for (const modal of document.querySelectorAll(".modal")) {
    modal.addEventListener("pointerdown", (e) => { if (e.target === modal) closeModal(); });
    for (const close of modal.querySelectorAll("[data-close]")) close.addEventListener("click", closeModal);
  }
}

// ---- tooltips -------------------------------------------------------------------------
// `data-tip` is the text, `data-key` the shortcut. Native titles stay for the
// panel's longer explanations.
let tipTimer = null;
let tipFor = null;

function hideTip() {
  clearTimeout(tipTimer);
  tipTimer = null;
  tipFor = null;
  const tip = $("tooltip");
  if (tip) tip.hidden = true;
}

function showTip(el) {
  const tip = $("tooltip");
  if (!tip || !document.body.contains(el)) return;
  tip.textContent = el.dataset.tip;
  if (el.dataset.key) {
    const k = document.createElement("kbd");
    k.textContent = el.dataset.key;
    tip.appendChild(k);
  }
  tip.hidden = false;
  const r = el.getBoundingClientRect();
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  // Below the element, or above it near the bottom; to the side for the rail.
  let x = r.left + r.width / 2 - w / 2;
  let y = r.bottom + 8;
  if (el.closest("#editBar")) {
    x = r.right + 8;
    y = r.top + r.height / 2 - h / 2;
  } else if (el.closest("#sidePanel .tabs, #navGizmo")) {
    x = r.left - w - 8;
    y = r.top + r.height / 2 - h / 2;
  } else if (y + h > window.innerHeight - 4) {
    y = r.top - h - 8;
  }
  tip.style.left = `${Math.max(6, Math.min(x, window.innerWidth - w - 6))}px`;
  tip.style.top = `${Math.max(6, Math.min(y, window.innerHeight - h - 6))}px`;
}

function installTooltips() {
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (el === tipFor) return;
    hideTip();
    if (!el || !prefs.tooltips || e.buttons) return;
    tipFor = el;
    tipTimer = setTimeout(() => showTip(el), TOOLTIP_DELAY_MS);
  });
  document.addEventListener("pointerdown", hideTip, true);
  document.addEventListener("keydown", hideTip, true);
  window.addEventListener("blur", hideTip);
}

// ---- breadcrumb, jobs, hints -----------------------------------------------------------
/** `city` / `scene`: text or null; `cityClickable` enables the city crumb. */
export function setCrumbs({ city, scene, sceneTitle } = {}) {
  const crumbs = $("crumbs");
  if (!crumbs) return;
  for (const el of crumbs.querySelectorAll('[data-crumb="city"]')) el.hidden = !city;
  for (const el of crumbs.querySelectorAll('[data-crumb="scene"]')) el.hidden = !scene;
  if (city) $("crumbCity").textContent = city;
  if (scene) {
    $("crumbSceneName").textContent = scene;
    $("crumbScene").title = sceneTitle || "";
  }
  $("cityDeselect")?.classList.toggle("current", !scene);
  crumbs.hidden = !city && !scene;
}

const jobs = new Map();

/** A long-running job for the status bar; `text` null ends it. */
export function setJob(key, text) {
  if (text) jobs.set(key, text);
  else jobs.delete(key);
  const box = $("jobs");
  if (!box) return;
  box.hidden = jobs.size === 0;
  $("jobsText").textContent = [...jobs.values()].join(" · ");
}

let hintsHtml = "";

/** Status-bar hints: a list of [keys, what] pairs, or a plain string. */
export function setHints(items) {
  const el = $("hints");
  if (!el) return;
  const html = !prefs.hints ? "" : typeof items === "string" ? escapeHtml(items)
    : items.map(([k, what]) => `<b>${escapeHtml(k)}</b> ${escapeHtml(what)}`).join('<span class="dot">·</span>');
  if (html === hintsHtml) return;
  hintsHtml = html;
  el.innerHTML = html;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- keys -------------------------------------------------------------------------------
const isTyping = (el) => !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
  || el.tagName === "SELECT" || el.isContentEditable);

const keyHandlers = []; // (event) => boolean, from the palette and the help sheet

export function onUiKey(fn) {
  keyHandlers.push(fn);
}

/**
 * Keys the interface owns: Escape for menus and dialogs, and the window
 * shortcuts. While a dialog is open the map gets no keys at all. Returns
 * true when the event was consumed.
 */
export function handleUiKey(e) {
  if (openModal) {
    if (e.key === "Escape") {
      e.preventDefault();
      closeModal();
      return true;
    }
    for (const fn of keyHandlers) if (fn(e)) return true;
    return !isTyping(e.target);
  }
  if (e.key === "Escape" && (openMenu || popover)) {
    closeMenus();
    closePopover();
    return true;
  }
  for (const fn of keyHandlers) if (fn(e)) return true;
  if (isTyping(e.target)) return false;
  if (e.key === "\\" && !e.ctrlKey && !e.metaKey && !e.altKey && !e.repeat) {
    togglePanel();
    return true;
  }
  return false;
}

// ---- viewport size ------------------------------------------------------------------------
/** Calls `fn(width, height)` whenever the 3D view's box changes size. */
export function observeViewport(fn) {
  const box = $("app");
  if (!box) return;
  let last = "";
  const run = () => {
    const w = Math.max(1, box.clientWidth);
    const h = Math.max(1, box.clientHeight);
    const key = `${w}x${h}`;
    if (key === last) return;
    last = key;
    fn(w, h);
    ui.emit("resize", { w, h });
  };
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(run).observe(box);
  window.addEventListener("resize", run);
  run();
}

export function installShell() {
  applyIcons();
  installPanel();
  installMenus();
  installModals();
  installTooltips();
}

if (TEST_HOOKS) {
  window.__ll2ui = {
    state: () => ({
      panelOpen: panelOpen(), tab: activeTab, available: { ...available },
      modal: openModal ? openModal.id : null, menu: openMenu ? openMenu.menu.id : null,
      popover: popover ? popover.el.id : null, hints: $("hints") ? $("hints").textContent : "",
      jobs: [...jobs.values()], prefs: { ...prefs },
    }),
    setPanelOpen: (open) => setPanelOpen(open),
    showTab: (name) => showTab(name),
  };
}
