// The Selection tab: every tag of the selected nodes, ways and relations, the
// relations they belong to (which lanelet uses this bound, as which side; the
// lane's traffic lights and signs, which belong to the lanelet, "via" it) and
// a relation's members. Every relation and member row selects that object.
//
// The map stream carries only type/subtype, so the full tags are asked for
// on demand with a read-only `describe` command that JOSM answers. Asked
// only while the tab is showing; asked again when the selection changes,
// when a selected way changes in the stream, when the tab opens and when the
// window gets the focus back (tags edited in JOSM do not reach the stream).
// Without JOSM (the scene server's read-only map) the tab falls back to the
// streamed tags and says so.
//
// With JOSM's map in Edit mode a value can be changed in place: click it,
// Enter (or leaving the field) applies, Esc cancels. That is a `set_tag` per
// object, applied by JOSM as one undo step; on several objects it changes
// the ones that carry the key. An empty value is refused: in JOSM it would
// delete the tag, and that is not what clearing a field should do.
//
// A relation's members go to the highlight, which is how a selected traffic
// light rule shows its lights and stop line on the map.
import { TEST_HOOKS } from "../util.js";
import { store } from "../store.js";
import { selection } from "../selection.js";
import { edit, editEvents, inspectedTokens } from "../edit.js";
import { sendCommand } from "../net.js";
import { toast } from "../hud.js";
import { setRelationMembers } from "../render/highlight.js";
import { ui, currentTab, panelOpen, escapeHtml, setTabBadge } from "./shell.js";

const DEBOUNCE_MS = 150;
const REPLY_TIMEOUT_MS = 4000;
const LIST_MAX = 60;          // objects listed by id
const FIRST_KEYS = ["type", "subtype"];

const $ = (id) => document.getElementById(id);

const state = {
  tokens: [],        // what is described
  key: "",           // tokens joined, to tell a new selection from the same one
  details: new Map(),// token -> { tags, memberships, nodes, missing }
  source: null,      // "josm" | "stream"
  note: "",          // why the streamed tags are all there is
  loading: false,
  stale: true,       // needs asking once the tab shows
  editing: null,     // { key, draft, caret, selection } while a value is being typed
  saving: false,     // a set_tag is on its way to JOSM
};
let rendering = false; // a blur while the body is rebuilt is not the user leaving the field
let timer = null;
let generation = 0;

const visible = () => panelOpen() && currentTab() === "inspect";

function schedule(force = false) {
  if (force) state.stale = true;
  const tokens = inspectedTokens();
  const key = tokens.join(",");
  if (key !== state.key) state.stale = true;
  if (!visible()) {
    // Nothing is asked while the tab is hidden; a dot says there is news.
    if (key !== state.key && tokens.length) setTabBadge("inspect", true);
    return;
  }
  clearTimeout(timer);
  timer = setTimeout(() => refresh(), DEBOUNCE_MS);
}

/** Ask JOSM about the current selection (or fall back to the stream). */
export function refresh() {
  clearTimeout(timer);
  const tokens = inspectedTokens();
  state.tokens = tokens;
  state.key = tokens.join(",");
  state.stale = false;
  const gen = ++generation;
  if (!tokens.length) {
    state.details = new Map();
    state.loading = false;
    state.editing = null;
    setRelationMembers(new Map());
    render();
    return Promise.resolve(null);
  }
  if (state.editing && state.key !== state.editing.selection) state.editing = null;
  state.loading = true;
  render();
  return sendCommand([{ op: "describe", ids: tokens }], { awaitResult: true, timeoutMs: REPLY_TIMEOUT_MS })
    .then((info) => {
      if (gen !== generation) return null;  // a newer selection won
      state.loading = false;
      const r = info && info.result;
      if (r && r.ok && Array.isArray(r.details)) {
        state.details = new Map(r.details.map((d) => [d.id, d]));
        state.source = "josm";
        state.note = "";
      } else {
        fromStream();
        state.note = !info || info.error ? "The viewer server did not answer."
          : !info.delivered ? "JOSM is not connected, so only type and subtype are known."
          : r && r.timeout ? "JOSM did not answer in time; showing type and subtype only."
          : "This map comes from the scene server, which sends type and subtype only.";
      }
      shareMembers();
      render();
      return state.details;
    });
}

/** Tell the highlight which members the described relations have. */
function shareMembers() {
  const members = new Map();
  for (const [token, d] of state.details) {
    if (token.startsWith("relation/") && Array.isArray(d.members)) members.set(token, d.members.map((m) => m.id));
  }
  setRelationMembers(members);
}

// ---- editing values ------------------------------------------------------------------
/** Values can be changed: JOSM's map, its answer, Edit mode. */
const canEdit = () => edit.on && state.source === "josm" && store.mapSource !== "scene_server";

function startEdit(key) {
  if (!canEdit() || state.loading || state.saving) return;
  const row = mergedTags(currentInfos()).find((r) => r.key === key);
  if (!row) return;
  const draft = row.same ? [...row.values.keys()][0] : "";
  // `selection`: a new selection ends the edit, a re-ask of the same one does not.
  state.editing = { key, draft, caret: draft.length, selection: state.key };
  render();
}

function cancelEdit() {
  if (!state.editing) return;
  state.editing = null;
  render();
}

/** Send the typed value to JOSM, for every described object that has the key. */
function commitEdit() {
  const e = state.editing;
  if (!e) return Promise.resolve(null);
  state.editing = null;
  const value = e.draft.trim();
  const infos = currentInfos();
  const row = mergedTags(infos).find((r) => r.key === e.key);
  if (!value) {
    if (row && row.same) toast(`An empty value would delete ${e.key} in JOSM; remove tags there`, "warn", 4000);
    render();
    return Promise.resolve(null);
  }
  const targets = infos.filter((info) => info.tags && e.key in info.tags && info.tags[e.key] !== value);
  if (!targets.length) {
    render();
    return Promise.resolve(null);
  }
  const ops = targets.map((info) => ({ op: "set_tag", id: info.id, key: e.key, value }));
  state.saving = true;
  render();
  return sendCommand(ops, { awaitResult: true }).then((info) => {
    state.saving = false;
    const r = info.result;
    if (!info.delivered) toast(`${e.key}: JOSM is not connected`, "error");
    else if (!r || r.timeout) toast(`${e.key}: JOSM did not answer`, "warn");
    else if (r.ok === false) toast(`${e.key}: ${r.message}`, "warn", 5000);
    else {
      for (const t of targets) t.tags[e.key] = value;  // until JOSM's own answer replaces it
      toast(`${e.key}=${value} on ${plural(targets.length, "object")}`, "info", 2000);
    }
    render();
    return refresh();
  });
}

const currentInfos = () => state.tokens.map((t) => state.details.get(t)).filter(Boolean);

/** What the stream knows: a way's type/subtype, a node's height, a lanelet's bounds. */
function fromStream() {
  state.source = "stream";
  state.details = new Map(state.tokens.map((t) => {
    const f = store.features.get(t);
    const info = { id: t, tags: f && f.tags ? { ...f.tags } : {}, memberships: [],
                   nodes: f && f.kind === "line" ? f.pos.length / 3 : undefined, missing: !f && t.startsWith("way/") };
    if (f && f.kind === "lanelet") {
      info.tags.type ??= "lanelet";
      info.members = [["left", f.lanelet.left], ["right", f.lanelet.right]].map(([role, id]) => {
        const w = store.features.get(id);
        return { id, role, tags: w && w.tags ? { ...w.tags } : {} };
      });
    }
    return [t, info];
  }));
  shareMembers();
}

// ---- rendering -------------------------------------------------------------------
const kindOf = (token) => token.slice(0, token.indexOf("/"));
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const sortKeys = (keys) => [...keys].sort((a, b) => {
  const ia = FIRST_KEYS.indexOf(a);
  const ib = FIRST_KEYS.indexOf(b);
  if (ia >= 0 || ib >= 0) return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  return a < b ? -1 : a > b ? 1 : 0;
});

/** Length and height range of a way, from the streamed geometry. */
function wayGeometry(token) {
  const f = store.features.get(token);
  if (!f || f.kind !== "line") return null;
  let len = 0;
  let zMin = Infinity;
  let zMax = -Infinity;
  const p = f.pos;
  for (let i = 0; i < p.length; i += 3) {
    zMin = Math.min(zMin, p[i + 2]);
    zMax = Math.max(zMax, p[i + 2]);
    if (i) len += Math.hypot(p[i] - p[i - 3], p[i + 1] - p[i - 2], p[i + 2] - p[i - 1]);
  }
  return { len, zMin, zMax };
}

/** One table over every described object: a value, or "different". */
function mergedTags(infos) {
  const keys = new Map(); // key -> Map(value -> count)
  for (const info of infos) {
    for (const [k, v] of Object.entries(info.tags || {})) {
      if (!keys.has(k)) keys.set(k, new Map());
      const values = keys.get(k);
      values.set(v, (values.get(v) || 0) + 1);
    }
  }
  return sortKeys(keys.keys()).map((k) => {
    const values = keys.get(k);
    const count = [...values.values()].reduce((a, b) => a + b, 0);
    return { key: k, values, count, same: values.size === 1 && count === infos.length };
  });
}

/**
 * Relations, merged over the objects: relation -> roles and how many use it.
 * Direct memberships first, then those through another relation (`via`).
 */
function mergedMemberships(infos) {
  const rel = new Map();
  for (const info of infos) {
    for (const m of info.memberships || []) {
      const key = `${m.id}|${m.via || ""}`;
      if (!rel.has(key)) rel.set(key, { id: m.id, via: m.via || "", tags: m.tags || {}, roles: new Map() });
      const r = rel.get(key);
      r.roles.set(m.role || "", (r.roles.get(m.role || "") || 0) + 1);
    }
  }
  const order = (r) => `${r.via ? 1 : 0}${r.tags.type || ""}${r.id}${r.via}`;
  return [...rel.values()].sort((a, b) => order(a).localeCompare(order(b)));
}

// The token a row selects, as a selection item (relations not in the stream
// are still selectable: the tab describes them).
const itemOf = (t) => {
  const kind = kindOf(t);
  return { type: kind, id: kind === "node" ? Number(t.slice(5)) : t };
};

// Long lanelet2 words ("regulatory_element") may wrap after an underscore.
const breakable = (text) => escapeHtml(text).replace(/_/g, "_<wbr>");

function memberRow(id, roleText, tags, extraNote = "") {
  const extra = Object.entries(tags || {}).filter(([k]) => k !== "type" && k !== "subtype")
    .map(([k, v]) => `${k}=${v}`).join("\n");
  const tip = [extra, `Select ${id}`].filter(Boolean).join("\n");
  return `<button type="button" class="memberRow" data-token="${escapeHtml(id)}" title="${escapeHtml(tip)}">`
    + `<span class="role">${breakable(roleText)}</span>`
    + `<span>${breakable(describeRelation(tags || {}))}${extraNote}</span>`
    + `<span class="muted inspectId">${escapeHtml(id)}</span></button>`;
}

function describeRelation(tags) {
  // A rule reads as what it is: "traffic_light", not "regulatory_element traffic_light".
  if (tags.type === "regulatory_element" && tags.subtype) return tags.subtype;
  const t = tags.type || "relation";
  const st = tags.subtype ? ` ${tags.subtype}` : "";
  return `${t}${st}`;
}

function render() {
  const body = $("inspectBody");
  if (!body) return;
  const tokens = state.tokens;
  const ways = tokens.filter((t) => kindOf(t) === "way").length;
  const rels = tokens.filter((t) => kindOf(t) === "relation").length;
  const nodes = tokens.length - ways - rels;
  $("inspectSummary").textContent = !tokens.length ? "Nothing selected"
    : [nodes ? plural(nodes, "node") : null, ways ? plural(ways, "way") : null,
       rels ? plural(rels, "relation") : null].filter(Boolean).join(" · ");
  $("inspectCopy").disabled = !tokens.length || state.loading;
  $("inspectRefresh").disabled = !tokens.length || state.loading;
  if (!tokens.length) {
    body.innerHTML = `<div class="empty">Click a line or a lane to see its tags (nodes too in Edit mode), or select something in JOSM.</div>`;
    return;
  }
  const infos = tokens.map((t) => state.details.get(t)).filter(Boolean);
  const parts = [];

  // Which objects: one header for a single object, a list for several.
  if (tokens.length === 1) {
    const t = tokens[0];
    const info = state.details.get(t);
    const geo = kindOf(t) === "way" ? wayGeometry(t) : null;
    const facts = [];
    if (info && info.nodes !== undefined) facts.push(plural(info.nodes, "node"));
    if (info && Array.isArray(info.members)) facts.push(plural(info.members.length, "member"));
    if (geo) {
      facts.push(`${geo.len.toFixed(1)} m`);
      facts.push(geo.zMax - geo.zMin < 0.005 ? `z ${geo.zMin.toFixed(2)} m` : `z ${geo.zMin.toFixed(2)} … ${geo.zMax.toFixed(2)} m`);
    } else if (kindOf(t) === "node") {
      const rec = store.node(Number(t.slice(5)));
      if (rec) facts.push(`z ${rec.z.toFixed(2)} m`);
    }
    parts.push(`<div class="inspectHead"><span class="inspectId">${escapeHtml(t)}</span>`
      + `<span class="muted">${escapeHtml(facts.join(" · "))}</span></div>`);
    if (info && info.missing) parts.push(`<div class="hint warnText">No longer in JOSM's data.</div>`);
  } else {
    const items = tokens.slice(0, LIST_MAX).map((t) => {
      const info = state.details.get(t);
      const tags = info && info.tags ? info.tags : {};
      const what = [tags.type, tags.subtype].filter(Boolean).join(" ");
      return `<button type="button" class="inspectItem" data-token="${escapeHtml(t)}" title="Select only this one">`
        + `<span class="inspectId">${escapeHtml(t)}</span><span class="muted">${escapeHtml(what)}</span></button>`;
    });
    if (tokens.length > LIST_MAX) items.push(`<div class="hint">and ${tokens.length - LIST_MAX} more</div>`);
    parts.push(`<div class="inspectList">${items.join("")}</div>`);
  }

  // Tags.
  const rows = mergedTags(infos);
  if (state.editing && !canEdit()) state.editing = null;
  const editable = canEdit() && !state.loading && !state.saving;
  const busy = state.loading || state.saving;
  parts.push(`<div class="subHead">Tags${busy ? ' <span class="spinner"></span>' : ""}</div>`);
  if (!rows.length) {
    parts.push(`<div class="hint">${state.loading ? "Asking JOSM…" : "No tags."}</div>`);
  } else {
    parts.push('<table class="tagTable"><tbody>');
    for (const r of rows) {
      let value;
      if (r.same) value = escapeHtml([...r.values.keys()][0]);
      else {
        const list = [...r.values.entries()].map(([v, n]) => `${v} (${n})`).join("\n");
        const firstOnly = r.values.size === 1 ? escapeHtml([...r.values.keys()][0]) : `${r.values.size} values`;
        value = `<span class="different" title="${escapeHtml(list)}">${firstOnly}</span>`
          + (r.count < infos.length ? ` <span class="muted">on ${r.count} of ${infos.length}</span>` : "");
      }
      if (state.editing && state.editing.key === r.key) {
        const hint = r.same ? "" : `${r.values.size} values: one for all ${r.count}`;
        value = `<input type="text" class="tagInput" data-key="${escapeHtml(r.key)}" spellcheck="false"`
          + ` value="${escapeHtml(state.editing.draft)}" placeholder="${escapeHtml(hint)}"`
          + ` aria-label="Value of ${escapeHtml(r.key)}">`;
        parts.push(`<tr><th>${escapeHtml(r.key)}</th><td class="tagValue editing">${value}</td></tr>`);
      } else if (editable) {
        const on = r.count < infos.length ? ` (on the ${r.count} that have it)` : "";
        parts.push(`<tr><th>${escapeHtml(r.key)}</th><td class="tagValue editable" data-key="${escapeHtml(r.key)}"`
          + ` title="Click to change${on}. Enter applies, Esc cancels">${value}</td></tr>`);
      } else {
        parts.push(`<tr><th>${escapeHtml(r.key)}</th><td>${value}</td></tr>`);
      }
    }
    parts.push("</tbody></table>");
  }

  // Relations.
  const memberOf = mergedMemberships(infos);
  if (memberOf.length || state.source === "josm") {
    parts.push('<div class="subHead">Member of</div>');
    if (!memberOf.length) parts.push('<div class="hint">No relation.</div>');
    for (const r of memberOf) {
      const roles = [...r.roles.entries()].map(([role, n]) => (role || "(no role)")
        + (infos.length > 1 ? ` ×${n}` : "")).join(", ");
      const via = r.via ? ` <span class="muted">via ${escapeHtml(r.via)}</span>` : "";
      parts.push(memberRow(r.id, roles, r.tags, via));
    }
  }

  // A single relation's members (a lanelet's bounds and rules, a traffic light's lights and stop line).
  const one = tokens.length === 1 ? state.details.get(tokens[0]) : null;
  if (one && Array.isArray(one.members)) {
    parts.push(`<div class="subHead">Members (${one.members.length})</div>`);
    if (!one.members.length) parts.push('<div class="hint">No members.</div>');
    for (const m of one.members.slice(0, LIST_MAX)) {
      const tags = m.tags || {};
      parts.push(memberRow(m.id, m.role || "(no role)", kindOf(m.id) === "relation" ? tags
        : { ...tags, type: tags.type || kindOf(m.id) }));
    }
    if (one.members.length > LIST_MAX) parts.push(`<div class="hint">and ${one.members.length - LIST_MAX} more</div>`);
  }
  if (state.note) parts.push(`<div class="hint warnText">${escapeHtml(state.note)}</div>`);
  else if (state.source === "josm" && store.mapSource === "scene_server") {
    parts.push('<div class="hint">From the scene server\'s lanelet2 map (read-only).</div>');
  } else if (state.source === "josm") {
    parts.push(`<div class="hint">From JOSM. ${edit.on ? "Click a value to change it (one undo step in JOSM)."
      : "Values can be changed in Edit mode."} Tags edited in JOSM show here once you come back to this window.</div>`);
  }
  rendering = true;
  try {
    body.innerHTML = parts.join("");
  } finally {
    rendering = false;
  }
  const input = body.querySelector(".tagInput");
  if (input && state.editing) {
    input.focus();
    const at = Math.min(state.editing.caret ?? input.value.length, input.value.length);
    input.setSelectionRange(at, at);
  }
}

/** The tags shared by every described object, as `key=value` lines. */
export function tagsAsText() {
  const infos = state.tokens.map((t) => state.details.get(t)).filter(Boolean);
  return mergedTags(infos).filter((r) => r.same).map((r) => `${r.key}=${[...r.values.keys()][0]}`).join("\n");
}

export function installInspector() {
  $("inspectRefresh")?.addEventListener("click", () => refresh());
  $("inspectCopy")?.addEventListener("click", () => {
    const text = tagsAsText();
    const done = () => toast(`Copied ${text ? text.split("\n").length : 0} tag(s)`, "info", 1500);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, () => toast("The browser refused the clipboard", "warn"));
    } else {
      toast("No clipboard in this browser", "warn");
    }
  });
  const body = $("inspectBody");
  body?.addEventListener("click", (e) => {
    const cell = e.target.closest("td.editable[data-key]");
    if (cell) {
      startEdit(cell.dataset.key);
      return;
    }
    const item = e.target.closest("[data-token]");
    if (!item) return;
    selection.set([itemOf(item.dataset.token)]);
  });
  body?.addEventListener("input", (e) => {
    if (!e.target.classList.contains("tagInput") || !state.editing) return;
    state.editing.draft = e.target.value;
    state.editing.caret = e.target.selectionStart;
  });
  body?.addEventListener("keydown", (e) => {
    if (!e.target.classList.contains("tagInput")) return;
    if (e.key === "Enter") {
      e.preventDefault();
      e.stopPropagation();
      commitEdit();
    } else if (e.key === "Escape") {
      // The field's own Esc: not the selection's, not the panel's.
      e.preventDefault();
      e.stopPropagation();
      cancelEdit();
    }
  });
  body?.addEventListener("focusout", (e) => {
    if (rendering || !e.target.classList.contains("tagInput") || !state.editing) return;
    commitEdit();
  });
  selection.on("changed", () => schedule());
  editEvents.on("mode", () => {
    if (!edit.on) state.editing = null;
    render();
    schedule();
  });
  // A selected way changed in the stream (geometry, type or subtype).
  store.on("upsert", (f) => {
    if (state.key && state.tokens.includes(f.id)) schedule(true);
  });
  window.addEventListener("focus", () => { if (state.tokens.length) schedule(true); });
  ui.on("tab", (name) => { if (name === "inspect" && (state.stale || inspectedTokens().join(",") !== state.key)) refresh(); });
  ui.on("panel", (open) => { if (open && currentTab() === "inspect" && state.stale) refresh(); });
  render();
}

if (TEST_HOOKS) {
  window.__ll2inspect = {
    state: () => ({ tokens: [...state.tokens], source: state.source, note: state.note, loading: state.loading,
                    details: Object.fromEntries(state.details), text: tagsAsText(),
                    editOn: edit.on }),
    refresh: () => refresh(),
    editing: () => (state.editing ? { ...state.editing } : null),
    saving: () => state.saving,
  };
}
