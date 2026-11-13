// The Layers tab's line controls: Styled / Simple, the 3D switch for curbs,
// rails and walls, and a legend drawn from the same style table the shader
// uses (render/linestyles.js), so it cannot show something else.
import { TEST_HOOKS } from "../util.js";
import { lineLayer } from "../render/lines.js";
import { store } from "../store.js";
import { LINE_STYLES } from "../render/linestyles.js";
import { prefs, savePrefs, escapeHtml } from "./shell.js";

const $ = (id) => document.getElementById(id);

const GROUPS = [
  ["Markings", ["line_thin", "line_thin/dashed", "line_thin/solid_solid", "line_thin/solid_dashed", "line_thick",
    "line_thick/dashed", "line_thick/solid_solid", "stop_line", "pedestrian_marking", "bike_marking",
    "bike_marking/dashed", "marking/halt", "zebra_marking", "zig-zag"]],
  ["Borders", ["curbstone", "curbstone/low", "curbstone/low/entry", "road_border", "guard_rail", "jersey_barrier",
    "wall", "fence"]],
  ["Other", ["virtual", "rail", "building", "trajectory/car", "traffic_sign", "traffic_light", "default"]],
];

/** A small picture of one style: 54 x 14 px, one metre of marking = 6 px along. */
function swatch(s) {
  const W = 54, H = 14, PX_PER_M = 6;
  const across = Math.max(2, Math.min(10, s.width * 22));
  const y0 = (H - across) / 2;
  const parts = [];
  const dashes = (y, h, on, off, color, gap) => {
    const period = (on + off) * PX_PER_M;
    if (gap) parts.push(`<rect x="0" y="${y}" width="${W}" height="${h}" fill="${gap}"/>`);
    for (let x = 0; x < W; x += period) {
      parts.push(`<rect x="${x}" y="${y}" width="${Math.max(1, on * PX_PER_M)}" height="${h}" fill="${color}"/>`);
    }
  };
  const solid = (y, h, color) => parts.push(`<rect x="0" y="${y}" width="${W}" height="${h}" fill="${color}"/>`);
  const pattern = s.pattern || "solid";
  if (["double", "solid_dashed", "dashed_solid"].includes(pattern)) {
    const stripe = Math.max(1.5, across * (s.stripe / s.width));
    const top = [y0, stripe], bottom = [y0 + across - stripe, stripe];
    // Looking along the way (left to right here), the left stripe is on top.
    const leftDashed = pattern === "dashed_solid";
    const rightDashed = pattern === "solid_dashed";
    if (leftDashed) dashes(...top, s.dash[0], s.dash[1], s.color, s.gap); else solid(...top, s.color);
    if (rightDashed) dashes(...bottom, s.dash[0], s.dash[1], s.color, s.gap); else solid(...bottom, s.color);
  } else if (pattern === "dashed" || pattern === "dotted") {
    dashes(y0, across, s.dash[0], s.dash[1], s.color, s.gap);
  } else {
    solid(y0, across, s.color);
  }
  if (s.raise) {
    parts.push(`<rect x="0" y="0" width="${W}" height="2" fill="${s.raise.color}" opacity="0.9"/>`);
  }
  return `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">${parts.join("")}</svg>`;
}

function renderLegend() {
  const el = $("lineLegend");
  if (!el) return;
  const rows = [];
  for (const [name, keys] of GROUPS) {
    rows.push(`<div class="grp">${escapeHtml(name)}</div>`);
    for (const key of keys) {
      const s = LINE_STYLES[key];
      if (!s) continue;
      const size = s.raise ? ` · ${Math.round(s.raise.top * 100)} cm high` : s.width >= 0.1 ? ` · ${Math.round(s.width * 100)} cm` : "";
      const dash = s.dash ? ` · ${s.dash[0]} / ${s.dash[1]} m` : "";
      rows.push(`${swatch(s)}<span title="${escapeHtml(key)}">${escapeHtml(s.label)}<span class="dim">${escapeHtml(size + dash)}</span></span>`);
    }
  }
  el.innerHTML = rows.join("");
}

function refresh() {
  $("lineStyledBtn")?.classList.toggle("on", lineLayer.styled);
  $("lineSimpleBtn")?.classList.toggle("on", !lineLayer.styled);
  const raised = $("raisedBtn");
  if (raised) {
    raised.classList.toggle("on", lineLayer.raised);
    raised.disabled = !lineLayer.styled;
  }
}

export function setLineStyle(styled) {
  lineLayer.setStyled(styled);
  prefs.lineStyle = styled ? "styled" : "simple";
  savePrefs();
  refresh();
}

export function setRaised(on) {
  lineLayer.setRaised(on);
  prefs.raised = !!on;
  savePrefs();
  refresh();
}

export function installLineStyleUi() {
  // Remembered per browser; styled with raised shapes by default.
  lineLayer.setStyled(prefs.lineStyle !== "simple");
  lineLayer.setRaised(prefs.raised !== false);
  $("lineStyledBtn")?.addEventListener("click", () => setLineStyle(true));
  $("lineSimpleBtn")?.addEventListener("click", () => setLineStyle(false));
  $("raisedBtn")?.addEventListener("click", () => setRaised(!lineLayer.raised));
  $("lineLegendBtn")?.addEventListener("click", () => {
    const el = $("lineLegend");
    el.hidden = !el.hidden;
    $("lineLegendBtn").classList.toggle("on", !el.hidden);
    if (!el.hidden && !el.innerHTML) renderLegend();
  });
  refresh();
}

if (TEST_HOOKS) {
  window.__ll2lines = {
    stats: () => lineLayer.stats(),
    setStyled: (on) => setLineStyle(on),
    setRaised: (on) => setRaised(on),
    // Segment i of a way as the paint draws it: [start, end, [d0, d1], style key].
    segment: (id, i) => {
      const f = store.features.get(id);
      const g = f && f.vChunk && f.vChunk.paint.geometry;
      if (!g || !g.attributes.aStart || f.sOff === undefined) return null;
      const k = f.sOff + i;
      const a = g.attributes;
      return [[a.aStart.getX(k), a.aStart.getY(k), a.aStart.getZ(k)], [a.aEnd.getX(k), a.aEnd.getY(k), a.aEnd.getZ(k)],
              [a.aDist.getX(k), a.aDist.getY(k)], Object.keys(LINE_STYLES)[a.aStyle.getX(k)]];
    },
    legendRows: () => document.querySelectorAll("#lineLegend svg").length,
  };
}
