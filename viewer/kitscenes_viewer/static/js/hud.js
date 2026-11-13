// Readouts: connection, counts, camera rows and frame debug (Debug panel),
// profiling, and the toast.
import { PROFILE } from "./util.js";
import { camera, view, headingDeg, ortho, metresPerPixel } from "./camera.js";

const el = (id) => document.getElementById(id);

export const hud = {
  status: el("status"),
  conn: el("conn"),
  count: el("count"),
  seq: el("seq"),
  anchor: el("anchor"),
  edit: el("edit"),
  look: el("look"),
  camPos: el("camPos"),
  camAng: el("camAng"),
  camPivot: el("camPivot"),
  zoomTarget: el("zoomTarget"),
  josmView: el("josmView"),
  frameDebug: el("frameDebug"),
  perf: el("perf"),
  render: el("render"),
};

// With ?profile=1 the Debug panel also shows the apply and render timings.
if (PROFILE) {
  for (const id of ["perfRow", "perf", "renderRow", "render"]) {
    const row = el(id);
    if (row) row.hidden = false;
  }
}

const fmt1 = (v) => Number(v).toFixed(1);
export const fmtVec3 = (v) => `(${fmt1(v.x)}, ${fmt1(v.y)}, ${fmt1(v.z)})`;

export function setConn(live, text) {
  hud.status.classList.toggle("live", !!live);
  hud.conn.textContent = text;
}

export function refreshCounts(count, seq, anchor) {
  hud.count.textContent = String(count);
  if (seq !== undefined && seq !== null) hud.seq.textContent = String(seq);
  if (anchor && Number.isFinite(anchor.lat) && Number.isFinite(anchor.lon)) hud.anchor.textContent = `${anchor.lat.toFixed(5)}, ${anchor.lon.toFixed(5)}`;
}

/** Camera rows; `nav` supplies the last orbit pivot. */
export function refreshCameraHud(nav) {
  if (!hud.camPos) return;
  hud.camPos.textContent = fmtVec3(camera.position);
  hud.camAng.textContent =
    `heading ${fmt1(headingDeg())}° pitch ${fmt1((view.pitch * 180) / Math.PI)}° `
    + (ortho.on ? `ortho ${fmt1(metresPerPixel() * 100)} cm/px` : `fov ${fmt1(camera.fov)}°`);
  if (hud.zoomTarget) {
    const z = nav.zoomTarget;
    hud.zoomTarget.textContent = z
      ? `${fmtVec3(z)}, ${fmt1(camera.position.distanceTo(z))} m (point under the cursor)`
      : "-";
  }
  if (!hud.camPivot) return;
  const p = nav.lastPivot;
  if (p) {
    hud.camPivot.textContent =
      `${fmtVec3(p)} on ${nav.lastPivotKind}, ${fmt1(camera.position.distanceTo(p))} m`;
  } else {
    hud.camPivot.textContent = nav.lastPivotKind === "none" ? "none (turned in place)" : "-";
  }
}

export function setEditHud(text) {
  if (hud.edit) hud.edit.textContent = text;
}

/** The selection card over the view in Edit mode: what is selected, which tool. */
export function setSelectionCard(text) {
  const what = el("selWhat");
  if (what) what.textContent = text;
}

export function setLookHud(text) {
  if (hud.look) hud.look.textContent = text;
}

export function setJosmViewStatus(text) {
  if (hud.josmView) hud.josmView.textContent = text;
  console.log("[viewer] josm view:", text);
}

export function setFrameDebug(lines) {
  if (!hud.frameDebug) return;
  hud.frameDebug.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
}

/** Frame-debug panel for F / B: the box and where the camera was placed. */
export function reportFrameDebug(label, box, params) {
  if (!box || box.isEmpty()) {
    setFrameDebug([`${label}: empty / invalid bounds`, "no geometry to frame"]);
    return;
  }
  const c = box.getCenter(params.target.clone());
  const s = box.getSize(params.target.clone());
  setFrameDebug([
    label,
    `bbox min ${fmtVec3(box.min)} max ${fmtVec3(box.max)}`,
    `center ${fmtVec3(c)} size (${fmt1(s.x)}, ${fmt1(s.y)}, ${fmt1(s.z)}) m`,
    `frame radius ${fmt1(params.radius)} m dist ${fmt1(params.dist)} m`,
    `placed cam ${fmtVec3(params.camPos)} looking at ${fmtVec3(params.target)}`,
  ]);
}

let toastTimer = null;

/** Short message over the bottom of the view; kind is "info", "warn" or "error". */
export function toast(text, kind = "info", ms = 4000) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = text;
  el.className = `show ${kind}`;
  if (toastTimer !== null) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = kind; toastTimer = null; }, ms);
  if (kind === "error") console.warn("[viewer]", text);
}

export function setPerfLine(line) {
  if (hud.perf) hud.perf.textContent = line;
}

export function setRenderLine(line) {
  if (hud.render) hud.render.textContent = line;
}
