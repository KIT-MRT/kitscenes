// End-to-end checks of the interface around the 3D view: the side panel and
// its tabs, remembered sections, the command palette, the keyboard sheet,
// settings, the compass, menus and popovers, tooltips, the status-bar hints,
// a read-only map, and a narrow window.
//   node viewer/tests/browser/e2e_ui.mjs [--shots DIR]
import { openViewer, Checks, dist2, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;

const features = [
  { id: "way/10", kind: "line", tags: { type: "line_thin" },
    pts: [0, 0, Z, 40, 0, Z, 80, 0, Z], nodes: [1001, 1002, 1003] },
  { id: "way/11", kind: "line", tags: { type: "line_thin" },
    pts: [0, 30, Z, 80, 30, Z], nodes: [1101, 1102] },
];

const v = await openViewer({ shotsDir });
const ui = () => v.evaluate("window.__ll2ui.state()");
const hidden = (id) => v.evaluate(`(() => { const e = document.getElementById(${JSON.stringify(id)});`
  + " return !e || e.hidden || !!e.closest('[hidden]') || getComputedStyle(e).display === 'none'; })()");
const frames = (n = 2) => v.evaluate(`new Promise((r) => { let k = ${n}; const f = () => (--k <= 0 ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); })`);
const box = (sel) => v.evaluate(`(() => { const r = document.querySelector(${JSON.stringify(sel)}).getBoundingClientRect();`
  + " return { x: r.left, y: r.top, w: r.width, h: r.height, cx: r.left + r.width / 2, cy: r.top + r.height / 2 }; })()");
async function clickEl(sel) {
  const b = await box(sel);
  await v.click(b.cx, b.cy);
  await sleep(120);
}
async function waitFor(pred, ms = 4000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(50);
  }
  return false;
}
async function reload() {
  await v.cdp("Page.reload", { ignoreCache: true });
  await sleep(300);
  await waitFor(() => v.evaluate("typeof window.__ll2test === 'object' && typeof window.__ll2ui === 'object'")
    .catch(() => false), 30000);
  await v.waitForFeatures(2);
  await frames(3);
}

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("map streamed", await v.waitForFeatures(2));
  await v.key("b");
  await frames();

  // ---- layout ------------------------------------------------------------------------
  let s = await ui();
  t.check("a wide window starts with the side panel open, on Layers", s.panelOpen && s.tab === "layers", JSON.stringify(s));
  t.check("no scene tab without sensor data, no debug tab by default",
    !s.available.scene && !s.available.debug, JSON.stringify(s.available));
  let canvas = await box("canvas");
  const panel = await box("#sidePanel");
  t.check("the 3D view ends where the panel starts", Math.abs(canvas.x + canvas.w - panel.x) <= 1,
    `canvas right ${canvas.x + canvas.w}, panel left ${panel.x}`);
  const topbar = await box("#topbar");
  const status = await box("#statusbar");
  t.check("and sits between the top bar and the status bar",
    Math.abs(canvas.y - (topbar.y + topbar.h)) <= 1 && Math.abs(canvas.y + canvas.h - status.y) <= 1,
    JSON.stringify({ canvas, topbar, status }));
  t.check("no camera pad, no help wall, no HUD box any more",
    !(await v.evaluate("!!document.getElementById('camPad') || !!document.getElementById('help') || !!document.getElementById('hud')")));
  t.check("every icon placeholder got its icon",
    await v.evaluate("[...document.querySelectorAll('[data-icon]')].every((e) => e.querySelector('svg'))"));
  await v.shot("ui_layout.png");

  // ---- the panel: toggle, tabs, remembered ------------------------------------------------
  const widthOpen = canvas.w;
  // A projected point must land where it is drawn after the resize, i.e. the
  // camera's aspect followed the new canvas.
  await v.key("\\", { code: "Backslash" });
  await frames(3);
  s = await ui();
  canvas = await box("canvas");
  t.check("\\ folds the panel away and the view takes the room", !s.panelOpen && canvas.w > widthOpen + 200,
    `${widthOpen} -> ${canvas.w}`);
  const aspect = await v.evaluate("(() => { const c = document.querySelector('canvas'); return [c.width / c.height, window.__ll2test.camera().fov]; })()");
  const camAspect = await v.evaluate("(() => { const p = window.__ll2test.project(80, 0, 100); const q = window.__ll2test.project(0, 0, 100); return [p, q]; })()");
  t.check("the drawing buffer follows the new size", Math.abs(aspect[0] - canvas.w / canvas.h) < 0.02, JSON.stringify(aspect));
  t.check("and projected points stay on the canvas", camAspect.every((p) => p[0] >= canvas.x - 1 && p[0] <= canvas.x + canvas.w + 1),
    JSON.stringify(camAspect));
  await clickEl('#sidePanel .tabs [data-tab="layers"]');
  s = await ui();
  t.check("a tab opens the panel again", s.panelOpen && s.tab === "layers");
  await clickEl('#sidePanel .tabs [data-tab="layers"]');
  t.check("the active tab folds it, as in most editors", !(await ui()).panelOpen);
  await clickEl("#panelBtn");
  t.check("the top-bar button opens it too", (await ui()).panelOpen);

  await clickEl('.sect[data-sect="map"] > .sectHead');
  t.check("a section header collapses its section",
    await v.evaluate("document.querySelector('.sect[data-sect=\"map\"]').classList.contains('closed')"));
  await clickEl("#panelBtn");
  await reload();
  s = await ui();
  t.check("after a reload the panel is still folded away", !s.panelOpen, JSON.stringify(s));
  t.check("and the section still collapsed",
    await v.evaluate("document.querySelector('.sect[data-sect=\"map\"]').classList.contains('closed')"));
  await clickEl("#panelBtn");
  await clickEl('.sect[data-sect="map"] > .sectHead');
  t.check("the panel and section open again", (await ui()).panelOpen && !(await hidden("lanesBtn")));

  // ---- dragging the panel's edge -----------------------------------------------------------
  const w0 = (await box("canvas")).w;
  const edge = await box("#panelResize");
  await v.drag("left", edge.cx, edge.cy + 100, -120, 0, { steps: 8 });
  await frames(3);
  const w1 = (await box("canvas")).w;
  t.check("dragging the panel's edge widens the panel", Math.abs(w0 - w1 - 120) <= 2, `${w0} -> ${w1}`);
  await reload();
  t.check("and the width is remembered", Math.abs((await box("canvas")).w - w1) <= 2);
  const edge2 = await box("#panelResize");
  await v.cdp("Input.dispatchMouseEvent", { type: "mousePressed", x: edge2.cx, y: edge2.cy + 100, button: "left", buttons: 1, clickCount: 1 });
  await v.cdp("Input.dispatchMouseEvent", { type: "mouseReleased", x: edge2.cx, y: edge2.cy + 100, button: "left", buttons: 0, clickCount: 1 });
  await v.cdp("Input.dispatchMouseEvent", { type: "mousePressed", x: edge2.cx, y: edge2.cy + 100, button: "left", buttons: 1, clickCount: 2 });
  await v.cdp("Input.dispatchMouseEvent", { type: "mouseReleased", x: edge2.cx, y: edge2.cy + 100, button: "left", buttons: 0, clickCount: 2 });
  await frames(3);
  t.check("a double click puts the default width back", Math.abs((await box("canvas")).w - w0) <= 2);

  // ---- layer rows ---------------------------------------------------------------------------
  const lanesVisible = () => v.evaluate("window.__ll2test.lanelets().visible");
  const before = await lanesVisible();
  await clickEl("#lanesBtn");
  t.check("a layer row toggles its layer and its switch",
    (await lanesVisible()) === !before && (await v.evaluate("document.getElementById('lanesBtn').classList.contains('on')")) === !before);
  await clickEl("#lanesBtn");
  // A slider dragged with the mouse must not keep the keys from the map.
  const slider = await box("#iconAlpha");
  await v.drag("left", slider.x + slider.w - 6, slider.cy, -30, 0, { steps: 4 });
  const camA = await v.camera();
  await v.hold("w", "KeyW", 300);
  t.check("after dragging a slider, WASD still walks", dist3(camA.pos, (await v.camera()).pos) > 0.01);
  await v.evaluate("(() => { const e = document.getElementById('iconAlpha'); e.value = '100'; e.dispatchEvent(new Event('input')); })()");

  // ---- menus and popovers -------------------------------------------------------------------
  await clickEl("#viewMenuBtn");
  t.check("the Camera menu opens", !(await hidden("viewMenu")));
  const c = await box("canvas");
  await v.click(c.x + 200, c.y + 200);
  t.check("a click on the map closes it", await hidden("viewMenu"));
  await clickEl("#viewMenuBtn");
  await v.key("Escape");
  t.check("Esc closes it too", await hidden("viewMenu"));
  await clickEl("#viewMenuBtn");
  let cam0 = await v.camera();
  await v.drag("left", c.x + 300, c.y + 300, 120, 40); // orbit away first
  await clickEl("#viewMenuBtn");
  await clickEl("#bevBtn");
  let cam1 = await v.camera();
  t.check("a menu item runs and closes the menu",
    (await hidden("viewMenu")) && Math.abs(cam1.pitch + Math.PI / 2) < 0.05, JSON.stringify(cam1));
  await clickEl("#connBtn");
  const pop = await v.evaluate("document.getElementById('connPop').textContent");
  t.check("the connection popover says where the map comes from",
    !(await hidden("connPop")) && pop.includes("features from JOSM") && pop.includes("possible"), pop);
  await v.key("Escape");
  t.check("and closes with Esc", await hidden("connPop"));

  // ---- tooltips ------------------------------------------------------------------------------
  const help = await box("#helpBtn");
  await v.hover(help.cx, help.cy);
  await sleep(700);
  const tip = await v.evaluate("(() => { const e = document.getElementById('tooltip'); return e.hidden ? null : e.textContent; })()");
  t.check("hovering a button shows its tooltip with the shortcut", tip && tip.includes("Keyboard") && tip.includes("?"), tip);
  await v.hover(c.x + 100, c.y + 100);
  await sleep(100);
  t.check("and moving away hides it", await hidden("tooltip"));

  // ---- command palette --------------------------------------------------------------------------
  await v.key("k", { modifiers: ["ctrl"] });
  s = await ui();
  t.check("Ctrl+K opens the command palette", s.modal === "palette", JSON.stringify(s));
  t.check("with the search field focused", await v.evaluate("document.activeElement && document.activeElement.id === 'paletteInput'"));
  cam0 = await v.camera();
  await v.cdp("Input.insertText", { text: "lanelet" });
  await sleep(100);
  const first = await v.evaluate("document.querySelector('#paletteList .palItem.active .ttl').textContent");
  t.check("typing finds the command", first === "Lanelet surfaces and arrows", first);
  await v.hold("w", "KeyW", 200);
  cam1 = await v.camera();
  t.check("keys typed into the palette do not move the camera", dist3(cam0.pos, cam1.pos) < 1e-6);
  const lanesBefore = await lanesVisible();
  await v.key("Enter");
  t.check("Enter runs it and closes the palette",
    (await lanesVisible()) === !lanesBefore && (await ui()).modal === null);
  await v.key("l");
  t.check("(L puts the lanelets back)", (await lanesVisible()) === lanesBefore);
  await v.key("k", { modifiers: ["ctrl"] });
  await v.cdp("Input.insertText", { text: "move tool" });
  await sleep(100);
  const disabled = await v.evaluate("[...document.querySelectorAll('#paletteList .palItem')].find((e) => e.textContent.includes('Move tool'))?.className");
  t.check("commands that do not apply now are shown dimmed", disabled && disabled.includes("disabled"), disabled);
  await v.key("Enter");
  t.check("and Enter does not run them", (await ui()).modal === "palette" && !(await v.evaluate("window.__ll2test.edit().on")));
  await v.key("Escape");
  t.check("Esc closes the palette", (await ui()).modal === null);
  await clickEl("#paletteBtn");
  t.check("the Search button opens it too", (await ui()).modal === "palette");
  await v.key("Escape");

  // ---- keyboard sheet ----------------------------------------------------------------------------
  await v.key("?", { code: "Slash", modifiers: ["shift"] });
  s = await ui();
  const sheet = await v.evaluate("document.getElementById('helpBody').textContent");
  t.check("? opens the keyboard and mouse sheet", s.modal === "helpSheet", JSON.stringify(s));
  t.check("which lists the mouse and the commands' keys",
    sheet.includes("Orbit the point under the cursor") && sheet.includes("Frame the whole map")
    && sheet.includes("Follow the car"), sheet.slice(0, 200));
  await v.shot("ui_help.png");
  cam0 = await v.camera();
  await v.hold("w", "KeyW", 200);
  t.check("while it is open the map gets no keys", dist3(cam0.pos, (await v.camera()).pos) < 1e-6);
  await v.key("Escape");
  t.check("Esc closes it", (await ui()).modal === null);

  // ---- settings -------------------------------------------------------------------------------------
  await clickEl("#settingsBtn");
  t.check("the settings dialog opens", (await ui()).modal === "settingsDlg");
  await clickEl("#setDebug");
  s = await ui();
  t.check("ticking Debug offers the debug tab", s.available.debug && s.prefs.debug, JSON.stringify(s));
  await v.evaluate("(() => { const e = document.getElementById('setWalkSpeed'); e.value = '2'; e.dispatchEvent(new Event('input')); })()");
  t.check("the walk speed reads 4×", (await v.hud("setWalkSpeedLabel")) === "4×", await v.hud("setWalkSpeedLabel"));
  await v.key("Escape");
  s = await ui();
  t.check("Esc closes settings; the debug panel is there", s.modal === null && s.tab === "debug", JSON.stringify(s));
  t.check("and it holds the camera readouts", !(await hidden("camPos")) && (await v.hud("camPos")) !== "-");
  // Walk at 4x, then at 1x: the same hold goes clearly further at 4x.
  const walkFor = async () => {
    const a = await v.camera();
    await v.hold("w", "KeyW", 400);
    const b = await v.camera();
    return dist3(a.pos, b.pos);
  };
  const fast = await walkFor();
  await clickEl("#settingsBtn");
  await v.evaluate("(() => { const e = document.getElementById('setWalkSpeed'); e.value = '0'; e.dispatchEvent(new Event('input')); })()");
  await v.key("Escape");
  const slow = await walkFor();
  t.check("the walk speed setting scales WASD", fast > slow * 1.8, `${fast.toFixed(2)} m at 4x, ${slow.toFixed(2)} m at 1x`);
  await clickEl("#settingsBtn");
  await clickEl("#setReset");
  await v.key("Escape");
  s = await ui();
  t.check("Reset puts the interface back (no debug tab)", !s.available.debug && s.tab === "layers" && s.panelOpen,
    JSON.stringify(s));

  // ---- compass ---------------------------------------------------------------------------------------
  const cc = await box("canvas");
  await v.drag("right", cc.cx, cc.cy, 260, 0); // turn the view
  cam0 = await v.camera();
  t.check("(the view is turned away from north)", Math.abs(cam0.yaw) > 0.3, String(cam0.yaw));
  const roseTurned = await v.evaluate("document.getElementById('compassRose').style.transform");
  t.check("the compass rose turns with the view", /rotate\(-?[1-9]/.test(roseTurned), roseTurned);
  await clickEl("#compassBtn");
  cam1 = await v.camera();
  t.check("clicking the compass faces north", Math.abs(Math.atan2(Math.sin(cam1.yaw), Math.cos(cam1.yaw))) < 1e-3,
    String(cam1.yaw));
  await clickEl("#orthoBtn");
  t.check("2D switches to the flat map view", (await v.evaluate("window.__ll2map.ortho().on"))
    && (await v.evaluate("document.getElementById('orthoBtn').classList.contains('on')")));
  const hints2d = (await ui()).hints;
  t.check("and the status bar says what the mouse does there", hints2d.includes("pan") && hints2d.includes("turn"), hints2d);
  await v.key("v");

  // ---- status hints follow the mode ---------------------------------------------------------------------
  t.check("view mode hints", (await ui()).hints.includes("orbit"), (await ui()).hints);
  await clickEl("#editBtn");
  t.check("Edit in the top bar turns edit mode on", await v.evaluate("window.__ll2test.edit().on"));
  t.check("edit mode hints", (await ui()).hints.includes("select"), (await ui()).hints);
  t.check("the tool rail and the selection card show", !(await hidden("editBar")) && !(await hidden("selCard")));
  await v.shot("ui_edit.png");
  await clickEl("#viewModeBtn");
  t.check("View turns it off again", !(await v.evaluate("window.__ll2test.edit().on")) && (await hidden("editBar")));

  // ---- the Selection tab: every tag, from JOSM -----------------------------------------------------------
  const TAGS = {
    "way/10": { type: "line_thin", subtype: "dashed", file_origin: "/maps/a.osm", lane_change: "yes" },
    "way/11": { type: "line_thin", subtype: "solid", file_origin: "/maps/a.osm" },
  };
  const describes = () => v.ops("describe");
  v.reply = (cmd) => {
    const d = (cmd.ops || []).find((o) => o.op === "describe");
    if (!d) return { ok: true, message: "ok" };
    return { ok: true, message: "described", details: d.ids.map((id) => ({
      id, tags: TAGS[id] || {}, nodes: 3,
      memberships: id === "way/10" ? [{ id: "relation/7", role: "left", tags: { type: "lanelet", subtype: "road" } }] : [],
    })) };
  };
  const insp = () => v.evaluate("window.__ll2inspect.state()");
  await clickEl('#sidePanel .tabs [data-tab="inspect"]');
  t.check("the Selection tab says how to select something",
    (await v.evaluate("document.getElementById('inspectBody').textContent")).includes("Click a line"));
  v.commands.length = 0;
  await v.key("e");
  let at = await v.reveal([20, 0, Z]);
  await v.click(at[0], at[1]);
  t.check("selecting a way asks JOSM to describe it",
    await waitFor(async () => (await insp()).source === "josm"), JSON.stringify(await insp()));
  t.check("once, for that way", describes().length === 1 && describes()[0].ids.join() === "way/10",
    JSON.stringify(describes()));
  let body = await v.evaluate("document.getElementById('inspectBody').textContent");
  t.check("every tag shows, not just type and subtype",
    body.includes("file_origin") && body.includes("/maps/a.osm") && body.includes("lane_change"), body);
  t.check("with the relation it belongs to and its role", body.includes("left") && body.includes("lanelet road")
    && body.includes("relation/7"), body);
  t.check("and the way's length from the stream", body.includes("80.0 m"), body);
  await v.shot("ui_inspect.png");
  at = await v.reveal([40, 30, Z]);
  await v.click(at[0], at[1], { modifiers: ["shift"] });
  t.check("adding a second way describes both",
    await waitFor(async () => (await insp()).tokens.length === 2 && !(await insp()).loading));
  body = await v.evaluate("document.getElementById('inspectBody').textContent");
  t.check("a tag they disagree on reads as two values", body.includes("2 values"), body);
  t.check("a tag only one has says so", body.includes("on 1 of 2"), body);
  t.check("the summary counts them", (await v.hud("inspectSummary")) === "2 ways", await v.hud("inspectSummary"));
  t.check("Copy takes the tags they share", (await insp()).text === "type=line_thin\nfile_origin=/maps/a.osm",
    JSON.stringify((await insp()).text));
  await clickEl('#inspectBody .inspectItem[data-token="way/11"]');
  t.check("clicking one of them selects only that one",
    await waitFor(async () => (await insp()).tokens.join() === "way/11"));
  // Not showing: nothing is asked.
  await clickEl('#sidePanel .tabs [data-tab="layers"]');
  const asked = describes().length;
  await v.key("Escape");
  at = await v.reveal([20, 0, Z]);
  await v.click(at[0], at[1]);
  await sleep(500);
  t.check("while the tab is not showing, JOSM is not asked", describes().length === asked);
  await clickEl('#sidePanel .tabs [data-tab="inspect"]');
  t.check("opening it asks for what is selected now",
    await waitFor(async () => (await insp()).tokens.join() === "way/10" && (await insp()).source === "josm"));
  // Without JOSM's answer (the scene server refuses edits and describes):
  v.reply = (cmd) => ({ ok: false, message: "The 3D scene server is read-only" });
  await clickEl("#inspectRefresh");
  t.check("without JOSM the streamed tags show, and why",
    await waitFor(async () => (await insp()).source === "stream"), JSON.stringify(await insp()));
  const st = await insp();
  t.check("type and subtype from the stream", st.details["way/10"].tags.type === "line_thin" && st.note.includes("scene server"),
    JSON.stringify(st));
  v.reply = () => ({ ok: true, message: "ok" });
  await v.key("Escape");
  await v.key("e");
  await clickEl('#sidePanel .tabs [data-tab="layers"]');

  // ---- a read-only map -------------------------------------------------------------------------------------
  v.sendScene({ type: "snapshot", source: "scene_server", anchor: { lat: 49.0, lon: 8.4 }, features });
  await waitFor(() => v.evaluate("document.getElementById('editBtn').classList.contains('blocked')"));
  t.check("a map from the scene server marks Edit as unavailable",
    await v.evaluate("document.getElementById('editBtn').classList.contains('blocked')"));
  await v.key("e");
  const toast = await v.evaluate("window.__ll2test.toast()");
  t.check("E does not enter edit mode, and says why",
    !(await v.evaluate("window.__ll2test.edit().on")) && toast && toast.text.includes("read-only"), JSON.stringify(toast));
  await clickEl("#connBtn");
  t.check("the connection popover says so too",
    (await v.evaluate("document.getElementById('connPop').textContent")).includes("scene server (view only)"));
  await v.key("Escape");

  // ---- selecting to inspect, in View mode (read-only map) ----------------------------------------------------
  // The scene server answers describe from its lanelet2 map; the fake bridge plays it.
  v.reply = (cmd) => {
    const d = (cmd.ops || []).find((o) => o.op === "describe");
    if (!d) return { ok: false, message: "The 3D scene server is read-only" };
    return { ok: true, message: "Described from the scene server's map", details: d.ids.map((id) => ({
      id, tags: { type: "line_thin", subtype: "dashed", lane_change: "no" }, nodes: 3,
      memberships: [{ id: "relation/7", role: "left", tags: { type: "lanelet", subtype: "road" } }] })) };
  };
  await clickEl('#sidePanel .tabs [data-tab="inspect"]');
  const cam0v = await v.camera();
  at = await v.reveal([20, 0, Z]);
  const camAfterReveal = await v.camera();
  await v.click(at[0], at[1]);
  t.check("in View mode a click selects the line", (await v.evaluate("window.__ll2test.selection()")).ways.join() === "way/10",
    JSON.stringify(await v.evaluate("window.__ll2test.selection()")));
  t.check("without the edit tools or the gizmo", !(await v.evaluate("window.__ll2test.edit().on")) && (await hidden("editBar"))
    && (await v.evaluate("window.__ll2test.gizmo()")).at === null);
  t.check("and the click did not move the camera", JSON.stringify((await v.camera()).pos) === JSON.stringify(camAfterReveal.pos));
  t.check("the scene server's tags show, and where they come from",
    await waitFor(async () => (await v.evaluate("document.getElementById('inspectBody').textContent")).includes("lane_change")),
    await v.evaluate("document.getElementById('inspectBody').textContent"));
  t.check("(from the scene server's map)",
    (await v.evaluate("document.getElementById('inspectBody').textContent")).includes("scene server"));
  t.check("the status bar says a click inspects", (await ui()).hints.includes("inspect"), (await ui()).hints);
  const cc2 = await box("canvas");
  await v.drag("left", cc2.cx, cc2.cy + 40, 80, 0);
  t.check("a drag still orbits and keeps the selection",
    (await v.evaluate("window.__ll2test.selection()")).ways.join() === "way/10");
  await v.key("Escape");
  t.check("Esc deselects in View mode too", (await v.evaluate("window.__ll2test.selection()")).ways.length === 0);

  // ---- View mode picks ways and lanes, not nodes ---------------------------------------------------------------
  const lane = { id: "relation/7", kind: "lanelet", tags: { subtype: "road" }, pts: [40, 15, Z],
                 left: "way/11", right: "way/10", lrev: false, rrev: false, two: false, arrow: [40, 15, Z, 1, 0, 0, 30] };
  v.sendScene({ type: "snapshot", source: "scene_server", anchor: { lat: 49.0, lon: 8.4 }, features: [...features, lane] });
  await waitFor(() => v.evaluate("window.__ll2test.lanelets().count === 1").catch(() => false));
  const LL_DETAILS = {
    "way/10": { id: "way/10", tags: { type: "line_thin" }, nodes: 3, memberships: [
      { id: "relation/7", role: "right", tags: { type: "lanelet", subtype: "road" } },
      { id: "relation/20", role: "regulatory_element", tags: { type: "regulatory_element", subtype: "traffic_light" },
        via: "relation/7" }] },
    "relation/7": { id: "relation/7", tags: { type: "lanelet", subtype: "road" }, memberships: [], members: [
      { id: "way/11", role: "left", tags: { type: "line_thin" } },
      { id: "way/10", role: "right", tags: { type: "line_thin" } },
      { id: "relation/20", role: "regulatory_element", tags: { type: "regulatory_element", subtype: "traffic_light" } }] },
    "relation/20": { id: "relation/20", tags: { type: "regulatory_element", subtype: "traffic_light" }, memberships: [
      { id: "relation/7", role: "regulatory_element", tags: { type: "lanelet", subtype: "road" } }], members: [
      { id: "way/30", role: "refers", tags: { type: "traffic_light", subtype: "red_yellow_green" } },
      { id: "way/31", role: "ref_line", tags: { type: "stop_line" } }] },
  };
  v.reply = (cmd) => {
    const d = (cmd.ops || []).find((o) => o.op === "describe");
    if (!d) return { ok: false, message: "The 3D scene server is read-only" };
    return { ok: true, message: "described", details: d.ids.map((id) => LL_DETAILS[id] || { id, tags: {}, memberships: [] }) };
  };
  const vsel = () => v.evaluate("window.__ll2test.selection()");
  at = await v.reveal([0, 0, Z]);
  await v.click(at[0], at[1]);
  let s1 = await vsel();
  t.check("a View-mode click on a node selects its way, not the node",
    s1.nodes.length === 0 && s1.ways.join() === "way/10", JSON.stringify(s1));
  t.check("the lane's traffic light shows, via the lanelet",
    await waitFor(async () => (await v.evaluate("document.getElementById('inspectBody').textContent")).includes("via relation/7")),
    await v.evaluate("document.getElementById('inspectBody').textContent"));
  at = await v.reveal([40, 15, Z]);
  t.check("hovering the lane between two lines marks the lanelet",
    await waitFor(async () => {
      await v.hover(at[0], at[1]);
      return (await v.evaluate("window.__ll2test.hover()")) === "relation/7";
    }), String(await v.evaluate("window.__ll2test.hover()")));
  await v.click(at[0], at[1]);
  s1 = await vsel();
  t.check("and a click there selects it", s1.relations.join() === "relation/7" && !s1.ways.length, JSON.stringify(s1));
  t.check("the lanelet stays in the viewer: JOSM is not told to select a relation",
    !v.ops("select").some((o) => o.ids.some((id) => id.startsWith("relation/"))), JSON.stringify(v.ops("select")));
  t.check("its members are listed",
    await waitFor(async () => (await v.evaluate("document.getElementById('inspectBody').textContent")).includes("Members (3)")),
    await v.evaluate("document.getElementById('inspectBody').textContent"));
  t.check("the summary counts a relation", (await v.hud("inspectSummary")) === "1 relation", await v.hud("inspectSummary"));
  await v.shot("ui_lanelet.png");
  v.sendScene({ type: "selection", nodes: [], ways: [] });
  await sleep(300);
  t.check("an empty JOSM selection that matches leaves the lanelet selected",
    (await vsel()).relations.join() === "relation/7", JSON.stringify(await vsel()));
  await clickEl('#inspectBody .memberRow[data-token="relation/20"]');
  t.check("a member row selects that relation (the traffic light)",
    await waitFor(async () => (await vsel()).relations.join() === "relation/20"), JSON.stringify(await vsel()));
  t.check("which lists its lights and stop line",
    await waitFor(async () => {
      const b = await v.evaluate("document.getElementById('inspectBody').textContent");
      return b.includes("refers") && b.includes("ref_line") && b.includes("way/31");
    }), await v.evaluate("document.getElementById('inspectBody').textContent"));
  await v.key("Escape");
  // Nodes in View mode again, with the setting.
  await v.evaluate("document.getElementById('setViewNodes').click()");
  at = await v.reveal([0, 0, Z]);
  await v.click(at[0], at[1]);
  s1 = await vsel();
  t.check("with \"View clicks select nodes\" on, a click on a node selects the node",
    s1.nodes.join() === "node/1001", JSON.stringify(s1));
  await v.evaluate("document.getElementById('setViewNodes').click()");
  t.check("(the setting is remembered)", (await ui()).prefs.viewNodes === false, JSON.stringify((await ui()).prefs));
  await v.key("Escape");
  v.reply = () => ({ ok: true, message: "ok" });
  void cam0v;
  await clickEl('#sidePanel .tabs [data-tab="layers"]');

  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  await waitFor(() => v.evaluate("!document.getElementById('editBtn').classList.contains('blocked')"));
  await v.key("e");
  t.check("once JOSM sends the map, editing works again", await v.evaluate("window.__ll2test.edit().on"));
  await v.key("e");

  // ---- changing tag values (JOSM's map, Edit mode) ------------------------------------------------------------
  const JOSM_TAGS = {
    "way/10": { type: "line_thin", subtype: "dashed", lane_change: "yes" },
    "way/11": { type: "line_thin", subtype: "solid", lane_change: "both" },
  };
  let refuse = null;
  v.reply = (cmd) => {
    const sets = (cmd.ops || []).filter((o) => o.op === "set_tag");
    if (sets.length) {
      if (refuse) return { ok: false, message: refuse };
      for (const o of sets) JOSM_TAGS[o.id][o.key] = o.value;
      return { ok: true, message: `Applied ${sets.length} change(s)` };
    }
    const d = (cmd.ops || []).find((o) => o.op === "describe");
    if (!d) return { ok: true, message: "ok" };
    return { ok: true, message: "described", details: d.ids.map((id) => ({
      id, tags: { ...(JOSM_TAGS[id] || {}) }, nodes: 2, memberships: [] })) };
  };
  const bodyText = () => v.evaluate("document.getElementById('inspectBody').textContent");
  const typeText = async (text) => {
    await v.evaluate("document.activeElement.select()");
    await v.cdp("Input.insertText", { text });
    await sleep(80);
  };
  const editCell = async (key) => {
    const sel = `#inspectBody td.editable[data-key="${key}"]`;
    await waitFor(() => v.evaluate(`!!document.querySelector(${JSON.stringify(sel)})`));
    await clickEl(sel);
  };
  await clickEl('#sidePanel .tabs [data-tab="inspect"]');
  await v.key("Escape");
  const pickAt = async (p) => {
    const xy = await v.reveal(p);
    return { xy, pick: await v.evaluate(`window.__ll2test.pick(${xy[0]}, ${xy[1]})`),
             el: await v.evaluate(`document.elementFromPoint(${xy[0]}, ${xy[1]})?.tagName`) };
  };
  let probe = await pickAt([20, 0, Z]);
  await v.click(probe.xy[0], probe.xy[1]);
  t.check("in View mode JOSM's values are shown but not editable",
    await waitFor(async () => (await insp()).tokens.join() === "way/10" && (await insp()).source === "josm"
      && !(await insp()).loading)
    && !(await v.evaluate("!!document.querySelector('#inspectBody td.editable')"))
    && (await bodyText()).includes("Values can be changed in Edit mode"),
    JSON.stringify(probe) + " " + await bodyText());
  await v.key("e");
  t.check("in Edit mode every value is",
    await waitFor(() => v.evaluate("document.querySelectorAll('#inspectBody td.editable').length === 3")),
    await bodyText());
  v.commands.length = 0;
  await editCell("lane_change");
  t.check("a click turns the value into a field, focused and filled in",
    (await v.evaluate("window.__ll2inspect.editing()"))?.key === "lane_change"
    && (await v.evaluate("document.activeElement.classList.contains('tagInput') && document.activeElement.value")) === "yes");
  await typeText("no");
  await v.key("e");
  t.check("typing an E into the field does not leave Edit mode", await v.evaluate("window.__ll2test.edit().on"));
  await typeText("no");
  await v.key("Enter");
  t.check("Enter sends one set_tag to JOSM",
    await waitFor(() => v.ops("set_tag").length === 1)
    && JSON.stringify(v.ops("set_tag")) === JSON.stringify([{ op: "set_tag", id: "way/10", key: "lane_change", value: "no" }]),
    JSON.stringify(v.ops("set_tag")));
  t.check("and the tab shows JOSM's new value",
    await waitFor(async () => (await insp()).details["way/10"]?.tags.lane_change === "no" && !(await insp()).loading),
    JSON.stringify((await insp()).details));
  await editCell("lane_change");
  await typeText("zzz");
  await v.key("Escape");
  t.check("Esc cancels: nothing sent, the selection stays",
    v.ops("set_tag").length === 1 && !(await v.evaluate("window.__ll2inspect.editing()"))
    && (await vsel()).ways.join() === "way/10", JSON.stringify(await vsel()));
  await editCell("lane_change");
  await typeText("");
  await v.key("Enter");
  const emptyToast = await v.evaluate("window.__ll2test.toast()");
  t.check("an empty value is refused, not sent as a deletion",
    v.ops("set_tag").length === 1 && emptyToast && emptyToast.text.includes("empty"), JSON.stringify(emptyToast));
  await editCell("subtype");
  await typeText("solid_solid");
  await clickEl("#inspectSummary");
  t.check("leaving the field applies it too",
    await waitFor(() => v.ops("set_tag").some((o) => o.key === "subtype" && o.value === "solid_solid")),
    JSON.stringify(v.ops("set_tag")));
  probe = await pickAt([40, 30, Z]);
  await v.click(probe.xy[0], probe.xy[1], { modifiers: ["shift"] });
  t.check("two ways selected",
    await waitFor(async () => (await insp()).tokens.length === 2 && !(await insp()).loading),
    JSON.stringify(probe) + " " + JSON.stringify(await vsel()));
  v.commands.length = 0;
  await editCell("lane_change");
  t.check("a value they disagree on starts empty", (await v.evaluate("document.activeElement.value")) === "");
  await typeText("left");
  await v.key("Enter");
  t.check("and is set on both, in one command (one undo step)",
    await waitFor(() => v.ops("set_tag").length === 2)
    && v.commands.filter((c) => (c.ops || []).some((o) => o.op === "set_tag")).length === 1
    && JSON.stringify(v.ops("set_tag").map((o) => o.id).sort()) === JSON.stringify(["way/10", "way/11"]),
    JSON.stringify(v.commands));
  refuse = "The edit layer is hidden in JOSM; show it to edit from the 3D viewer";
  await editCell("type");
  await typeText("line_thick");
  await v.key("Enter");
  t.check("JOSM's refusal shows",
    await waitFor(async () => ((await v.evaluate("window.__ll2test.toast()"))?.text || "").includes("hidden")));
  t.check("and the old value stays", (await insp()).details["way/10"].tags.type === "line_thin");
  refuse = null;
  await v.key("e");
  t.check("leaving Edit mode takes the fields away",
    await waitFor(() => v.evaluate("!document.querySelector('#inspectBody td.editable')")));
  await v.key("Escape");

  // ---- a selected relation shows its members on the map --------------------------------------------------------
  const light = { id: "way/30", kind: "line", tags: { type: "traffic_light", subtype: "red_yellow_green" },
                  pts: [70, 50, Z + 3, 74, 50, Z + 3], nodes: [3001, 3002] };
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features: [...features, lane, light] });
  await waitFor(() => v.evaluate("window.__ll2test.lanelets().count === 1").catch(() => false));
  const REL = {
    "way/10": { id: "way/10", tags: { type: "stop_line" }, nodes: 3, memberships: [
      { id: "relation/20", role: "ref_line", tags: { type: "regulatory_element", subtype: "traffic_light" } },
      { id: "relation/21", role: "ref_line", tags: { type: "regulatory_element", subtype: "right_of_way" } }] },
    "relation/20": { id: "relation/20", tags: { type: "regulatory_element", subtype: "traffic_light" }, memberships: [],
      members: [{ id: "way/30", role: "refers", tags: light.tags }, { id: "way/10", role: "ref_line", tags: {} },
                { id: "way/99", role: "refers", tags: {} }] },
    "relation/21": { id: "relation/21", tags: { type: "regulatory_element", subtype: "right_of_way" }, memberships: [],
      members: [{ id: "relation/7", role: "right_of_way", tags: { type: "lanelet" } },
                { id: "node/3001", role: "refers", tags: {} }] },
  };
  v.reply = (cmd) => {
    const d = (cmd.ops || []).find((o) => o.op === "describe");
    if (!d) return { ok: true, message: "ok" };
    return { ok: true, message: "described", details: d.ids.map((id) => REL[id] || { id, tags: {}, memberships: [] }) };
  };
  const hl = () => v.evaluate("window.__ll2test.highlight()");
  at = await v.reveal([20, 0, Z]);
  await v.click(at[0], at[1]);
  await waitFor(async () => (await bodyText()).includes("relation/20"));
  t.check("a selected way highlights itself, in yellow", (await hl()).segments === 2 && (await hl()).members.segments === 0,
    JSON.stringify(await hl()));
  await clickEl('#inspectBody .memberRow[data-token="relation/20"]');
  t.check("a traffic light rule highlights its light and its stop line, as members",
    await waitFor(async () => (await vsel()).relations.join() === "relation/20" && (await hl()).members.segments === 3),
    JSON.stringify(await hl()));
  t.check("in the member colour, not the selection's", (await hl()).segments === 0, JSON.stringify(await hl()));
  t.check("(a member outside the stream is simply left out)", (await hl()).members.triangles === 0, JSON.stringify(await hl()));
  {
    const mid = await v.reveal([72, 50, Z + 3]);
    await frames(3);
    const px = await v.pixel(mid[0], mid[1]);
    t.check("on screen the light is red", px[0] > 180 && px[1] < 90 && px[2] < 110, JSON.stringify(px));
  }
  await v.shot("ui_regelem_members.png");
  await v.key("Escape");
  at = await v.reveal([20, 0, Z]);
  await v.click(at[0], at[1]);
  await waitFor(async () => (await bodyText()).includes("relation/21"));
  await clickEl('#inspectBody .memberRow[data-token="relation/21"]');
  t.check("a right of way highlights its lanelet (bounds and surface) and its node",
    await waitFor(async () => { const h = (await hl()).members; return h.triangles > 0 && h.segments === 3 && h.dots === 1; }),
    JSON.stringify(await hl()));
  await v.key("Escape");
  t.check("deselecting clears it", JSON.stringify((await hl()).members) === JSON.stringify({ segments: 0, dots: 0, triangles: 0 }),
    JSON.stringify(await hl()));
  v.reply = () => ({ ok: true, message: "ok" });
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  await clickEl('#sidePanel .tabs [data-tab="layers"]');

  // ---- a narrow window (the viewer next to JOSM) ---------------------------------------------------------------
  await v.cdp("Emulation.setDeviceMetricsOverride", { width: 700, height: 700, deviceScaleFactor: 1, mobile: false });
  await frames(4);
  await v.evaluate("window.__ll2ui.setPanelOpen(true)");
  await frames(3);
  canvas = await box("canvas");
  const tabs = await box("#sidePanel .tabs");
  t.check("in a narrow window the open panel floats over the map instead of squeezing it",
    Math.abs(canvas.x + canvas.w - tabs.x) <= 2 && canvas.w > 600, JSON.stringify({ canvas, tabs }));
  t.check("and the top bar still fits", await v.evaluate("document.getElementById('topbar').scrollWidth <= window.innerWidth + 1"));
  await v.shot("ui_narrow.png");
  await v.cdp("Emulation.clearDeviceMetricsOverride");
  await frames(3);

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} catch (err) {
  t.check("no exception", false, String(err && err.stack ? err.stack : err));
} finally {
  await v.close();
}

console.log(`\n${t.failures ? `${t.failures} of ${t.count} FAILED` : `all ${t.count} passed`}`);
process.exit(t.failures ? 1 : 0);
