// End-to-end checks of the keyboard controls: Space / C everywhere,
// "keys move the selection" (T), height-only walking, zoom target marker.
// Held keys move per rendered frame, and software GL on a busy machine
// renders few, so these checks test the direction of a move, not its size.
//   node viewer/tests/browser/e2e_controls.mjs [--shots DIR]
import { openViewer, Checks, dist2, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;

const features = [
  { id: "way/10", kind: "line", tags: { type: "line_thin" },
    pts: [0, 0, Z, 10, 0, Z, 20, 0, Z], nodes: [1001, 1002, 1003] },
  { id: "way/11", kind: "line", tags: { type: "line_thin" },
    pts: [0, 10, Z, 20, 10, Z], nodes: [1101, 1102] },
];

const v = await openViewer({ shotsDir });
const node = (id) => v.evaluate(`window.__ll2test.node(${id})`);
const nodes = (ids) => Promise.all(ids.map(node));
const moveCommands = () => v.commands.filter((c) => (c.ops || []).some((o) => o.op === "move_node"));
async function waitFor(pred, ms = 2000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(40);
  }
  return false;
}

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("scene streamed", await v.waitForFeatures(2));
  await v.key("b");

  // ---- JOSM follows the point the camera looks at (all nodes still at Z) ---------------------
  const views = () => v.ops("set_view");
  const recenter = async () => {
    v.commands.length = 0;
    await v.evaluate("document.getElementById('recenterJosmBtn').click()");
    await waitFor(() => views().length > 0);
    return views()[0];
  };
  let cam = await v.camera();
  let op = await recenter();
  t.check("top-down: Recenter JOSM centres JOSM under the camera",
    op && Math.hypot(op.x - cam.pos[0], op.y - cam.pos[1]) < 0.01, JSON.stringify(op));
  await v.drag("right", 640, 360, 0, -400); // tilt up to an oblique view
  cam = await v.camera();
  op = await recenter();
  const centre = await v.evaluate("(() => { const r = document.querySelector('canvas').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()");
  const onScreen = await v.project([op.x, op.y, Z]);
  t.check("oblique: it centres JOSM on the ground point in the middle of the view",
    dist2(onScreen, centre) < 3 && Math.hypot(op.x - cam.pos[0], op.y - cam.pos[1]) > 5,
    `${JSON.stringify(op)} is ${dist2(onScreen, centre).toFixed(1)} px from the centre, pitch ${(cam.pitch * 180 / Math.PI).toFixed(0)} deg`);
  v.commands.length = 0;
  const posBefore = cam.pos;
  await v.drag("right", 640, 360, 400, 0); // turn in place
  const followed = await waitFor(() => views().length > 0, 3000);
  t.check("turning in place pans JOSM toward the new view direction",
    followed && dist3((await v.camera()).pos, posBefore) < 1e-9 && Math.hypot(views()[0].x - op.x, views()[0].y - op.y) > 20,
    JSON.stringify(views()));

  await v.key("b");

  // ---- Space / C move the camera in the normal (non-FPS) mode ---------------------------
  let c0 = await v.camera();
  await v.hold(" ", "Space", 300);
  let c1 = await v.camera();
  t.check("Space lifts the camera outside FPS mode", c1.pos[2] - c0.pos[2] > 0.01, `${(c1.pos[2] - c0.pos[2]).toFixed(2)} m`);
  await v.hold("c", "KeyC", 300);
  const c2 = await v.camera();
  t.check("C lowers it", c2.pos[2] < c1.pos[2] - 0.01);
  c0 = await v.camera();
  await v.hold("ArrowLeft", "ArrowLeft", 300, ["alt"]);
  c1 = await v.camera();
  t.check("alt+left turns the camera left", c1.yaw > c0.yaw + 0.001 && dist3(c1.pos, c0.pos) < 1e-9,
    `${((c1.yaw - c0.yaw) * 180 / Math.PI).toFixed(1)} deg`);
  await v.key("b");

  // ---- keys move the selection --------------------------------------------------------------
  await v.key("e");
  const p = await v.project([5, 0, Z]);
  await v.click(p[0], p[1]);
  await v.key("t");
  t.check("T makes the keys move the selection", (await v.evaluate("window.__ll2test.edit()")).keysMove === "selection");
  const ids = [1001, 1002, 1003];
  let before = await nodes(ids);
  c0 = await v.camera();
  v.commands.length = 0;
  await v.hold("d", "KeyD", 400);
  let after = await nodes(ids);
  c1 = await v.camera();
  const dx = after[0][0] - before[0][0];
  t.check("D moves the selection right (east in a north-up view), camera stays",
    dx > 0.001 && after.every((q, i) => Math.abs(q[0] - before[i][0] - dx) < 1e-9 && q[1] === before[i][1] && q[2] === before[i][2])
      && dist3(c0.pos, c1.pos) < 1e-9, `dx=${dx.toFixed(3)}`);
  t.check("the untouched way stays put", (await node(1101))[0] === 0);
  await waitFor(() => moveCommands().length > 0, 1500);
  t.check("one key hold is one command, level: x/y only",
    moveCommands().length === 1 && moveCommands()[0].ops.every((o) => o.x !== undefined && o.z === undefined),
    JSON.stringify(moveCommands().map((c) => c.ops.length)));

  before = await nodes(ids);
  v.commands.length = 0;
  await v.hold(" ", "Space", 300);
  after = await nodes(ids);
  t.check("Space lifts the selection", after.every((q, i) => q[2] > before[i][2] + 0.001 && q[0] === before[i][0]));
  await waitFor(() => moveCommands().length > 0, 1500);
  t.check("a vertical key move sends z only", moveCommands().length === 1
    && moveCommands()[0].ops.every((o) => o.z !== undefined && o.x === undefined));

  before = await nodes(ids);
  await v.hold("ArrowLeft", "ArrowLeft", 400, ["alt"]);
  after = await nodes(ids);
  const angle = (q) => Math.atan2(q[1] - after[1][1], q[0] - after[1][0]);
  const angle0 = (q) => Math.atan2(q[1] - before[1][1], q[0] - before[1][0]);
  const turned = angle(after[2]) - angle0(before[2]);
  t.check("alt+left turns the selection counter-clockwise about its centre",
    turned > 0.0005 && dist2(after[1], before[1]) < 1e-6, `${(turned * 180 / Math.PI).toFixed(2)} deg`);

  // ---- height only: walking moves the camera, Space the selection -------------------------
  await v.key("h");
  before = await nodes(ids);
  c0 = await v.camera();
  await v.hold("w", "KeyW", 300);
  await v.hold(" ", "Space", 300);
  after = await nodes(ids);
  c1 = await v.camera();
  t.check("height only: W walks the camera", dist3(c0.pos, c1.pos) > 0.01);
  t.check("height only: Space still lifts the selection, x/y fixed",
    after.every((q, i) => q[2] > before[i][2] && q[0] === before[i][0] && q[1] === before[i][1]));
  await v.key("h");

  // ---- T back: keys move the camera again ------------------------------------------------
  await v.key("t");
  before = await nodes(ids);
  c0 = await v.camera();
  await v.hold("d", "KeyD", 300);
  t.check("T again: keys move the camera", dist3((await v.camera()).pos, c0.pos) > 0.01
    && (await nodes(ids)).every((q, i) => dist3(q, before[i]) < 1e-9));

  // ---- wheel shows where it zooms to --------------------------------------------------------
  await v.key("b");
  const w = await v.project([10, 10, Z]);
  await v.wheel(w[0], w[1], -100, 1);
  const marker = await v.evaluate("document.getElementById('zoomTarget').textContent");
  t.check("wheel marks its target in the HUD", marker.includes("(10.0, 10.0, 100.0)"), marker);
  await v.shot("zoom_marker.png");

  // ---- street-level imagery ------------------------------------------------------------
  await v.evaluate("window.__opened = []; window.open = (u) => { window.__opened.push(u); return null; }");
  const clickEl = async (sel) => {
    const r = await v.evaluate(`(() => { const b = document.querySelector(${JSON.stringify(sel)}).getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()`);
    await v.click(r[0], r[1]);
  };
  await clickEl("#imageryBtn");
  t.check("Street view opens its provider menu", await v.evaluate("!document.getElementById('imageryMenu').hidden"));
  await clickEl('#imageryMenu [data-provider="google"]');
  const opened = await v.evaluate("window.__opened");
  const ps = await nodes(ids);
  const cx = ps.reduce((a, q) => a + q[0], 0) / ps.length;
  const cy = ps.reduce((a, q) => a + q[1], 0) / ps.length;
  const R = 6378137.0;
  const lat = 49.0 + (cy / R) * 180 / Math.PI;
  const lon = 8.4 + (cx / (R * Math.cos(49.0 * Math.PI / 180))) * 180 / Math.PI;
  const m = /viewpoint=([-\d.]+),([-\d.]+)&heading=(\d+)/.exec(opened[0] || "");
  t.check("Google Street View opens at the selection's centre with the camera heading",
    m && Math.abs(Number(m[1]) - lat) < 1e-6 && Math.abs(Number(m[2]) - lon) < 1e-6
      && Number(m[3]) === Math.round((((-(await v.camera()).yaw * 180 / Math.PI) % 360) + 360) % 360) % 360,
    opened[0]);
  t.check("the menu closes after a choice", await v.evaluate("document.getElementById('imageryMenu').hidden"));

  // A clicked toolbar button must not keep focus, or Enter clicks it again.
  // (Space is safe either way: the viewer prevents its default as a move key.)
  await clickEl("#lanesBtn");
  const lanesAfterClick = (await v.evaluate("window.__ll2test.lanelets()")).visible;
  await v.cdp("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", text: "\r", windowsVirtualKeyCode: 13 });
  await v.cdp("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
  await sleep(100);
  t.check("Enter after clicking a button does not click it again",
    (await v.evaluate("window.__ll2test.lanelets()")).visible === lanesAfterClick);

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} finally {
  await v.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
