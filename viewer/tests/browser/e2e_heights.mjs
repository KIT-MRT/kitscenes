// End-to-end checks of typed heights and snapping.
//   node viewer/tests/browser/e2e_heights.mjs [--shots DIR]
import { openViewer, Checks, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;

const line = (id, nodes, pts) => ({ id: `way/${id}`, kind: "line", tags: { type: "line_thin" }, pts: pts.flat(), nodes });
const features = [
  line(10, [1001, 1002, 1003], [[0, 0, Z], [10, 0, Z], [20, 0, Z]]),
  line(11, [1101, 1102], [[0, 10, Z], [-10, 10, Z]]),
  line(20, [2001, 2002], [[3, 10, 103], [6, 13, 103]]),          // position snap target
  line(60, [6001, 6002], [[40, 0, Z], [40, -8, Z]]),             // height snap: moves
  line(61, [6101, 6102], [[43, 0, 104.2], [52, 0, 104.2]]),      //   ... to 6101's height
  // A lanelet at 106 m whose bound nodes are all >10 m from node 7001, so
  // only its surface can be the snap target.
  line(70, [7011, 7012], [[20, -20, 106], [80, -20, 106]]),
  line(71, [7021, 7022], [[20, -24, 106], [80, -24, 106]]),
  { id: "relation/700", kind: "lanelet", tags: { subtype: "road" }, pts: [41, -22, 106],
    left: "way/70", right: "way/71", lrev: false, rrev: false, two: false, arrow: [41, -22, 106, 1, 0, 0, 4] },
  line(72, [7001, 7002], [[50, -22, Z], [50, -40, Z]]),
];

const v = await openViewer({ shotsDir });
const node = (id) => v.evaluate(`window.__ll2test.node(${id})`);
const editState = () => v.evaluate("window.__ll2test.edit()");
const zValue = () => v.evaluate("document.getElementById('zField').value");
const moveOps = () => v.ops("move_node");
async function waitFor(pred, ms = 2000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(40);
  }
  return false;
}
async function select(ids) {
  await v.key("Escape");
  for (const [i, id] of ids.entries()) {
    const p = await v.reveal(await node(id));
    await v.click(p[0], p[1], { modifiers: i ? ["shift"] : [] });
  }
}
async function typeHeight(text) {
  await v.key("z");
  await v.cdp("Input.insertText", { text });
  await v.cdp("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
  await v.cdp("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
  await sleep(150);
}
/** Drag a gizmo handle so the pivot moves by `delta` (world), overshooting `extraPx`. */
async function dragPivot(pivotPt, axis, delta, extraPx = 3) {
  await v.reveal(pivotPt);
  const cam = await v.camera();
  const scale = 0.2226 * dist3(cam.pos, pivotPt);
  const h = pivotPt.map((c, k) => c + axis[k] * 0.35 * scale);
  const s0 = await v.project(h);
  const s1 = await v.project(h.map((c, k) => c + delta[k]));
  const dx = s1[0] - s0[0], dy = s1[1] - s0[1];
  const len = Math.hypot(dx, dy) || 1;
  await v.drag("left", s0[0], s0[1], dx + (dx / len) * extraPx, dy + (dy / len) * extraPx, { steps: 12 });
  await sleep(100);
}

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("scene streamed", await v.waitForFeatures(9));
  await v.key("b");
  await v.key("e");

  // ---- typed heights ----------------------------------------------------------------
  await v.key("Escape");
  const p10 = await v.reveal([5, 0, Z]);
  await v.click(p10[0], p10[1]);
  t.check("the Z field shows the selection's height", (await zValue()) === "100", await zValue());
  v.commands.length = 0;
  await typeHeight("+0.5");
  t.check("+0.5 raises every selected node by 0.5 m",
    (await Promise.all([1001, 1002, 1003].map(node))).every((q) => Math.abs(q[2] - 100.5) < 1e-9));
  await waitFor(() => moveOps().length > 0);
  t.check("as one height-only command", v.commands.filter((c) => (c.ops || []).some((o) => o.op === "move_node")).length === 1
    && moveOps().length === 3 && moveOps().every((o) => o.z === 100.5 && o.x === undefined));
  await typeHeight("=99");
  t.check("=99 sets an absolute height", (await node(1002))[2] === 99);
  v.commands.length = 0;
  await typeHeight("abc");
  await sleep(200);
  t.check("garbage changes nothing and says how to type", (await node(1002))[2] === 99 && moveOps().length === 0
    && ((await v.evaluate("window.__ll2test.toast()")) || {}).text?.includes("+0.2"));
  await v.key("z");
  await v.cdp("Input.dispatchKeyEvent", { type: "keyDown", key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38 });
  await v.cdp("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
  await sleep(150);
  t.check("arrow up in the field steps 1 cm", (await node(1002))[2] === 99.01, String((await node(1002))[2]));

  // ---- a late refusal reverts only what no later gesture moved -------------------------
  let refuseFirst = null;
  v.reply = (cmd) => {
    if (!refuseFirst && (cmd.ops || []).some((o) => o.op === "move_node")) {
      return new Promise((r) => { refuseFirst = () => r({ ok: false, message: "refused for the test" }); });
    }
    return { ok: true, message: "ok" };
  };
  await select([1001, 1002]);
  const before1001 = (await node(1001))[2];
  await typeHeight("=101"); // refused, but only after the next move
  await select([1002]);
  await typeHeight("=102");
  refuseFirst();
  await sleep(300);
  t.check("a late refusal reverts the nodes only it moved", (await node(1001))[2] === before1001,
    `${(await node(1001))[2]} (was ${before1001})`);
  t.check("... but not a node a later accepted move moved again", (await node(1002))[2] === 102,
    String((await node(1002))[2]));
  v.reply = () => ({ ok: true, message: "ok" });

  // ---- snap a single node onto another ------------------------------------------------
  await v.key("m");
  t.check("M turns snapping on", (await editState()).snap);
  await select([1101]);
  let a = await node(1101);
  const target = await node(2001);
  await dragPivot(a, [1, 0, 0], [target[0] - a[0], 0, 0]);
  a = await node(1101);
  t.check("a dragged node snaps onto the node it is dropped on (x, y and z)",
    dist3(a, target) < 1e-9, JSON.stringify(a));
  await v.key("z", { modifiers: ["ctrl"] }); // undo in JOSM (the fake bridge ignores it); put it back locally
  await typeHeight("=100");

  // ---- height snap to the nearest node, in height-only mode ----------------------------
  const sp = await v.reveal([40, 0, Z]);
  await v.drag("left", sp[0], sp[1], 0, -110); // orbit about node 6001 to an oblique view
  await v.key("h");
  await select([6001]);
  let n = await node(6001);
  await dragPivot(n, [0, 0, 1], [0, 0, 104.2 - n[2]]);
  n = await node(6001);
  t.check("a height drag snaps to the nearest node's height", n[2] === 104.2 && n[0] === 40 && n[1] === 0,
    JSON.stringify(n));
  await v.shot("snap_height.png");

  // Ctrl inverts: same drag without snapping lands a little off.
  await typeHeight("=100");
  n = await node(6001);
  await dragPivot(n, [0, 0, 1], [0, 0, 104.2 - n[2]], 12);
  await sleep(50);
  // (Ctrl has to be held on the pointer events of the drag.)
  const unsnapped = await node(6001);
  await typeHeight("=100");
  n = await node(6001);
  await v.reveal(n);
  const cam = await v.camera();
  const scale = 0.2226 * dist3(cam.pos, n);
  const h = [n[0], n[1], n[2] + 0.35 * scale];
  const s0 = await v.project(h);
  const s1 = await v.project([h[0], h[1], h[2] + 4.2]);
  const dy = s1[1] - s0[1];
  await v.drag("left", s0[0], s0[1], 0, dy - 3, { steps: 12, modifiers: ["ctrl"] });
  await sleep(100);
  n = await node(6001);
  t.check("holding ctrl drags without snapping", Math.abs(n[2] - 104.2) > 1e-6 && Math.abs(n[2] - 104.2) < 1,
    `${n[2]} (plain drag with snap: ${unsnapped[2]})`);

  // ---- height snap to the lanelet surface ----------------------------------------------
  await select([7001]);
  n = await node(7001);
  await dragPivot(n, [0, 0, 1], [0, 0, 106 - n[2]]);
  n = await node(7001);
  t.check("a height drag snaps to the lanelet surface under the node", Math.abs(n[2] - 106) < 1e-6, JSON.stringify(n));

  await v.key("m");
  t.check("M turns snapping off", !(await editState()).snap);
  await typeHeight("=100");
  n = await node(7001);
  await dragPivot(n, [0, 0, 1], [0, 0, 106 - n[2]]);
  n = await node(7001);
  t.check("without snapping the same drag stops where the mouse does", Math.abs(n[2] - 106) > 1e-6, String(n[2]));

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} finally {
  await v.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
