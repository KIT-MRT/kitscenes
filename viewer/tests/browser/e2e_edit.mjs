// End-to-end checks of 3D editing: selection, JOSM sync, gizmo moves,
// height-only, rotation, refusals, box select, cycling, delete, undo.
//   node viewer/tests/browser/e2e_edit.mjs [--shots DIR]
import { openViewer, Checks, dist2, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100; // absolute heights, like a real map's `ele`

function line(id, nodeIds, pts) {
  return { id: `way/${id}`, kind: "line", tags: { type: "line_thin" }, pts: pts.flat(), nodes: nodeIds };
}
const features = [
  line(10, [1001, 1002, 1003, 1004], [[0, 0, Z], [10, 0, Z], [20, 0, Z], [30, 0, Z]]),
  line(11, [1101, 1102, 1103], [[0, 10, Z], [10, 10, Z], [20, 10, Z]]),
  // node 1201 and node 1301 sit on exactly the same spot
  line(12, [1201, 1202], [[40, 5, Z], [45, 5, Z]]),
  line(13, [1301, 1302], [[40, 5, Z], [40, 15, Z]]),
];

const v = await openViewer({ shotsDir });
const node = (id) => v.evaluate(`window.__ll2test.node(${id})`);
const sel = () => v.evaluate("window.__ll2test.selection()");
const editState = () => v.evaluate("window.__ll2test.edit()");
const toastNow = () => v.evaluate("window.__ll2test.toast()");
const sameSet = (a, b) => a.length === b.length && a.every((x) => b.includes(x));
async function centroid(ids) {
  const ps = await Promise.all(ids.map(node));
  return [0, 1, 2].map((k) => ps.reduce((s, p) => s + p[k], 0) / ps.length);
}
// Screen spot on a gizmo handle: `frac` of the handle's length along `axis`
// from the pivot (TransformControls scales handles by 0.2226 * distance here).
async function handleSpot(pivotPt, axis, frac) {
  const cam = await v.camera();
  const scale = 0.2226 * dist3(cam.pos, pivotPt);
  return v.project(pivotPt.map((c, k) => c + axis[k] * frac * scale));
}
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
  t.check("scene streamed", await v.waitForFeatures(4));
  await v.key("b");

  // ---- hover and selecting ------------------------------------------------------
  const hoverAt = async (pt) => {
    const q = await v.project(pt);
    await v.hover(q[0], q[1]);
    return v.evaluate("window.__ll2test.hover()");
  };
  // Selecting to inspect works in View mode too, so the hover shows there.
  t.check("in View mode the hover marks what a click would select too", (await hoverAt([5, 0, Z])) === "way/10");
  await v.key("e");
  t.check("E turns edit mode on", (await editState()).on);
  t.check("edit toolbar shows", await v.evaluate("!document.getElementById('editBar').hidden"));
  t.check("hovering a segment marks the way a click would select", (await hoverAt([5, 0, Z])) === "way/10");
  t.check("hovering a node marks the node", (await hoverAt([0, 10, Z])) === "node/1101");
  await v.shot("edit_hover.png");
  t.check("hovering empty ground marks nothing", (await hoverAt([5, 5, Z])) === null);
  await hoverAt([0, 10, Z]);
  await v.keyDown("w", "KeyW");
  await sleep(200);
  const whileMoving = await v.evaluate("window.__ll2test.hover()");
  await v.keyUp("w", "KeyW");
  await v.hover(...(await v.project([0, 10, Z])).slice(0, 2));
  t.check("the mark hides while the camera moves and comes back when it rests",
    whileMoving === null && (await v.evaluate("window.__ll2test.hover()")) === "node/1101", String(whileMoving));
  await v.key("b");
  await hoverAt([5, 0, Z]);

  let p = await v.project([5, 0, Z]);
  await v.click(p[0], p[1]);
  let s = await sel();
  t.check("click on a segment selects the way", sameSet(s.ways, ["way/10"]) && s.nodes.length === 0, JSON.stringify(s));
  t.check("selection is sent to JOSM", await waitFor(() => v.ops("select").some((o) => sameSet(o.ids, ["way/10"]))),
    JSON.stringify(v.ops("select")));

  p = await v.project([0, 10, Z]);
  await v.click(p[0], p[1], { modifiers: ["shift"] });
  s = await sel();
  t.check("shift+click adds a node", sameSet(s.nodes, ["node/1101"]) && sameSet(s.ways, ["way/10"]), JSON.stringify(s));
  await v.shot("edit_selection.png");

  // ---- gizmo move of the whole selection -------------------------------------------
  const moved = [1001, 1002, 1003, 1004, 1101];
  const before = await Promise.all(moved.map(node));
  let piv = await centroid(moved);
  let a = await handleSpot(piv, [1, 0, 0], 0.35);
  v.commands.length = 0;
  await v.drag("left", a[0], a[1], 90, 0, { steps: 10 });
  const after = await Promise.all(moved.map(node));
  const dx = after[0][0] - before[0][0];
  t.check("X drag moves every selected node by the same +x",
    dx > 0.5 && after.every((q, i) => Math.abs(q[0] - before[i][0] - dx) < 1e-6
      && Math.abs(q[1] - before[i][1]) < 1e-6 && Math.abs(q[2] - before[i][2]) < 1e-6), `dx=${dx.toFixed(3)}`);
  await waitFor(() => v.ops("move_node").length > 0);
  let mv = v.ops("move_node");
  t.check("one command moves all 5 nodes", v.commands.filter((c) => (c.ops || []).some((o) => o.op === "move_node")).length === 1
    && sameSet(mv.map((o) => o.id), moved.map((id) => `node/${id}`)), JSON.stringify(mv));
  t.check("a level move sends x/y but no z", mv.every((o) => o.x !== undefined && o.y !== undefined && o.z === undefined));
  t.check("the unselected way 11 kept its other nodes", (await node(1102))[0] === 10);

  // ---- height only ----------------------------------------------------------------
  const sp = await v.project([15, 5, Z]);
  await v.drag("left", sp[0] + 200, sp[1] + 180, 0, -120); // orbit to an oblique view
  await v.key("h");
  t.check("H switches to height only", (await editState()).heightOnly);
  const beforeH = await Promise.all(moved.map(node));
  piv = await centroid(moved);
  a = await handleSpot(piv, [0, 0, 1], 0.35);
  v.commands.length = 0;
  await v.drag("left", a[0], a[1], 0, -60, { steps: 10 });
  const afterH = await Promise.all(moved.map(node));
  const dz = afterH[0][2] - beforeH[0][2];
  t.check("Z drag lifts every node, x/y untouched",
    dz > 0.2 && afterH.every((q, i) => Math.abs(q[2] - beforeH[i][2] - dz) < 1e-6
      && q[0] === beforeH[i][0] && q[1] === beforeH[i][1]), `dz=${dz.toFixed(3)}`);
  await waitFor(() => v.ops("move_node").length > 0);
  mv = v.ops("move_node");
  t.check("a height-only move sends z only (lat/lon untouched in JOSM)",
    mv.length === 5 && mv.every((o) => o.z !== undefined && o.x === undefined && o.y === undefined), JSON.stringify(mv[0]));
  await v.shot("edit_height.png");

  // ---- JOSM refuses: the move is reverted and explained ------------------------------
  v.reply = () => ({ ok: false, message: "The edit layer is hidden in JOSM; show it to edit from the 3D viewer" });
  const beforeR = await Promise.all(moved.map(node));
  a = await handleSpot(await centroid(moved), [0, 0, 1], 0.35);
  await v.drag("left", a[0], a[1], 0, -60, { steps: 10 });
  const reverted = await waitFor(async () => {
    const now = await Promise.all(moved.map(node));
    return now.every((q, i) => dist3(q, beforeR[i]) < 1e-9);
  });
  t.check("refused move is put back", reverted);
  const toast = await toastNow();
  t.check("and the reason is shown", toast && toast.kind.includes("error") && toast.text.includes("hidden"),
    JSON.stringify(toast));
  v.reply = () => ({ ok: true, message: "ok" });

  // ---- rotate ---------------------------------------------------------------------
  await v.key("h"); // back to free moves
  await v.key("r");
  t.check("R switches to rotate", (await editState()).tool === "rotate");
  const beforeRot = await Promise.all(moved.map(node));
  const c0 = await centroid(moved);
  a = await handleSpot(c0, [1, 0, 0], 0.5); // on the Z ring (radius 0.5)
  v.commands.length = 0;
  // TransformControls turns a Z rotation from the drag along Z x eye, which
  // is screen-horizontal here: a straight-up drag barely turns at all, and
  // how little depended on the exact framing.
  await v.drag("left", a[0], a[1], 70, 0, { steps: 10 });
  const afterRot = await Promise.all(moved.map(node));
  const c1 = await centroid(moved);
  const pairsKept = afterRot.every((q, i) => afterRot.every((r, j) =>
    Math.abs(dist3(q, r) - dist3(beforeRot[i], beforeRot[j])) < 1e-6));
  const turned = dist3(afterRot[0], beforeRot[0]) > 0.1;
  t.check("rotation turns the selection rigidly about its centre",
    turned && pairsKept && dist3(c0, c1) < 1e-6 && afterRot.every((q, i) => q[2] === beforeRot[i][2]),
    `first node moved ${dist3(afterRot[0], beforeRot[0]).toFixed(3)} m`);
  await waitFor(() => v.ops("move_node").length > 0);
  t.check("a rotation sends x/y only", v.ops("move_node").every((o) => o.z === undefined && o.x !== undefined));
  // Nothing to grab where TransformControls' view-axis ring (radius 0.75,
  // facing the camera) would be: it hides it while X/Y are hidden, and a
  // tilt about the view axis must never reach the selection.
  {
    const cam = await v.camera();
    const pxPerUnit = 0.2226 * (await v.evaluate("document.querySelector('canvas').clientHeight")) / (2 * Math.tan((cam.fov * Math.PI) / 360));
    const s1 = await v.project(c1);
    const gizmoAt = (await v.evaluate("window.__ll2test.gizmo()")).at;
    v.commands.length = 0;
    await v.drag("left", s1[0] + 0.75 * pxPerUnit, s1[1], 0, -70, { steps: 10 });
    await sleep(200);
    const afterE = await Promise.all(moved.map(node));
    t.check("the rotate gizmo has no view-axis ring",
      gizmoAt && dist3(gizmoAt, c1) < 1e-6 && afterE.every((q, i) => dist3(q, afterRot[i]) === 0)
      && v.ops("move_node").length === 0, JSON.stringify(gizmoAt));
  }
  await v.key("g");

  // ---- delete, undo, redo -----------------------------------------------------------
  v.commands.length = 0;
  await v.key("Delete");
  await waitFor(() => v.ops("delete_selection").length > 0);
  const del = v.ops("delete_selection")[0];
  t.check("Del asks JOSM to delete the selection", del && sameSet(del.ids, ["node/1101", "way/10"]), JSON.stringify(del));
  await v.key("z", { modifiers: ["ctrl"] });
  await v.key("y", { modifiers: ["ctrl"] });
  await v.key("z", { modifiers: ["ctrl", "shift"] });
  await waitFor(() => v.ops("redo").length >= 2);
  t.check("ctrl+Z undoes, ctrl+Y and ctrl+shift+Z redo in JOSM",
    v.ops("undo").length === 1 && v.ops("redo").length === 2, JSON.stringify(v.ops().map((o) => o.op)));

  // ---- box select ---------------------------------------------------------------------
  await v.key("b");
  await v.key("Escape");
  t.check("Esc clears the selection", (await sel()).nodes.length === 0 && (await sel()).ways.length === 0);
  // The moves and the rotation above took node 1101 far off its way; put it
  // back (locally) so way 11 lies straight inside the box again.
  await v.evaluate(`void window.__ll2test.moveLocal([{ id: 1101, x: 0, y: 10, z: ${Z} }])`);
  // Pan way 11 into open view, clear of the HUD and edit bar on the left.
  const mid11 = await v.project([10, 10, Z]);
  await v.drag("middle", mid11[0], mid11[1], 760 - mid11[0], 460 - mid11[1]);
  const b0 = await v.project([-2, 12, Z]);
  const b1 = await v.project([22, 8, Z]);
  await v.drag("left", b0[0], b0[1], b1[0] - b0[0], b1[1] - b0[1], { modifiers: ["ctrl"] });
  s = await sel();
  t.check("ctrl+drag box selects the nodes inside and the way entirely inside",
    sameSet(s.nodes, ["node/1101", "node/1102", "node/1103"]) && sameSet(s.ways, ["way/11"]), JSON.stringify(s));
  t.check("box select did not orbit the camera", Math.abs((await v.camera()).pitch + Math.PI / 2) < 1e-9);

  // ---- middle-click cycles through coincident items ------------------------------------
  p = await v.reveal([40, 5, Z]);
  const seen = [];
  for (let i = 0; i < 4; i++) {
    await v.click(p[0], p[1], { button: "middle" });
    const cur = await sel();
    seen.push([...cur.nodes, ...cur.ways].join(","));
  }
  t.check("middle-click walks both coincident nodes, then the ways",
    seen[0].startsWith("node/") && seen[1].startsWith("node/") && seen[0] !== seen[1]
      && seen.slice(2).every((x) => x.startsWith("way/")), JSON.stringify(seen));

  // ---- JOSM -> viewer selection ---------------------------------------------------------
  v.commands.length = 0;
  v.sendScene({ type: "selection", nodes: [1003], ways: ["way/12"] });
  const applied = await waitFor(async () => {
    const cur = await sel();
    return sameSet(cur.nodes, ["node/1003"]) && sameSet(cur.ways, ["way/12"]);
  });
  t.check("JOSM's selection shows in the viewer", applied, JSON.stringify(await sel()));
  await sleep(300);
  t.check("and is not echoed back to JOSM", v.ops("select").length === 0, JSON.stringify(v.ops("select")));

  await v.key("e");
  v.sendScene({ type: "selection", nodes: [1002], ways: [] });
  await sleep(200);
  t.check("in View mode JOSM's selection shows right away (to inspect it)", sameSet((await sel()).nodes, ["node/1002"]),
    JSON.stringify(await sel()));
  await v.key("e");
  t.check("and stays when edit mode turns on", sameSet((await sel()).nodes, ["node/1002"]), JSON.stringify(await sel()));

  // ---- interpolate heights -------------------------------------------------------------
  await v.key("Escape");
  const w11 = await v.project([15, 10, Z]);
  await v.click(w11[0], w11[1]);
  v.commands.length = 0;
  v.reply = () => ({ ok: true, message: "Interpolated 1 height(s) along way/11" });
  await v.key("i");
  await waitFor(() => v.ops("interpolate_height").length > 0);
  let ip = v.ops("interpolate_height")[0];
  t.check("I interpolates along the selected way, between its ends",
    ip && ip.way === "way/11" && ip.anchors.length === 0, JSON.stringify(ip));
  t.check("and shows JOSM's answer", await waitFor(async () => {
    const tt = await toastNow();
    return tt && tt.text.includes("Interpolated");
  }));
  await v.key("Escape");
  for (const id of [1101, 1103]) { // 1101 was moved earlier: use where it is now
    const q = await v.project(await node(id));
    await v.click(q[0], q[1], { modifiers: ["shift"] });
  }
  v.commands.length = 0;
  await v.key("i");
  await waitFor(() => v.ops("interpolate_height").length > 0);
  ip = v.ops("interpolate_height")[0];
  t.check("two selected nodes of one way are the anchors",
    ip && ip.way === "way/11" && sameSet(ip.anchors, ["node/1101", "node/1103"]), JSON.stringify(ip));

  // A move JOSM accepts with a warning (height jump) shows the warning.
  v.reply = () => ({ ok: true, message: "ok", warning: "Height jump of 3.5 m over 5.0 m (70 % grade) between node/1 and node/2 on way/9" });
  await v.key("t");
  await v.hold(" ", "Space", 250);
  await v.key("t");
  t.check("a warning in JOSM's answer is shown", await waitFor(async () => {
    const tt = await toastNow();
    return tt && tt.kind.includes("warn") && tt.text.includes("Height jump");
  }, 3000), JSON.stringify(await toastNow()));
  v.reply = () => ({ ok: true, message: "ok" });

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} finally {
  await v.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
