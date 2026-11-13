// End-to-end checks of the 3D viewer in headless Chrome. See README.md.
//   node viewer/tests/browser/e2e_viewer.mjs [--shots DIR]
import { openViewer, Checks, dist2, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const DEG = 180 / Math.PI;
const t = new Checks();

// ---- scene: a 100 m lane, an elevated crossing line, two nodes 30 cm apart --
// Protocol v2, as the JOSM plugin sends it: flat pts, numeric node ids.
function line(id, nodeBase, pts) {
  return { id: `way/${id}`, kind: "line", tags: { type: "line_thin" }, pts: pts.flat(),
    nodes: pts.map((_, i) => nodeBase + i) };
}
const ys = Array.from({ length: 11 }, (_, i) => i * 10);
const features = [
  line(1, 101, ys.map((y) => [-1.75, y, 0])),
  line(2, 201, ys.map((y) => [1.75, y, 0])),
  line(3, 301, [-20, -10, 0, 10, 20].map((x) => [x, 50, 6])),
  line(4, 401, [[5, 20, 0], [5.3, 20, 0]]),
];
// Icon features: a sign panel (4 corners, faced) and a road arrow.
features.push({ id: "way/6", kind: "line", tags: { type: "traffic_sign", subtype: "de274-30" },
  pts: [3, 40, 2, 3, 40, 2.6, 3.6, 40, 2.6, 3.6, 40, 2], nodes: [601, 602, 603, 604] });
features.push({ id: "way/7", kind: "line", tags: { type: "arrow", subtype: "straight" },
  pts: [-1, 60, 0, -1, 63, 0, 1, 63, 0, 1, 60, 0], nodes: [701, 702, 703, 704] });
// Traffic lights the Autoware way: the line is the light's bottom edge, left
// to right as the approaching car sees it, `height` its vertical size. Car
// drives +y, so the light at y = 30 faces -y. The rest must stay on the old
// path: no height, three points, a height that is not a number.
const AW_LEFT = [-3, 30, 5];
const AW_RIGHT = [-2.6, 30, 5];
const light = (id, tags, pts) => ({ id: `way/${id}`, kind: "line",
  tags: { type: "traffic_light", subtype: "red_yellow_green", ...tags },
  pts: pts.flat(), nodes: pts.map((_, i) => id * 100 + i) });
features.push(light(8, { height: "0.5" }, [AW_LEFT, AW_RIGHT]));
features.push(light(9, {}, [[3, 30, 5], [3.4, 30, 5]]));
features.push(light(10, { height: "0.5" }, [[6, 30, 5], [6.2, 30, 5], [6.4, 30, 5]]));
features.push(light(11, { height: "high" }, [[9, 30, 5], [9.4, 30, 5]]));
// Signs drawn the same way are read the same way.
features.push({ id: "way/12", kind: "line", tags: { type: "traffic_sign", subtype: "de274-30", height: "0.6" },
  pts: [-6, 30, 2, -5.4, 30, 2], nodes: [1201, 1202] });
features.push({ id: "way/13", kind: "line", tags: { type: "traffic_sign", subtype: "de274-30" },
  pts: [-9, 30, 2, -8.4, 30, 2], nodes: [1301, 1302] });
const N1 = [5, 20, 0];
const N2 = [5.3, 20, 0];

const v = await openViewer({ shotsDir });
async function waitFor(pred, ms = 4000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(60);
  }
  return false;
}
try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49, lon: 8.4 }, features });
  t.check("scene streamed", await v.waitForFeatures(12));

  // ---- camera ---------------------------------------------------------------
  await v.key("b");
  let cam = await v.camera();
  t.check("B: straight down, north up", Math.abs(cam.yaw) < 1e-9 && Math.abs(cam.pitch * DEG + 90) < 1e-6,
    `yaw ${cam.yaw} pitch ${cam.pitch * DEG}`);
  t.check("HUD shows the new camera rows", (await v.hud("camAng")).startsWith("heading"));

  const P1 = [1.75, 35, 0];
  const s1 = await v.project(P1);
  await v.drag("left", s1[0], s1[1], 160, -90, { midShot: "orbit_mid.png" });
  let piv = await v.pivot();
  t.check("orbit: pivot is the picked line point", piv.kind === "line" && dist3(piv.point, P1) < 0.05,
    JSON.stringify(piv));
  let s1b = await v.project(piv.point);
  t.check("orbit: pivot stays on the press pixel", dist2(s1, s1b) < 0.5,
    `${s1.slice(0, 2).map((x) => x.toFixed(2))} -> ${s1b.slice(0, 2).map((x) => x.toFixed(2))}`);
  cam = await v.camera();
  t.check("orbit: view turned and tilted", Math.abs(cam.yaw) > 0.1 && cam.pitch * DEG > -80,
    `yaw ${(cam.yaw * DEG).toFixed(1)} pitch ${(cam.pitch * DEG).toFixed(1)}`);

  const P2 = [-20, 50, 6];
  const s2 = await v.project(P2);
  await v.drag("left", s2[0], s2[1], -80, 40);
  piv = await v.pivot();
  t.check("orbit #2: pivot on the elevated line", dist3(piv.point, P2) < 0.05, JSON.stringify(piv));
  t.check("orbit #2: pivot pixel fixed", dist2(s2, await v.project(piv.point)) < 0.5);

  let before = await v.camera();
  await v.drag("right", 640, 400, 200, 0);
  cam = await v.camera();
  t.check("look: position unchanged", dist3(cam.pos, before.pos) < 1e-9);
  t.check("look: drag right turns right by 200 px * 0.0025 rad",
    Math.abs((before.yaw - cam.yaw) - 0.5) < 1e-6, `${((before.yaw - cam.yaw) * DEG).toFixed(2)} deg`);

  for (let i = 0; i < 4 && (await v.camera()).pitch * DEG < 10; i++) await v.drag("right", 640, 600, 0, -400);
  cam = await v.camera();
  t.check("look up above the horizon", cam.pitch * DEG > 5, (cam.pitch * DEG).toFixed(1));
  before = cam;
  await v.drag("left", 640, 60, 100, 0);
  cam = await v.camera();
  piv = await v.pivot();
  t.check("left-drag on sky turns in place", dist3(cam.pos, before.pos) < 1e-9 && piv.kind === "none",
    JSON.stringify(piv));

  await v.key("b");
  const P3 = [-1.75, 45, 0];
  const s3 = await v.project(P3);
  await v.drag("middle", s3[0], s3[1], 120, -70);
  t.check("middle-drag pan keeps the grabbed point under the cursor",
    dist2(await v.project(P3), [s3[0] + 120, s3[1] - 70]) < 0.5);
  const s3b = await v.project(P3);
  await v.drag("left", s3b[0], s3b[1], -60, 30, { modifiers: ["shift"] });
  t.check("shift+left-drag pans", dist2(await v.project(P3), [s3b[0] - 60, s3b[1] + 30]) < 0.5);

  const sc = await v.project([0, 50, 0]);
  await v.drag("left", sc[0], sc[1], 0, -150);
  const V = [-1.75, 60, 0];
  const sv = await v.project(V);
  cam = await v.camera();
  const d0 = dist3(cam.pos, V);
  await v.wheel(sv[0], sv[1], -100, 3);
  cam = await v.camera();
  const d1 = dist3(cam.pos, V);
  t.check("wheel: moves toward the cursor point by 0.85^3", Math.abs(d1 / d0 - 0.85 ** 3) < 0.01,
    `${d0.toFixed(2)} -> ${d1.toFixed(2)} m`);
  t.check("wheel: point stays under the cursor", dist2(await v.project(V), sv) < 0.5);
  await v.wheel(sv[0], sv[1], 100, 3);
  cam = await v.camera();
  t.check("wheel out restores the distance", Math.abs(dist3(cam.pos, V) / d0 - 1) < 0.01);

  // ---- edit mode: picking and the gizmo -------------------------------------
  await v.key("b");
  const mid = [5.15, 20, 0];
  for (let i = 0; i < 15; i++) {
    const sm = await v.project(mid);
    if (dist3((await v.camera()).pos, mid) < 4) break;
    await v.wheel(sm[0], sm[1], -300, 1);
  }
  cam = await v.camera();
  t.check("zoomed close to the node pair", dist3(cam.pos, mid) < 6, `${dist3(cam.pos, mid).toFixed(2)} m`);
  await v.shot("close.png");
  await v.key("e");
  let p = await v.project(N2);
  await v.click(p[0], p[1]);
  t.check("click selects node/402", (await v.selected()) === "node/402", await v.selected());
  p = await v.project(N1);
  await v.click(p[0], p[1]);
  t.check("click on a dot under the gizmo arrow selects node/401", (await v.selected()) === "node/401",
    await v.selected());

  // Tilt so the gizmo is readable, re-select, then click on the X arrow, which
  // passes right over node/402 30 cm away: the selection must stay.
  p = await v.project(N1);
  await v.drag("left", p[0] + 200, p[1] + 150, 0, -120);
  piv = await v.pivot();
  t.check("orbit over empty ground uses a ground pivot", piv.kind === "ground", JSON.stringify(piv));
  p = await v.project(N1);
  await v.click(p[0], p[1]);
  t.check("re-select node/401", (await v.selected()) === "node/401");
  await v.shot("gizmo.png");
  {
    cam = await v.camera();
    const dN = dist3(cam.pos, N1);
    // Walk out along the X arrow (its picker reaches ~0.13 * distance) to a spot
    // 5-9 px past node/402's dot: the old 10 px node pick grabbed node/402 there.
    const dot402 = await v.project(N2);
    let onArrow = null;
    for (let dx = 0.3; dx < 0.13 * dN; dx += 0.002) {
      const q = await v.project([N1[0] + dx, N1[1], N1[2]]);
      const d = dist2(q, dot402);
      if (d > 5 && d < 9) { onArrow = q; break; }
    }
    t.check("found an arrow spot 5-9 px from node/402", onArrow !== null);
    if (onArrow) {
      await v.click(onArrow[0], onArrow[1]);
      t.check("click on a gizmo arrow near another node keeps the selection",
        (await v.selected()) === "node/401",
        `${await v.selected()} (click ${dist2(onArrow, dot402).toFixed(1)} px from node/402)`);
    }
  }

  before = await v.camera();
  v.commands.length = 0;
  p = await v.project(N1);
  await v.drag("left", p[0], p[1], 40, 0, { steps: 10 });
  await new Promise((r) => setTimeout(r, 300));
  cam = await v.camera();
  t.check("gizmo drag does not move the camera", dist3(cam.pos, before.pos) < 1e-9 && cam.yaw === before.yaw);
  const moves = v.commands.flatMap((c) => c.ops || []).filter((o) => o.op === "move_node");
  t.check("gizmo drag sends one move_node for node/401", moves.length === 1 && moves[0].id === "node/401",
    JSON.stringify(moves));
  await v.shot("after_gizmo.png");

  // ---- traffic icons --------------------------------------------------------
  // The server finds its style_images by itself: a viewer started by hand
  // (no JOSM, no --icons-dir) serves the package's icons.
  const iconResponse = await v.evaluate(
    "fetch('/style_images/274-30.png').then(r => r.status)");
  t.check("the server serves the shipped sign images", iconResponse === 200, String(iconResponse));
  const icons = () => v.evaluate("window.__ll2test.icons()");
  t.check("sign and arrow icons are built",
    await waitFor(async () => (await icons()).icons >= 2), JSON.stringify(await icons()));
  await v.key("f");
  t.check("and shown near the camera", (await icons()).shown >= 1, JSON.stringify(await icons()));

  const iconBtn = await v.evaluate("(() => { const b = document.getElementById('iconsBtn')"
    + ".getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()");
  await v.click(iconBtn[0], iconBtn[1]);
  t.check("the Icons button (real mouse click) hides them",
    !(await icons()).visible && (await icons()).shown === 0, JSON.stringify(await icons()));
  await v.key("n");
  t.check("N brings them back", (await icons()).visible);

  await v.evaluate("(() => { const el = document.getElementById('iconAlpha');"
    + " el.value = '40'; el.dispatchEvent(new Event('input')); })()");
  t.check("the alpha slider fades them so the cloud shows through",
    Math.abs((await icons()).opacity - 0.4) < 1e-9, JSON.stringify(await icons()));
  await v.shot("icons_faded.png");

  // Autoware traffic lights: an upright box on the line, `height` tall.
  const meshOf = (id) => v.evaluate(`window.__ll2test.iconMesh(${JSON.stringify(id)})`);
  t.check("the lights' icons are built", await waitFor(async () =>
    (await meshOf("way/8")) && (await meshOf("way/9")) && (await meshOf("way/10")) && (await meshOf("way/11"))));
  const aw = await meshOf("way/8");
  const zs = aw.map((p) => p[2]);
  const xs = aw.map((p) => p[0]);
  const ys2 = aw.map((p) => p[1]);
  t.check("an Autoware light stands on its line, as tall as its height tag",
    Math.abs(Math.min(...zs) - 5) < 1e-6 && Math.abs(Math.max(...zs) - 5.5) < 1e-6
    && Math.abs(Math.min(...xs) - AW_LEFT[0]) < 1e-6 && Math.abs(Math.max(...xs) - AW_RIGHT[0]) < 1e-6,
    JSON.stringify(aw.map((p) => p.map((n) => +n.toFixed(3)))));
  // Its icon face is lifted toward the approaching car (-y), not away.
  t.check("facing the car that sees start on the left and end on the right",
    ys2.every((y) => y < 30 && y > 29.9), JSON.stringify(ys2));
  const flatNoHeight = await meshOf("way/9");
  // The old path lays a horizontal 2-point light flat; that is what stays.
  t.check("without a height tag a 2-point light keeps the old icon",
    Math.max(...flatNoHeight.map((p) => p[2])) - Math.min(...flatNoHeight.map((p) => p[2])) < 1e-6,
    JSON.stringify(flatNoHeight.map((p) => p.map((n) => +n.toFixed(3)))));
  const old3 = await meshOf("way/10");
  t.check("so does a 3-point light with a height tag",
    Math.max(...old3.map((p) => p[2])) - Math.min(...old3.map((p) => p[2])) < 1e-6);
  const oldBad = await meshOf("way/11");
  t.check("and one whose height is not a number",
    Math.max(...oldBad.map((p) => p[2])) - Math.min(...oldBad.map((p) => p[2])) < 1e-6);
  t.check("the signs' icons are built", await waitFor(async () => (await meshOf("way/12")) && (await meshOf("way/13"))));
  const sign = await meshOf("way/12");
  t.check("an Autoware sign stands on its line, as tall as its height tag",
    Math.abs(Math.min(...sign.map((p) => p[2])) - 2) < 1e-6 && Math.abs(Math.max(...sign.map((p) => p[2])) - 2.6) < 1e-6
    && sign.every((p) => p[1] < 30 && p[1] > 29.9) && Math.abs(Math.min(...sign.map((p) => p[0])) + 6) < 1e-6
    && Math.abs(Math.max(...sign.map((p) => p[0])) + 5.4) < 1e-6,
    JSON.stringify(sign.map((p) => p.map((n) => +n.toFixed(3)))));
  const signNoHeight = await meshOf("way/13");
  t.check("a 2-point sign without a height tag keeps the old icon",
    Math.max(...signNoHeight.map((p) => p[2])) - Math.min(...signNoHeight.map((p) => p[2])) < 1e-6);
  await v.evaluate("(() => { const el = document.getElementById('iconAlpha');"
    + " el.value = '100'; el.dispatchEvent(new Event('input')); })()");
  await v.evaluate("window.__ll2city.lookAt([-3, 30, 4], 12)");
  await v.shot("autoware_lights.png");

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} finally {
  await v.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
