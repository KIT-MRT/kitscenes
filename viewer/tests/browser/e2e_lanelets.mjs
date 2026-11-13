// End-to-end checks of lanelet surfaces and direction arrows.
//   node viewer/tests/browser/e2e_lanelets.mjs [--shots DIR]
import { openViewer, Checks, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;

const xs = [0, 25, 50, 75, 100];
const way = (id, y, ids, reverse = false) => {
  const pts = xs.map((x) => [x, y, Z]);
  if (reverse) pts.reverse();
  return { id: `way/${id}`, kind: "line", tags: { type: "line_thin" }, pts: pts.flat(), nodes: ids };
};
// As the JOSM plugin sends lanelets: bound refs, reversal
// flags, arrow = x, y, z, dx, dy, dz, width.
const features = [
  way(1, 1.75, [11, 12, 13, 14, 15]),
  way(2, -1.75, [21, 22, 23, 24, 25]),
  way(3, 5.25, [31, 32, 33, 34, 35], true), // stored westward
  { id: "relation/100", kind: "lanelet", tags: { subtype: "road" }, pts: [35, 0, Z],
    left: "way/1", right: "way/2", lrev: false, rrev: false, two: false, arrow: [35, 0, Z, 1, 0, 0, 3.5] },
  { id: "relation/101", kind: "lanelet", tags: { subtype: "road", one_way: "no" }, pts: [35, 3.5, Z],
    left: "way/3", right: "way/1", lrev: true, rrev: false, two: true, arrow: [35, 3.5, Z, 1, 0, 0, 3.5] },
];

const v = await openViewer({ shotsDir });
const lanes = () => v.evaluate("window.__ll2test.lanelets()");
try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("scene streamed", await v.waitForFeatures(5));
  await v.key("b");
  await sleep(200);
  let l = await lanes();
  t.check("both lanelets are drawn with surfaces", l.count === 2 && l.surfaceTiles >= 1, JSON.stringify(l));
  t.check("one single-headed and one double-headed arrow", l.oneWayArrows === 1 && l.twoWayArrows === 1, JSON.stringify(l));
  await v.shot("lanelets_bev.png");

  // A one-way road that bicycles may use both ways: same arrow, violet.
  const withOverride = { ...features[3], tags: { subtype: "road", "one_way:bicycle": "no" }, owx: ["bicycle"] };
  v.sendScene({ type: "patch", ops: [{ op: "upsert", feature: withOverride }] });
  await sleep(300);
  l = await lanes();
  t.check("a one_way:<participant> override marks the arrow, keeping the car's shape",
    l.overrideArrows === 1 && l.oneWayArrows === 1 && l.twoWayArrows === 1, JSON.stringify(l));
  await v.shot("lanelets_override.png");
  v.sendScene({ type: "patch", ops: [{ op: "upsert", feature: features[3] }] });
  await sleep(300);
  t.check("and unmarks it when the override goes", (await lanes()).overrideArrows === 0);

  // Orbit from inside a lane, far from both bounds on screen (zoomed in so
  // the lane is wide): the pivot is the surface, not the ground-plane guess.
  let p = await v.project([60, 0, Z]);
  await v.wheel(p[0], p[1], -300, 4);
  p = await v.project([60, 0, Z]);
  await v.drag("left", p[0], p[1], 40, -60);
  let piv = await v.pivot();
  t.check("a press inside a lane pivots on the lanelet surface",
    piv.kind === "surface" && Math.abs(piv.point[2] - Z) < 1e-3 && Math.abs(piv.point[1]) < 0.5, JSON.stringify(piv));
  await v.shot("lanelets_oblique.png");

  // A bound moves in JOSM: the surface follows (rebuilt from the new way).
  v.sendScene({ type: "patch", ops: [{ op: "upsert", feature: way(2, -3, [21, 22, 23, 24, 25]) }] });
  await sleep(300);
  l = await lanes();
  t.check("a moved bound keeps both lanelets drawn", l.count === 2 && l.surfaceTiles >= 1);

  // The lanelet goes away: its arrow does too.
  v.sendScene({ type: "patch", ops: [{ op: "remove", id: "relation/101" }] });
  await sleep(300);
  l = await lanes();
  t.check("a removed lanelet takes its arrow along", l.count === 1 && l.twoWayArrows === 0 && l.oneWayArrows === 1,
    JSON.stringify(l));

  await v.key("l");
  l = await lanes();
  t.check("L hides the lanelet layer", l.visible === false);
  await v.key("b");
  p = await v.project([60, 0, Z]);
  await v.wheel(p[0], p[1], -300, 4);
  p = await v.project([60, 0, Z]);
  await v.drag("left", p[0], p[1], 40, -60);
  piv = await v.pivot();
  t.check("hidden surfaces are not picked", piv.kind !== "surface", JSON.stringify(piv));
  await v.key("l");
  t.check("L shows it again", (await lanes()).visible === true);

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} finally {
  await v.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
