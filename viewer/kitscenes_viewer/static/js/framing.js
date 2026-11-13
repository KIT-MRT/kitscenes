// Framing actions (F, B, Default BEV) over the map bounds, with the numbers
// the frame-debug panel shows.
import { now, plog, PROFILE } from "./util.js";
import { store } from "./store.js";
import { frameBox, bevBox, defaultBev } from "./camera.js";
import { reportFrameDebug, setFrameDebug } from "./hud.js";

export function frameAll() {
  if (store.featureCount() === 0) return;
  const t0 = PROFILE ? now() : 0;
  const box = store.mapBounds();
  if (box.isEmpty()) {
    reportFrameDebug("frameAll (F)", box, null);
    return;
  }
  reportFrameDebug("frameAll (F)", box, frameBox(box));
  if (PROFILE) plog(`frameAll features=${store.featureCount()} ms=${(now() - t0).toFixed(1)}`);
}

// Bird's-eye view: camera straight above the map, north (+Y) toward screen top.
export function bevNorth() {
  if (store.featureCount() === 0) return;
  const t0 = PROFILE ? now() : 0;
  const box = store.mapBounds();
  if (box.isEmpty()) {
    reportFrameDebug("BEV north (B)", box, null);
    return;
  }
  reportFrameDebug("BEV north (B)", box, bevBox(box));
  if (PROFILE) plog(`bevNorth features=${store.featureCount()} ms=${(now() - t0).toFixed(1)}`);
}

export function defaultBevView() {
  const h = defaultBev();
  setFrameDebug([
    "default BEV @ origin",
    `fixed tgt (0.0, 0.0, 0.0) cam (0.0, 0.0, ${h.toFixed(1)}) m`,
    "north (+Y) toward screen top; use when F/B framing fails",
  ]);
}
