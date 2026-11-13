// Shared flags and small helpers for the viewer modules.

const params = new URLSearchParams(location.search);

// Profiling is opt-in via ?profile=1. When on, the browser logs a per-stage
// timing breakdown for every message and shows live render stats, so
// large-map slowness can be attributed to a stage.
export const PROFILE = params.get("profile") === "1";

// ?test=1 exposes window.__ll2test for the headless E2E harness
// (viewer/tests/browser/): read-only views of camera, projection and selection.
export const TEST_HOOKS = params.get("test") === "1";

export const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export function plog(...args) {
  if (!PROFILE) return;
  console.log("[viewer][perf]", ...args);
  // Kept for the headless perf script (viewer/tests/browser/perf_viewer.mjs).
  (window.__perfLog || (window.__perfLog = [])).push(args.join(" "));
}

export const round3 = (v) => Math.round(v * 1000) / 1000;

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Minimal event emitter. `on` returns an unsubscribe function. */
export class Emitter {
  constructor() {
    this._handlers = new Map();
  }

  on(name, fn) {
    let hs = this._handlers.get(name);
    if (!hs) {
      hs = new Set();
      this._handlers.set(name, hs);
    }
    hs.add(fn);
    return () => hs.delete(fn);
  }

  emit(name, ...args) {
    const hs = this._handlers.get(name);
    if (!hs) return;
    for (const fn of hs) fn(...args);
  }
}
