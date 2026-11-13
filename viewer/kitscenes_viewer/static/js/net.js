// Server connection: the SSE scene stream in, commands to JOSM out.
import { now, PROFILE } from "./util.js";

const RESULT_TIMEOUT_MS = 8000;
let nextId = 1;
const pending = new Map(); // command id -> resolve(result)

/**
 * Subscribe to /events (EventSource reconnects on its own).
 * onMessage(msg, meta) gets each parsed message; meta carries parse timing
 * when profiling. onStatus(live, text) reflects the connection.
 */
export function connect(onMessage, onStatus) {
  onStatus(false, "connecting...");
  const es = new EventSource("/events");
  es.onopen = () => onStatus(true, "live");
  es.onmessage = (ev) => {
    let msg;
    const tp = PROFILE ? now() : 0;
    try { msg = JSON.parse(ev.data); } catch (_) { return; }
    const meta = PROFILE ? { parseMs: now() - tp, bytes: ev.data.length } : null;
    onMessage(msg, meta);
  };
  es.onerror = () => onStatus(false, "reconnecting...");
  return es;
}

/** Feed a `command_result` from JOSM to the send waiting for it. */
export function onCommandResult(msg) {
  const resolve = pending.get(msg.id);
  if (!resolve) return;
  pending.delete(msg.id);
  resolve({ ok: !!msg.ok, message: msg.message || "", warning: msg.warning || null,
            details: Array.isArray(msg.details) ? msg.details : null });
}

/**
 * POST ops to /command; the server forwards them to every connected JOSM
 * bridge. Resolves to { delivered, error?, result? } where `result` is JOSM's
 * { ok, message, warning } reply (only with `awaitResult`; { timeout: true } if none
 * came). `delivered` is 0 when no JOSM is connected.
 */
export function sendCommand(ops, { awaitResult = false, timeoutMs = RESULT_TIMEOUT_MS } = {}) {
  const id = awaitResult ? `c${nextId++}` : undefined;
  const body = { type: "command", ops };
  if (id) body.id = id;
  const reply = id ? new Promise((resolve) => {
    pending.set(id, resolve);
    setTimeout(() => {
      if (pending.delete(id)) resolve({ timeout: true });
    }, timeoutMs);
  }) : null;
  return fetch("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }).then(async (info) => {
    if (!reply) return info;
    if (!info.delivered) {
      pending.delete(id);
      return info;
    }
    return { ...info, result: await reply };
  }).catch((err) => {
    if (id) pending.delete(id);
    console.warn("[viewer] command failed", err);
    return { ok: false, delivered: 0, error: String(err && err.message ? err.message : err) };
  });
}
