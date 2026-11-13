// The Settings dialog: things set once rather than toggled while working.
// Everything is kept per browser with the other interface preferences.
import { prefs, savePrefs, resetPrefs, setTabAvailable, showTab } from "./shell.js";
import { setWalkFactor } from "../keys.js";
import { setViewSelectsNodes } from "../edit.js";
import { toast } from "../hud.js";

const $ = (id) => document.getElementById(id);
const walkLabel = (exp) => `${Number((2 ** exp).toFixed(2))}×`;

function apply() {
  setWalkFactor(2 ** (Number(prefs.walkExp) || 0));
  setTabAvailable("debug", !!prefs.debug);
  setViewSelectsNodes(!!prefs.viewNodes);
  if ($("setViewNodes")) $("setViewNodes").checked = !!prefs.viewNodes;
  if ($("setWalkSpeed")) $("setWalkSpeed").value = String(prefs.walkExp || 0);
  if ($("setWalkSpeedLabel")) $("setWalkSpeedLabel").textContent = walkLabel(Number(prefs.walkExp) || 0);
  if ($("setHints")) $("setHints").checked = !!prefs.hints;
  if ($("setDebug")) $("setDebug").checked = !!prefs.debug;
  if ($("setTooltips")) $("setTooltips").checked = !!prefs.tooltips;
}

export function installSettings() {
  $("setWalkSpeed")?.addEventListener("input", (e) => {
    prefs.walkExp = Number(e.target.value) || 0;
    savePrefs();
    apply();
  });
  $("setHints")?.addEventListener("change", (e) => {
    prefs.hints = e.target.checked;
    savePrefs();
  });
  $("setDebug")?.addEventListener("change", (e) => {
    prefs.debug = e.target.checked;
    savePrefs();
    apply();
    if (prefs.debug) showTab("debug");
  });
  $("setViewNodes")?.addEventListener("change", (e) => {
    prefs.viewNodes = e.target.checked;
    savePrefs();
    apply();
  });
  $("setTooltips")?.addEventListener("change", (e) => {
    prefs.tooltips = e.target.checked;
    savePrefs();
  });
  $("setReset")?.addEventListener("click", () => {
    resetPrefs();
    apply();
    toast("Interface reset to its defaults", "info", 2000);
  });
  apply();
}
