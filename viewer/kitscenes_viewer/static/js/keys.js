// Keyboard and the view buttons, plus FPS (captured-mouse) look.
//
// Gaming conventions in every mode: WASD / arrows walk, Space / C up / down,
// Shift fast, alt+left/right turn, alt+up/down up / down. In edit mode with
// "keys: selection" (T) the same keys move and turn the selection instead of
// the camera. FPS mode only adds the captured mouse and turns edit mode off.
//
// Hold-to-move keys are only recorded here; applyHeldCamera() acts in the
// render loop, so mouse look under pointer lock cannot starve WASD of
// key-repeat events.
import { canvas } from "./scene.js";
import { walk, elevate, look } from "./camera.js";
import { LOOK_RAD_PER_PX, endNav } from "./nav.js";
import { edit, setEditMode, editEvents, handleEditKey, keysMoveSelection, nudgeSelection } from "./edit.js";
import { frameAll, bevNorth, defaultBevView } from "./framing.js";
import { requestJosmRecenter } from "./josmview.js";
import { setLookHud } from "./hud.js";
import { laneletLayer } from "./render/lanelets.js";
import { iconLayer } from "./render/icons.js";
import { handleSceneKey } from "./sensors/scene.js";
import { handleUiKey } from "./ui/shell.js";

const CAM_PAN_STEP_M = 1;    // metres per pan / height tick (walk)
const CAM_FAST_STEP_M = 10;  // metres per tick with Shift (WASD / arrows / pad)
const CAM_ROT_STEP = 0.05;   // radians per step (~3°)
const CAM_HOLD_MS = 50;

const moveKeys = new Set();
let altHeld = false;
let ctrlHeld = false;
let shiftHeld = false;
let fpsLookEnabled = false;
let walkFactor = 1;          // Settings: walk speed

const moveStepM = () => (shiftHeld ? CAM_FAST_STEP_M : CAM_PAN_STEP_M) * walkFactor;

/** Scale how far the movement keys go per tick (Settings). */
export function setWalkFactor(f) {
  walkFactor = Number.isFinite(f) && f > 0 ? f : 1;
}

export const isPointerLocked = () => document.pointerLockElement === canvas;

/** Held keys as intents: right / forward / up / turn in -1..1. */
function heldIntents() {
  let right = 0, forward = 0, up = 0, turn = 0;
  if (moveKeys.has("KeyW")) forward += 1;
  if (moveKeys.has("KeyS")) forward -= 1;
  if (moveKeys.has("KeyA")) right -= 1;
  if (moveKeys.has("KeyD")) right += 1;
  if (moveKeys.has("Space")) up += 1;
  if (moveKeys.has("KeyC")) up -= 1;
  if (moveKeys.has("ArrowUp")) {
    if (altHeld) up += 1;
    else if (!ctrlHeld) forward += 1;
  }
  if (moveKeys.has("ArrowDown")) {
    if (altHeld) up -= 1;
    else if (!ctrlHeld) forward -= 1;
  }
  if (moveKeys.has("ArrowLeft") && !ctrlHeld) {
    if (altHeld) turn += 1;
    else right -= 1;
  }
  if (moveKeys.has("ArrowRight") && !ctrlHeld) {
    if (altHeld) turn -= 1;
    else right += 1;
  }
  const len = Math.hypot(right, forward);
  if (len > 1) {
    right /= len;
    forward /= len;
  }
  return { right, forward, up: Math.sign(up), turn: Math.sign(turn) };
}

/** Move the camera or, in "keys: selection", the selection. */
function applyIntents(intents, dt) {
  if (keysMoveSelection()) {
    nudgeSelection(intents, dt, shiftHeld);
    // Height-only moves the selection vertically only; walking still moves
    // the camera, so one can go around while adjusting heights.
    if (!edit.heightOnly) return;
    intents = { ...intents, up: 0, turn: 0 };
  }
  const dist = moveStepM() * (dt / (CAM_HOLD_MS / 1000));
  if (intents.right || intents.forward) walk(intents.right, intents.forward, dist);
  if (intents.up) elevate(intents.up, dist);
  if (intents.turn) look(-intents.turn * (dt / (CAM_HOLD_MS / 1000)) * CAM_ROT_STEP, 0);
}

/** Per frame: apply held movement keys, scaled to frame time. */
export function applyHeldCamera(dt) {
  if (!moveKeys.size || dt <= 0) return;
  applyIntents(heldIntents(), dt);
}

function refreshLookHud() {
  if (isPointerLocked()) setLookHud("FPS (mouse captured)");
  else if (fpsLookEnabled) setLookHud("FPS (click view)");
  else setLookHud("orbit at cursor");
}

function setFpsLook(on) {
  fpsLookEnabled = !!on;
  const btn = document.getElementById("fpsBtn");
  if (btn) btn.classList.toggle("on", fpsLookEnabled);
  if (!fpsLookEnabled && isPointerLocked()) document.exitPointerLock();
  if (fpsLookEnabled && edit.on) setEditMode(false);
  refreshLookHud();
}

function requestFpsLock() {
  if (!fpsLookEnabled || edit.on || isPointerLocked()) return;
  canvas.requestPointerLock();
}

function isTypingTarget(el) {
  if (!el || el === document.body || el === document.documentElement) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

export function installKeys() {
  editEvents.on("mode", (on) => { if (on && fpsLookEnabled) setFpsLook(false); });

  document.addEventListener("pointerlockchange", () => {
    if (isPointerLocked()) endNav(); // captured mouse look replaces any drag
    refreshLookHud();
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!isPointerLocked()) return;
    look(e.movementX * LOOK_RAD_PER_PX, -e.movementY * LOOK_RAD_PER_PX);
  });
  canvas.addEventListener("click", () => {
    if (fpsLookEnabled && !edit.on && !isPointerLocked()) requestFpsLock();
  });

  const on = (id, fn) => {
    const b = document.getElementById(id);
    if (b) b.addEventListener("click", fn);
  };
  on("fpsBtn", () => {
    const next = !fpsLookEnabled;
    setFpsLook(next);
    if (next) requestFpsLock();
  });
  on("recenterJosmBtn", () => requestJosmRecenter("button", true));
  on("bevBtn", () => bevNorth());
  on("topViewBtn", () => bevNorth());
  on("frameBtn", () => frameAll());
  on("frameAllBtn", () => frameAll());
  on("defaultBevBtn", () => defaultBevView());
  const lanesBtn = document.getElementById("lanesBtn");
  const toggleLanes = () => {
    laneletLayer.setVisible(!laneletLayer.visible);
    if (lanesBtn) lanesBtn.classList.toggle("on", laneletLayer.visible);
  };
  if (lanesBtn) {
    lanesBtn.classList.toggle("on", laneletLayer.visible);
    lanesBtn.addEventListener("click", toggleLanes);
  }
  const iconsBtn = document.getElementById("iconsBtn");
  const iconAlpha = document.getElementById("iconAlpha");
  if (iconsBtn) {
    iconsBtn.classList.toggle("on", iconLayer.visible);
    iconsBtn.addEventListener("click", () => {
      iconLayer.setVisible(!iconLayer.visible);
      iconsBtn.classList.toggle("on", iconLayer.visible);
      if (iconAlpha) iconAlpha.disabled = !iconLayer.visible;
    });
  }
  if (iconAlpha) {
    iconLayer.setOpacity(Number(iconAlpha.value) / 100);
    iconAlpha.addEventListener("input", () => iconLayer.setOpacity(Number(iconAlpha.value) / 100));
  }
  refreshLookHud();

  // Modifier state comes from every key event, not only from the modifier's
  // own keydown, which is missed when it was pressed outside the window.
  const trackModifiers = (e) => {
    altHeld = e.altKey;
    ctrlHeld = e.ctrlKey || e.metaKey;
    shiftHeld = e.shiftKey;
  };

  document.addEventListener("keydown", (e) => {
    trackModifiers(e);
    // Menus, dialogs and the window shortcuts first; an open dialog keeps
    // every key from the map.
    if (handleUiKey(e)) return;
    if (isTypingTarget(e.target)) return;
    if (handleEditKey(e)) return;
    if (!e.repeat && handleSceneKey(e)) return;

    if ((e.key === "f" || e.key === "F") && !e.repeat) {
      frameAll();
      return;
    }
    if ((e.key === "b" || e.key === "B") && !e.repeat) {
      bevNorth();
      return;
    }
    if ((e.key === "v" || e.key === "V") && !e.repeat && !e.ctrlKey && !e.metaKey) {
      document.getElementById("orthoBtn")?.click();
      return;
    }
    if ((e.key === "l" || e.key === "L") && !e.repeat && !e.ctrlKey && !e.metaKey) {
      document.getElementById("lanesBtn")?.click();
      return;
    }
    if ((e.key === "n" || e.key === "N") && !e.repeat && !e.ctrlKey && !e.metaKey) {
      document.getElementById("iconsBtn")?.click();
      return;
    }
    if ((e.key === "e" || e.key === "E") && !e.repeat && !e.ctrlKey && !e.metaKey) {
      setEditMode(!edit.on);
      return;
    }
    if ((e.code === "KeyW" || e.code === "KeyA" || e.code === "KeyS" || e.code === "KeyD")
        && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      moveKeys.add(e.code);
      return;
    }
    if ((e.code === "Space" || e.code === "KeyC") && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      moveKeys.add(e.code);
      return;
    }
    if (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const dx = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
        const dy = e.key === "ArrowUp" ? 1 : e.key === "ArrowDown" ? -1 : 0;
        look(dx * CAM_ROT_STEP, dy * CAM_ROT_STEP);
      } else {
        moveKeys.add(e.code);
      }
    }
  }, true);

  document.addEventListener("keyup", (e) => {
    trackModifiers(e);
    moveKeys.delete(e.code);
  }, true);

  window.addEventListener("blur", () => {
    moveKeys.clear();
    altHeld = false;
    ctrlHeld = false;
    shiftHeld = false;
  });
}
