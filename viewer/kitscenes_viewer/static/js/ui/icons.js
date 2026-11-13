// Line icons, drawn on a 24 px grid and stroked with the text colour.
// Inline so the viewer needs no network and no icon font.
const PATHS = {
  layers: '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/>',
  car: '<path d="M4 16v-3l2-5h12l2 5v3"/><path d="M3 16h18v2H3z"/><circle cx="7.5" cy="18.5" r="1.5"/><circle cx="16.5" cy="18.5" r="1.5"/><path d="M6.5 12h11"/>',
  city: '<path d="M3 21h18"/><path d="M5 21V9l6-3v15"/><path d="M11 21V11l8 3v7"/><path d="M8 11v.01M8 14v.01M8 17v.01M15 16v.01M15 19v.01"/>',
  bug: '<rect x="8" y="6" width="8" height="13" rx="4"/><path d="M10 6.5 9 4M14 6.5 15 4M4 11h4M16 11h4M4 16h4M16 16h4M12 10v9"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 1 1 3.4 2.4c-.6.3-1 .8-1 1.5v.7"/><path d="M12 17.2v.01"/>',
  settings: '<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/>',
  panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="3"/>',
  photo: '<path d="M4 8h3l2-3h6l2 3h3v11H4Z"/><circle cx="12" cy="13" r="3.5"/>',
  frame: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
  top: '<path d="M12 3v11"/><path d="m8 10 4 4 4-4"/><path d="M4 17.5 12 21l8-3.5"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="m12 6 2.5 6h-5Z" class="fill"/><path d="M9.5 12 12 18l2.5-6"/>',
  mouse: '<rect x="6" y="3" width="12" height="18" rx="6"/><path d="M12 7v4"/>',
  target: '<circle cx="12" cy="12" r="6.5"/><circle cx="12" cy="12" r="1.5"/><path d="M12 2v3.5M12 18.5V22M2 12h3.5M18.5 12H22"/>',
  move: '<path d="M12 3v18M3 12h18"/><path d="m9 6 3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
  rotate: '<path d="M20 12a8 8 0 1 1-2.4-5.7"/><path d="M20 4v5h-5"/>',
  height: '<path d="M12 3v14"/><path d="m8 7 4-4 4 4"/><path d="m8 13 4 4 4-4"/><path d="M5 21h14"/>',
  interp: '<path d="M5 18 19 6"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="12" r="1.6" class="fill"/>',
  magnet: '<path d="M6 4v7a6 6 0 0 0 12 0V4"/><path d="M6 8h4M14 8h4"/><path d="M10 4v7a2 2 0 0 0 4 0V4"/>',
  keys: '<rect x="3" y="7" width="18" height="10" rx="2"/><path d="M7 10.5v.01M11 10.5v.01M15 10.5v.01M8 14h8"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M9 7V4h6v3"/><path d="m6 7 1 13h10l1-13"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H12"/>',
  redo: '<path d="m15 14 5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H12"/>',
  play: '<path d="M7 4.5v15l12-7.5Z" class="fill"/>',
  pause: '<rect x="6.5" y="5" width="3.5" height="14" rx="1" class="fill"/><rect x="14" y="5" width="3.5" height="14" rx="1" class="fill"/>',
  prev: '<path d="M18 5v14L8 12Z" class="fill"/><path d="M6 5v14"/>',
  next: '<path d="M6 5v14l10-7Z" class="fill"/><path d="M18 5v14"/>',
  follow: '<path d="M12 3 5 20l7-4 7 4Z"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  unlock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.6-1.7"/>',
  image: '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="m21 16-5-5-9 8"/>',
  tag: '<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9Z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.3-4.3L4 9"/><path d="M4 4v5h5"/><path d="M4 13a8 8 0 0 0 14.3 4.3L20 15"/><path d="M20 20v-5h-5"/>',
  cloud: '<path d="M4 15v.01M7 12v.01M10 15v.01M13 11v.01M16 14v.01M19 11v.01M6 18v.01M12 18v.01M18 17v.01M9 9v.01M15 8v.01"/>',
};

/** An icon as markup; unknown names give an empty icon. */
export function iconSvg(name) {
  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${PATHS[name] || ""}</svg>`;
}

/** Fill every `[data-icon]` under `root` with its icon. */
export function applyIcons(root = document) {
  for (const el of root.querySelectorAll("[data-icon]")) {
    if (el.dataset.iconDone === el.dataset.icon) continue;
    el.innerHTML = iconSvg(el.dataset.icon);
    el.classList.add("ic");
    el.dataset.iconDone = el.dataset.icon;
  }
}

/** Swap the icon of one `[data-icon]` element. */
export function setIcon(el, name) {
  if (!el || el.dataset.icon === name) return;
  el.dataset.icon = name;
  applyIcons(el.parentElement || document);
}
