"""What the modules of kitscenes.visualization share: OpenCV and matplotlib (the ``viz`` extra), the output guard."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Iterable

from kitscenes._optional import require
from kitscenes.constants import KITSCENES_ROOT_ENV_VAR


def cv2() -> ModuleType:
    return require("cv2", "viz", package="opencv-python-headless", purpose="kitscenes.visualization")


def matplotlib(submodule: str = "") -> ModuleType:
    """``matplotlib`` or one of its submodules (``"pyplot"``, ``"colors"``, ``"lines"``)."""
    name = f"matplotlib.{submodule}" if submodule else "matplotlib"
    return require(name, "viz", package="matplotlib", purpose="kitscenes.visualization")


def guard_output_path(path: Path | str, roots: Iterable[Path | str] = ()) -> Path:
    """*path* resolved.

    Raises:
        ValueError: If it lies under ``$KITSCENES_ROOT`` or one of *roots* (e.g. the dataset read).
    """
    resolved = Path(path).expanduser().resolve()
    dataset_root = os.environ.get(KITSCENES_ROOT_ENV_VAR)
    for candidate in ([dataset_root] if dataset_root else []) + list(roots):
        root = Path(candidate).expanduser().resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write inside the dataset root ({root}). Choose a path outside it.")
    return resolved
