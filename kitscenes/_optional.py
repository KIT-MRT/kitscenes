"""Import an optional dependency, or fail with the extra that installs it."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from kitscenes.errors import MissingDependencyError


def require(module: str, extra: str, *, package: str = "", purpose: str = "") -> ModuleType:
    """The module *module*, imported.

    Args:
        module: Import name (``"cv2"``, ``"matplotlib.pyplot"``).
        extra: The kitscenes extra that installs it (``"viz"``).
        package: Distribution name, if not the import name's first part.
        purpose: What needs it (``"kitscenes.visualization"``), for the message.

    Raises:
        MissingDependencyError: If it is not installed, with the install command.
    """
    try:
        return import_module(module)
    except ImportError as exc:
        name = package or module.split(".", 1)[0]
        raise MissingDependencyError(
            f"{purpose or 'This'} needs {name} ({exc}). "
            f'Install it with: pip install "kitscenes[{extra}]"'
        ) from exc


#: Oldest torch for the torch features of kitscenes (``Tensor.scatter_reduce_``, used by the GPU voxel maps).
TORCH_MIN_VERSION = "1.13"


def require_torch(purpose: str, min_version: str = TORCH_MIN_VERSION) -> ModuleType:
    """torch, imported and at least *min_version* (default :data:`TORCH_MIN_VERSION`).

    Raises:
        MissingDependencyError: If torch is missing or older, with what to install.
    """
    torch = require("torch", "torch", purpose=purpose)
    found = tuple(int(p) for p in torch.__version__.split("+", 1)[0].split(".")[:2] if p.isdigit())
    if found < tuple(int(p) for p in min_version.split(".")):
        raise MissingDependencyError(
            f"{purpose} needs torch>={min_version}; torch {torch.__version__} is installed. "
            "Install a newer torch for your CUDA from https://pytorch.org/get-started/locally/"
        )
    return torch
