"""Test setup: which tests need data.

Most tests build small synthetic scenes and need no data. The tests marked
``real_data`` read a dataset copy and run only with ``KITSCENES_ROOT=<root> pytest``
(read-only; every scene under the root is checked, so a few scenes per split are enough).
The tests marked ``notebooks`` execute the notebooks and run only with
``KITSCENES_NOTEBOOK_ROOT=<root> pytest -m notebooks``. Tests of optional packages
(torch, lanelet2, OpenCV, ...) are skipped where the package is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR

NO_REAL_ROOT = f"needs a dataset copy: set ${KITSCENES_ROOT_ENV_VAR}"


def _real_root() -> Optional[Path]:
    root = os.environ.get(KITSCENES_ROOT_ENV_VAR)
    return Path(root) if root and Path(root).is_dir() else None


def pytest_collection_modifyitems(items: list) -> None:
    """Skip the ``real_data`` tests when no dataset copy is given."""
    if _real_root() is None:
        for item in items:
            if item.get_closest_marker("real_data"):
                item.add_marker(pytest.mark.skip(reason=NO_REAL_ROOT))


def pytest_generate_tests(metafunc) -> None:
    """A ``real_data`` test with a ``scene_id`` argument runs once per scene of the dataset copy."""
    if "scene_id" in metafunc.fixturenames and metafunc.definition.get_closest_marker("real_data"):
        root = _real_root()
        if root is None:
            metafunc.parametrize("scene_id", [pytest.param("", marks=pytest.mark.skip(reason=NO_REAL_ROOT), id="no-root")])
        else:
            import kitscenes

            metafunc.parametrize("scene_id", kitscenes.open(root).scene_ids)


@pytest.fixture(scope="session")
def real_root() -> Path:
    """The dataset copy in ``$KITSCENES_ROOT`` (read-only)."""
    root = _real_root()
    if root is None:
        pytest.skip(NO_REAL_ROOT)
    return root
