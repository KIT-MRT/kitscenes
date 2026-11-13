"""The notebooks run from top to bottom; opt-in: ``KITSCENES_NOTEBOOK_ROOT=<root> pytest -m notebooks``.

The root must hold what the notebooks read (the README's example scene with its maps and aerial tiles).
Each notebook is executed as a copy in a temporary folder; the root is only read.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT_ENV_VAR = "KITSCENES_NOTEBOOK_ROOT"

pytestmark = pytest.mark.notebooks


@pytest.mark.parametrize("notebook", sorted((REPO / "notebooks").glob("*.ipynb")), ids=lambda path: path.stem)
def test_notebook_runs(notebook: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = os.environ.get(NOTEBOOK_ROOT_ENV_VAR)
    if not root:
        pytest.skip(f"set ${NOTEBOOK_ROOT_ENV_VAR} to a dataset copy to run the notebooks")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    # The kernel is a new process: it gets the root, this checkout and caches of its own from the environment.
    monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, root)
    monkeypatch.setenv("KITSCENES_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    copy = Path(shutil.copyfile(notebook, tmp_path / notebook.name))
    nb = nbformat.read(copy, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})
    client.execute()  # raises CellExecutionError, with the traceback, at the first cell that fails
