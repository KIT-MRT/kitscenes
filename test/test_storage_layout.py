"""DirStorage: a scene folder as storage."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from kitscenes._storage import DirStorage, Storage


def test_dir_storage(tmp_path: Path) -> None:
    (tmp_path / "lidar_top").mkdir()
    (tmp_path / "lidar_top" / "0000000001.parquet").write_bytes(b"b")
    (tmp_path / "lidar_top" / "0000000000.parquet").write_bytes(b"a")
    (tmp_path / "lidar_top" / "sub").mkdir()
    storage = DirStorage(tmp_path)
    assert isinstance(storage, Storage)
    assert storage.list("lidar_top") == ["lidar_top/0000000000.parquet", "lidar_top/0000000001.parquet"]
    assert storage.list("missing") == []
    assert storage.read_bytes("lidar_top/0000000001.parquet") == b"b"
    assert storage.exists("lidar_top/0000000000.parquet") and not storage.exists("lidar_top")
    assert storage.is_dir("lidar_top") and not storage.is_dir("nope")
    with pytest.raises(FileNotFoundError):
        storage.read_bytes("lidar_top/sub")
    with pytest.raises(FileNotFoundError):
        storage.read_bytes("nope")
    assert pickle.loads(pickle.dumps(storage)) == storage
