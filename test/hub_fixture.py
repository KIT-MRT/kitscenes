"""A local stand-in for the hub: a packed release with its manifest, and downloads from it."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from kitscenes.cli import build_parser, cmd_download
from release_fixture import make_scene, pack_scene


def make_hub(tmp_path: Path) -> Path:
    """A packed release with a manifest, as the hub holds it."""
    hub = tmp_path / "hub"
    for split, name in (("train", "scene_a"), ("val", "scene_b"), ("test-e2e", "scene_c")):
        pack_scene(make_scene(hub / "data" / split / name, n=3, cameras=("camera_ring_front", "camera_ring_rear"),
                              lidars=("lidar_top",), radars=("radar_front",)))
    rows = []
    for path in sorted((hub / "data").rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(hub).as_posix()
        _, split, scene, *rest = rel.split("/")
        component = path.stem if path.suffix == ".zip" else "metadata"
        rows.append({"sequence_id": scene, "split": split, "component": component, "archive_path": rel,
                     "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "archive_size_bytes": path.stat().st_size})
    with open(hub / "data" / "sequence_archives.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return hub


def fetcher(hub: Path, calls: list):
    def download_one(f, staging: Path) -> Path:
        calls.append(f.path)
        target = staging / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hub / f.path, target)
        return target
    return download_one


def run_download(hub: Path, root: Path, argv: list[str], calls: list) -> int:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copyfile(hub / "data" / "sequence_archives.csv", root / "data" / "sequence_archives.csv")
    args = build_parser().parse_args(["download", str(root), *argv])
    return cmd_download(args, download_one=fetcher(hub, calls))


class FakeHub:
    """Two revisions of the hub folder; 'main' moves from rev-a to rev-b."""

    def __init__(self, revisions: dict, main: str) -> None:
        self.revisions, self.main, self.manifests = revisions, main, 0

    def resolve(self, revision):
        name = revision or self.main
        if name not in self.revisions:
            raise ValueError(f"no revision {name}")
        return name

    def download(self, filename, revision, local_dir: Path) -> Path:
        self.manifests += filename.endswith(".csv")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.revisions[revision or self.main] / filename, target)
        return target


def versioned_hub(tmp_path: Path) -> FakeHub:
    a = make_hub(tmp_path / "a")
    b = tmp_path / "b" / "hub"
    shutil.copytree(a, b)
    # rev-b changes one archive (and its manifest row)
    changed = b / "data/train/scene_a/lidar_top.zip"
    changed.write_bytes(changed.read_bytes() + b"\0")
    rows = list(csv.DictReader(open(b / "data/sequence_archives.csv")))
    for row in rows:
        if row["archive_path"] == "data/train/scene_a/lidar_top.zip":
            row["archive_sha256"] = hashlib.sha256(changed.read_bytes()).hexdigest()
            row["archive_size_bytes"] = changed.stat().st_size
    with open(b / "data/sequence_archives.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return FakeHub({"rev-a": a, "rev-b": b}, main="rev-a")


def hub_download(hub: FakeHub, root: Path, argv: list[str]) -> int:
    args = build_parser().parse_args(["download", str(root), *argv])
    return cmd_download(args, hub=hub)
