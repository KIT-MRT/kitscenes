"""The release archive format (format version 1): one zip per component, loose metadata.

A scene on the hub is the folder ``data/<split>/<scene_id>/`` holding

- **loose metadata**: ``timestamp.reference.txt``, ``poses.txt``,
  ``calibration/calib.json``, ``calibration/image_corrections.json`` (where the
  scene ships one), ``maps/map.osm`` and ``maps/origin.json`` (where the split
  ships maps), and ``index.parquet``;
- **one archive per component**, ``<component>.zip``:

  ===============================  =============================================
  component                        scene folders it holds
  ===============================  =============================================
  ``camera_*`` (9), ``lidar_*`` (7)  the sensor folder of the same name
  ``radar``                        ``radar_front``, ``radar_left``, ``radar_right``
  ``gnss``, ``gnss_ins``             the folder of the same name (one
                                   ``<topic>.parquet`` per message topic)
  ``processed_<product>``          ``processed/<product>`` (``ground_segmentation``,
                                   ``seamseg_instances``)
  ===============================  =============================================

Archives are uncompressed (``ZIP_STORED``) and deterministic (sorted members,
fixed dates and permissions), so unchanged data repacks to identical bytes.
In the camera and lidar archives (:data:`ALIGNED`) each member's data starts
at a multiple of :data:`ALIGN` bytes (padding in the local header's extra
field, id ``0xD935``), which NVIDIA DALI needs and which keeps reads
page-aligned. The archives of small files are not padded. Member names are ``<scene_id>/<path>``, so
extracting an archive into ``data/<split>/`` restores the scene folder.

``index.parquet`` has one row per archive member: ``component``, ``member``
(the path inside the scene folder, e.g. ``camera_ring_front/0000000000.jpg``),
``data_offset`` and ``size`` (the exact byte range in ``<component>.zip``)
and ``crc32``. A reader opens a scene with one small read and then reads every
member with one exact range, locally or over HTTP. It lists only what the
scene ships, so it never reveals data a split withholds.

The packer is :mod:`kitscenes.pack` (``kitscenes pack``).
"""

from __future__ import annotations

import dataclasses
import io
from typing import Optional, Sequence

import numpy as np

from kitscenes.constants import CAMERA_NAMES, LIDAR_NAMES, RADAR_NAMES

FORMAT_VERSION = 1
ALIGN = 4096
INDEX_FILE = "index.parquet"
ARCHIVE_SUFFIX = ".zip"
LOOSE_FILES = ("timestamp.reference.txt", "poses.txt", "calibration/calib.json", "calibration/image_corrections.json",
               "maps/map.osm", "maps/origin.json")
PROCESSED_PRODUCTS = ("ground_segmentation", "seamseg_instances")

#: Components whose member data is aligned to :data:`ALIGN`.
ALIGNED = frozenset(CAMERA_NAMES + LIDAR_NAMES)

#: Component -> the scene folders it holds.
COMPONENTS: dict[str, tuple[str, ...]] = {
    **{name: (name,) for name in CAMERA_NAMES + LIDAR_NAMES},
    "radar": RADAR_NAMES,
    "gnss": ("gnss",),
    "gnss_ins": ("gnss_ins",),
    **{f"processed_{p}": (f"processed/{p}",) for p in PROCESSED_PRODUCTS},
}

_FOLDER_TO_COMPONENT = {folder: comp for comp, folders in COMPONENTS.items() for folder in folders}


def component_of(member: str) -> Optional[str]:
    """The component holding a scene-relative path, None for loose files and unknown paths."""
    parts = member.split("/")
    for depth in (2, 1):
        if len(parts) > depth:
            comp = _FOLDER_TO_COMPONENT.get("/".join(parts[:depth]))
            if comp is not None:
                return comp
    return None


@dataclasses.dataclass(frozen=True)
class _ComponentIndex:
    """The members of one component: path -> (offset, size, crc32), folder -> files, all folders."""

    members: dict[str, tuple[int, int, int]]
    children: dict[str, list[str]]
    folders: frozenset


class ArchiveIndex:
    """The parsed ``index.parquet`` of one scene.

    Only the component column is decoded up front; a component's members are
    indexed on first use.

    Attributes:
        format_version: From the file's metadata.
        components: The components the scene ships.
    """

    def __init__(self, table, format_version: int) -> None:
        self.format_version = format_version
        self._table = table
        encoded = table.column("component").combine_chunks().dictionary_encode()
        self._codes = encoded.indices.to_numpy(zero_copy_only=False)
        names = encoded.dictionary.to_pylist()
        self._code = {name: code for code, name in enumerate(names)}
        self.components = frozenset(names[c] for c in np.unique(self._codes))
        self._parts: dict[str, _ComponentIndex] = {}

    def __repr__(self) -> str:
        return f"ArchiveIndex(members={len(self._codes)}, components={sorted(self.components)})"

    def __len__(self) -> int:
        return len(self._codes)

    @classmethod
    def from_bytes(cls, data: bytes, overlays: Sequence[bytes] = ()) -> "ArchiveIndex":
        """Parse ``index.parquet``; each of *overlays* (index rows of a re-written component, e.g. a transcoded
        camera's ``.transcoded/<camera>.parquet``) replaces the rows of its component."""
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        # ParquetFile.read without threads: ~3x faster than read_table for a file this small.
        table = pq.ParquetFile(io.BytesIO(data)).read(use_threads=False)
        meta = table.schema.metadata or {}
        version = int(meta.get(b"kitscenes_archive_format", b"1"))
        if version > FORMAT_VERSION:
            raise ValueError(f"index.parquet has archive format {version}; this kitscenes reads up to {FORMAT_VERSION}.")
        if overlays:
            parts = [pq.ParquetFile(io.BytesIO(o)).read(use_threads=False) for o in overlays]
            replaced = pa.array(sorted({c for p in parts for c in p.column("component").to_pylist()}), pa.string())
            keep = table.filter(pc.invert(pc.is_in(table.column("component"), value_set=replaced)))
            table = pa.concat_tables([keep.replace_schema_metadata(None), *(p.cast(keep.schema).replace_schema_metadata(None)
                                                                             for p in parts)])
        return cls(table, version)

    def _part(self, component: str) -> _ComponentIndex:
        part = self._parts.get(component)
        if part is None:
            rows = np.flatnonzero(self._codes == self._code[component])
            sub = self._table.take(rows)
            names = sub.column("member").to_pylist()
            values = zip(*(sub.column(c).to_numpy().tolist() for c in ("data_offset", "size", "crc32")))
            members = dict(zip(names, values))
            children: dict[str, list[str]] = {}
            folders = set()
            for name in names:
                parent = name.rpartition("/")[0]
                children.setdefault(parent, []).append(name)
            for parent in children:
                while parent and parent not in folders:
                    folders.add(parent)
                    parent = parent.rpartition("/")[0]
            part = _ComponentIndex(members, children, frozenset(folders))
            self._parts[component] = part  # a race builds it twice, harmlessly
        return part

    def _components_under(self, folder: str) -> list[str]:
        """Shipped components that may hold members in *folder* (or below it)."""
        component = component_of(folder + "/_")
        if component is not None:
            return [component] if component in self.components else []
        prefix = folder + "/" if folder else ""
        return [c for c in sorted(self.components) if any(f.startswith(prefix) for f in COMPONENTS.get(c, ()))]

    def entry(self, member: str) -> Optional[tuple[str, int, int, int]]:
        """``(component, data offset, size, crc32)`` of a scene-relative path, None if the index has no such member."""
        component = component_of(member)
        if component is None or component not in self.components:
            return None
        found = self._part(component).members.get(member)
        return None if found is None else (component, *found)

    def children(self, folder: str) -> list[str]:
        """The members directly in *folder*, sorted."""
        folder = folder.rstrip("/")
        if not folder:  # the scene folder holds only loose files, which are not members
            return []
        out = []
        for component in self._components_under(folder):
            out += self._part(component).children.get(folder, ())
        return sorted(out)

    def folder_components(self, folder: str) -> frozenset:
        """The components with members in *folder* or below it (empty: no such folder)."""
        folder = folder.rstrip("/")
        if not folder:
            return frozenset()
        prefix = folder + "/"
        found = []
        for component in self._components_under(folder):
            homes = COMPONENTS.get(component, ())
            # A shipped component has members, so a folder that is (or holds) all of its folders
            # has some of them: no need to index its members.
            if homes and all(h == folder or h.startswith(prefix) for h in homes):
                found.append(component)
            elif folder in self._part(component).folders:
                found.append(component)
        return frozenset(found)


def write_index(rows: list[tuple[str, str, int, int, int]]) -> bytes:
    """Serialize ``(component, member, data_offset, size, crc32)`` rows as ``index.parquet`` bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    table = pa.table({
        "component": pa.array([r[0] for r in rows], pa.string()),
        "member": pa.array([r[1] for r in rows], pa.string()),
        "data_offset": pa.array(np.array([r[2] for r in rows], np.int64)),
        "size": pa.array(np.array([r[3] for r in rows], np.int64)),
        "crc32": pa.array(np.array([r[4] for r in rows], np.uint32)),
    }).replace_schema_metadata({"kitscenes_archive_format": str(FORMAT_VERSION)})
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()
