"""Writing the release archive format (:mod:`kitscenes._archive`): aligned, deterministic zips and ``index.parquet``.

This is the packer the release was built with (the dataset pipeline adds
the release rules: split placement and the content per split). Here it
packs any scene folder, e.g.
to turn an extracted copy back into archives::

    from kitscenes.pack import pack_scene
    pack_scene("/data/extracted/train/<scene>", "/data/packed/data/train/<scene>")

or from the command line ``kitscenes pack SRC --out DST`` (``--help``).

The format, in short: one stored (uncompressed) zip per component, members
``<scene_id>/<path>`` in sorted order with a directory entry for every
folder, fixed dates and permissions, so the same files always give the
same bytes (and SHA-256). In the camera and lidar archives every member's
data starts at a multiple of 4096 bytes: the padding sits in the local
header's extra field (id ``0xD935``), the central directory carries none.
``index.parquet`` lists every member's exact byte range. Repacking an
unmodified extracted component reproduces the release's zip byte for byte
(same CPython ``zipfile``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from kitscenes._archive import ALIGN, ALIGNED, ARCHIVE_SUFFIX, COMPONENTS, INDEX_FILE, LOOSE_FILES, write_index

__all__ = [
    "ArchiveCheckFailed", "UnknownContent", "ZipMember", "PackedFile",
    "write_aligned_zip", "check_zip", "classify", "pack_component", "pack_scene",
]

#: A member's content: a file, or bytes.
Source = Union[Path, str, bytes, bytearray, memoryview]

_DATE = (1980, 1, 1, 0, 0, 0)
_FILE_ATTR = 0o100644 << 16
_DIR_ATTR = (0o40755 << 16) | 0x10
_PAD_ID = 0xD935
_CHUNK = 8 * 1024 * 1024


class UnknownContent(ValueError):
    """The scene folder holds files the format does not know (e.g. a stray ``processed/<drive>_deskewed/``)."""


class ArchiveCheckFailed(RuntimeError):
    """A written archive does not hold exactly what was packed."""


@dataclasses.dataclass(frozen=True)
class ZipMember:
    """One file in a written zip: its name and the exact byte range of its data."""

    name: str
    data_offset: int
    size: int
    crc32: int


@dataclasses.dataclass(frozen=True)
class PackedFile:
    """A file :func:`pack_scene` wrote: path relative to the scene folder, component (``metadata`` for loose
    files), SHA-256 and size (a manifest row without the split and scene)."""

    path: str
    component: str
    sha256: str
    size: int


def _zipinfo(name: str, attr: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=_DATE)
    zi.compress_type = zipfile.ZIP_STORED
    zi.create_system = 3
    zi.external_attr = attr
    return zi


def _pad_extra(data_offset: int, align: int) -> bytes:
    pad = (-data_offset) % align
    if pad == 0:
        return b""
    while pad < 4:  # an extra field needs at least id + length
        pad += align
    return struct.pack("<HH", _PAD_ID, pad - 4) + b"\0" * (pad - 4)


def _parents(name: str) -> list[str]:
    """The folder entries of a member name, outermost first (``a/``, ``a/b/`` for ``a/b/c``)."""
    parts = name.split("/")[:-1]
    return ["/".join(parts[:i]) + "/" for i in range(1, len(parts) + 1)]


def write_aligned_zip(entries: Iterable[tuple[str, Source]], zip_path: Path | str, align: int = ALIGN) -> list[ZipMember]:
    """Write ``(member name, content)`` pairs as a stored, deterministic zip, member data aligned to *align*.

    *entries* must come sorted by name (they are written as they come, so a
    generator can produce the content on the fly, e.g. transcoded images).
    A directory entry is added before the first member of every folder,
    which gives the order of sorting folders and files together. *align*
    1 writes no padding (the small-file archives).

    Returns:
        The members with their data offsets, read back from the written headers.

    Raises:
        ValueError: If the names are not sorted and unique.
        ArchiveCheckFailed: If a member's data did not land aligned.
    """
    zip_path = Path(zip_path)
    written: set[str] = set()
    last = None
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, content in entries:
            if last is not None and name <= last:
                raise ValueError(f"{zip_path}: members must be sorted and unique ({last!r} before {name!r})")
            last = name
            for folder in _parents(name):
                if folder not in written:
                    zf.writestr(_zipinfo(folder, _DIR_ATTR), b"")
                    written.add(folder)
            zi = _zipinfo(name, _FILE_ATTR)
            is_file = isinstance(content, (str, Path))
            zi.file_size = Path(content).stat().st_size if is_file else len(content)
            zip64 = zi.file_size * 1.05 > zipfile.ZIP64_LIMIT
            zi.extra = b""
            zi.CRC = 0
            zi.compress_size = 0
            base = len(zi.FileHeader(zip64))
            zi.extra = _pad_extra(zf.fp.tell() + base, align) if align > 1 else b""
            with zf.open(zi, "w", force_zip64=zip64) as dst:
                if is_file:
                    with open(content, "rb") as src:
                        shutil.copyfileobj(src, dst, _CHUNK)
                else:
                    dst.write(content)
            zi.extra = b""  # keep the central directory free of padding
    members = read_members(zip_path)
    if align > 1:
        bad = [m.name for m in members if m.data_offset % align]
        if bad:
            raise ArchiveCheckFailed(f"{zip_path}: {len(bad)} member(s) not aligned to {align}, e.g. {bad[0]}")
    return members


def read_members(zip_path: Path | str) -> list[ZipMember]:
    """The file members of a zip with the offsets of their data (from the local headers)."""
    out = []
    with zipfile.ZipFile(zip_path) as zf, open(zip_path, "rb") as fh:
        for info in zf.infolist():
            if info.is_dir():
                continue
            fh.seek(info.header_offset + 26)
            name_len, extra_len = struct.unpack("<HH", fh.read(4))
            out.append(ZipMember(info.filename, info.header_offset + 30 + name_len + extra_len, info.file_size, info.CRC))
    return out


def check_zip(zip_path: Path | str, members: Sequence[ZipMember]) -> str:
    """SHA-256 of a written zip, and in the same read the CRC-32 of every member's data against *members*.

    Raises:
        ArchiveCheckFailed: If a member's bytes on disk do not match its CRC-32.
    """
    h = hashlib.sha256()
    spans = sorted((m.data_offset, m.data_offset + m.size, m.name, m.crc32) for m in members)
    crcs = {name: 0 for _, _, name, _ in spans}
    pos, first = 0, 0
    with open(zip_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
            end = pos + len(chunk)
            j = first
            while j < len(spans) and spans[j][0] < end:
                a, b, name, _ = spans[j]
                lo, hi = max(a, pos), min(b, end)
                if hi > lo:
                    crcs[name] = zlib.crc32(chunk[lo - pos:hi - pos], crcs[name])
                if b <= end and j == first:
                    first += 1
                j += 1
            pos = end
    bad = [name for _, _, name, crc in spans if crcs[name] & 0xFFFFFFFF != crc]
    if bad:
        raise ArchiveCheckFailed(f"{zip_path}: CRC-32 mismatch in {len(bad)} member(s), e.g. {bad[:3]}")
    return h.hexdigest()


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file, as hex."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(scene_dir: Path | str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """The files of a scene folder, scene-relative and sorted: ``(loose, {component: files}, unknown)``.

    Hidden folders at the top (``.extracted/``, ``.transcoded/``: the
    API's bookkeeping) and ``index.parquet`` are skipped.
    """
    from kitscenes._archive import component_of

    scene_dir = Path(scene_dir)
    loose, unknown = [], []
    packed: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(scene_dir):
        if Path(dirpath) == scene_dir:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        dirnames.sort()
        for fn in sorted(filenames):
            rel = (Path(dirpath) / fn).relative_to(scene_dir).as_posix()
            if rel == INDEX_FILE or (rel.endswith(ARCHIVE_SUFFIX) and "/" not in rel):
                continue
            if rel in LOOSE_FILES:
                loose.append(rel)
                continue
            component = component_of(rel)
            if component is None:
                unknown.append(rel)
            else:
                packed.setdefault(component, []).append(rel)
    return sorted(loose), {c: sorted(f) for c, f in packed.items()}, unknown


def pack_component(scene_dir: Path | str, component: str, zip_path: Path | str,
                   files: Optional[Sequence[str]] = None) -> list[ZipMember]:
    """Pack one component of an extracted scene folder into *zip_path* (members ``<scene_id>/<path>``).

    Args:
        files: The component's scene-relative files (default: from :func:`classify`).

    Returns:
        The written members; their names are ``<scene_id>/<path>``.
    """
    scene_dir = Path(scene_dir)
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}")
    if files is None:
        files = classify(scene_dir)[1].get(component, [])
    if not files:
        raise ValueError(f"{scene_dir} has no files of component {component}")
    align = ALIGN if component in ALIGNED else 1
    entries = sorted((f"{scene_dir.name}/{rel}", scene_dir / rel) for rel in files)
    return write_aligned_zip(entries, zip_path, align)


def index_rows(component: str, members: Sequence[ZipMember]) -> list[tuple[str, str, int, int, int]]:
    """``index.parquet`` rows of a component's zip members (member names without the ``<scene_id>/`` prefix)."""
    return [(component, m.name.split("/", 1)[1], m.data_offset, m.size, m.crc32) for m in members]


def pack_scene(scene_dir: Path | str, out_dir: Path | str, *, overwrite: bool = False) -> list[PackedFile]:
    """Pack an extracted scene folder into the archive format at *out_dir* (a new scene folder of the same name).

    Loose metadata is copied, every component becomes ``<component>.zip``,
    and ``index.parquet`` lists their members.
    Each zip is checked after writing (every member's CRC-32 and the whole
    listing against the folder). The scene is written to
    ``.partial-<scene_id>`` next to *out_dir* and renamed when complete.

    Returns:
        Every written file with its SHA-256 (loose files, archives, index).

    Raises:
        UnknownContent: If the folder holds files outside the format.
        FileExistsError: If *out_dir* exists (unless *overwrite*).
    """
    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    if out_dir.name != scene_dir.name:
        raise ValueError(f"the packed scene keeps its name: {out_dir} should end in /{scene_dir.name}")
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} exists")
    loose, packed, unknown = classify(scene_dir)
    if unknown:
        raise UnknownContent(f"{scene_dir}: {len(unknown)} file(s) outside the archive format, e.g. {unknown[:3]}")
    staging = out_dir.parent / f".partial-{scene_dir.name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for rel in loose:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(scene_dir / rel, target)
        rows: list[tuple[str, str, int, int, int]] = []
        members: dict[str, list[ZipMember]] = {}
        for component in sorted(packed):
            members[component] = pack_component(scene_dir, component, staging / f"{component}{ARCHIVE_SUFFIX}",
                                                packed[component])
            rows += index_rows(component, members[component])
        (staging / INDEX_FILE).write_bytes(write_index(rows))
        _check_listing(scene_dir, loose, packed, rows, staging)
        out = []
        for rel in sorted([*loose, INDEX_FILE, *(f"{c}{ARCHIVE_SUFFIX}" for c in packed)]):
            path = staging / rel
            component = rel[:-len(ARCHIVE_SUFFIX)] if rel.endswith(ARCHIVE_SUFFIX) and "/" not in rel else "metadata"
            sha = check_zip(path, members[component]) if component in members else sha256_file(path)
            out.append(PackedFile(rel, component, sha, path.stat().st_size))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(staging, out_dir)
        return out
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _check_listing(scene_dir: Path, loose: list[str], packed: dict[str, list[str]],
                   rows: list[tuple[str, str, int, int, int]], staging: Path) -> None:
    """Every file of the folder is in the packed scene exactly once, with its size."""
    listed = {rel: (scene_dir / rel).stat().st_size for rel in [*loose, *(r for fs in packed.values() for r in fs)]}
    got: dict[str, int] = {}
    for _, member, _, size, _ in rows:
        if member in got:
            raise ArchiveCheckFailed(f"{scene_dir}: {member} is in the archives twice")
        got[member] = size
    for rel in loose:
        got[rel] = (staging / rel).stat().st_size
    missing = sorted(set(listed) - set(got))
    extra = sorted(set(got) - set(listed))
    wrong = sorted(r for r in set(listed) & set(got) if listed[r] != got[r])
    if missing or extra or wrong:
        raise ArchiveCheckFailed(f"{scene_dir}: packed scene differs from the folder: missing {missing[:3]}, "
                                 f"extra {extra[:3]}, other size {wrong[:3]}")
