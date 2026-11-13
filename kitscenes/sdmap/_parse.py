"""OpenStreetMap files (PBF, XML) -> compact arrays (:class:`OSMData`), with pyosmium.

The arrays keep every node, way and relation of the file with all tags, in
file order. Tags are stored once as a string table and per element as index
pairs (CSR), so a city of a million nodes stays at tens of MB in memory and in
the cache (:mod:`kitscenes.sdmap._cache`).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Bumped whenever the parsed arrays change, so old caches are rebuilt (2: the PBF header time in ``osm_base``).
PARSER_VERSION = 2

#: Relation member types as stored in :attr:`OSMData.member_type`.
MEMBER_TYPES = ("node", "way", "relation")


@dataclasses.dataclass
class OSMData:
    """The parsed file: WGS84, OSM IDs, all tags. Arrays in file order.

    Tags of element ``i`` of a kind are ``strings[k]: strings[v]`` for
    ``k, v`` in ``<kind>_tag_k[ptr[i]:ptr[i + 1]]``, ``<kind>_tag_v[...]``.
    """

    osm_base: Optional[str]         # data state: Overpass <meta osm_base>, or the PBF header time
    strings: list[str]
    node_ids: np.ndarray            # (N,) int64
    node_lat: np.ndarray            # (N,) float64
    node_lon: np.ndarray            # (N,) float64
    node_tag_ptr: np.ndarray        # (N + 1,) int64
    node_tag_k: np.ndarray          # int32 into strings
    node_tag_v: np.ndarray
    way_ids: np.ndarray             # (W,) int64
    way_node_ptr: np.ndarray        # (W + 1,) int64
    way_nodes: np.ndarray           # node indices (into node_ids), int64; -1 for nodes missing from the file
    way_tag_ptr: np.ndarray
    way_tag_k: np.ndarray
    way_tag_v: np.ndarray
    rel_ids: np.ndarray             # (R,) int64
    rel_tag_ptr: np.ndarray
    rel_tag_k: np.ndarray
    rel_tag_v: np.ndarray
    rel_member_ptr: np.ndarray      # (R + 1,) int64
    member_type: np.ndarray         # uint8 index into MEMBER_TYPES
    member_ref: np.ndarray          # int64 OSM ID of the member
    member_role: np.ndarray         # int32 into strings

    def tags(self, kind: str, i: int) -> dict[str, str]:
        """All tags of element *i* of *kind* (``"node"``, ``"way"``, ``"relation"``)."""
        prefix = {"node": "node", "way": "way", "relation": "rel"}[kind]
        ptr = getattr(self, f"{prefix}_tag_ptr")
        a, b = int(ptr[i]), int(ptr[i + 1])
        if a == b:
            return {}
        ks = getattr(self, f"{prefix}_tag_k")[a:b]
        vs = getattr(self, f"{prefix}_tag_v")[a:b]
        s = self.strings
        return {s[k]: s[v] for k, v in zip(ks.tolist(), vs.tolist())}

    def tag_value_index(self, kind: str, key: str) -> np.ndarray:
        """(n,) string index of tag *key* of every element of *kind*, -1 where it is absent."""
        prefix = {"node": "node", "way": "way", "relation": "rel"}[kind]
        ptr = getattr(self, f"{prefix}_tag_ptr")
        out = np.full(len(ptr) - 1, -1, dtype=np.int64)
        try:
            k = self.strings.index(key)
        except ValueError:
            return out
        ks = getattr(self, f"{prefix}_tag_k")
        pos = np.flatnonzero(ks == k)
        if len(pos):
            owner = np.searchsorted(ptr, pos, side="right") - 1
            out[owner] = getattr(self, f"{prefix}_tag_v")[pos]
        return out

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_ways(self) -> int:
        return len(self.way_ids)

    @property
    def num_relations(self) -> int:
        return len(self.rel_ids)


class _Builder:
    """Collects elements while parsing; :meth:`finish` makes the arrays."""

    def __init__(self) -> None:
        self.strings: list[str] = []
        self._string_index: dict[str, int] = {}
        self.osm_base: Optional[str] = None
        self.node_ids: list[int] = []
        self.node_lat: list[float] = []
        self.node_lon: list[float] = []
        self.node_tag_ptr: list[int] = [0]
        self.node_tags: list[int] = []  # k, v interleaved
        self.way_ids: list[int] = []
        self.way_node_ptr: list[int] = [0]
        self.way_refs: list[int] = []
        self.way_tag_ptr: list[int] = [0]
        self.way_tags: list[int] = []
        self.rel_ids: list[int] = []
        self.rel_tag_ptr: list[int] = [0]
        self.rel_tags: list[int] = []
        self.rel_member_ptr: list[int] = [0]
        self.member_type: list[int] = []
        self.member_ref: list[int] = []
        self.member_role: list[int] = []

    def finish(self) -> OSMData:
        node_ids = np.array(self.node_ids, dtype=np.int64)
        refs = np.array(self.way_refs, dtype=np.int64)
        order = np.argsort(node_ids, kind="stable")
        sorted_ids = node_ids[order]
        pos = np.searchsorted(sorted_ids, refs)
        pos = np.minimum(pos, max(len(sorted_ids) - 1, 0))
        found = (sorted_ids[pos] == refs) if len(sorted_ids) else np.zeros(len(refs), bool)
        way_nodes = np.where(found, order[pos] if len(order) else -1, -1).astype(np.int64)
        if not found.all():
            logger.debug("%d way node references are not in the file", int((~found).sum()))

        def pairs(flat: list[int]) -> tuple[np.ndarray, np.ndarray]:
            a = np.array(flat, dtype=np.int32).reshape(-1, 2)
            return np.ascontiguousarray(a[:, 0]), np.ascontiguousarray(a[:, 1])

        nk, nv = pairs(self.node_tags)
        wk, wv = pairs(self.way_tags)
        rk, rv = pairs(self.rel_tags)
        return OSMData(
            osm_base=self.osm_base,
            strings=self.strings,
            node_ids=node_ids,
            node_lat=np.array(self.node_lat, dtype=np.float64),
            node_lon=np.array(self.node_lon, dtype=np.float64),
            node_tag_ptr=np.array(self.node_tag_ptr, dtype=np.int64),
            node_tag_k=nk, node_tag_v=nv,
            way_ids=np.array(self.way_ids, dtype=np.int64),
            way_node_ptr=np.array(self.way_node_ptr, dtype=np.int64),
            way_nodes=way_nodes,
            way_tag_ptr=np.array(self.way_tag_ptr, dtype=np.int64),
            way_tag_k=wk, way_tag_v=wv,
            rel_ids=np.array(self.rel_ids, dtype=np.int64),
            rel_tag_ptr=np.array(self.rel_tag_ptr, dtype=np.int64),
            rel_tag_k=rk, rel_tag_v=rv,
            rel_member_ptr=np.array(self.rel_member_ptr, dtype=np.int64),
            member_type=np.array(self.member_type, dtype=np.uint8),
            member_ref=np.array(self.member_ref, dtype=np.int64),
            member_role=np.array(self.member_role, dtype=np.int32),
        )


def parse(path: Path | str) -> OSMData:
    """Parse an OpenStreetMap file (``.osm.pbf``, ``.osm`` XML) with pyosmium."""
    import osmium

    b = _Builder()
    index = b._string_index
    strings = b.strings

    def intern(s: str) -> int:
        i = index.get(s)
        if i is None:
            i = index[s] = len(strings)
            strings.append(s)
        return i

    node_ids, node_lat, node_lon = b.node_ids, b.node_lat, b.node_lon
    node_tags, node_ptr = b.node_tags, b.node_tag_ptr
    way_ids, way_refs, way_node_ptr, way_tags, way_ptr = b.way_ids, b.way_refs, b.way_node_ptr, b.way_tags, b.way_tag_ptr
    kinds = {"n": 0, "w": 1, "r": 2}
    for obj in osmium.FileProcessor(str(path)):
        if obj.is_node():
            loc = obj.location
            node_ids.append(obj.id)
            node_lat.append(loc.lat)
            node_lon.append(loc.lon)
            if len(obj.tags):
                for t in obj.tags:
                    node_tags.append(intern(t.k))
                    node_tags.append(intern(t.v))
            node_ptr.append(len(node_tags) // 2)
        elif obj.is_way():
            way_ids.append(obj.id)
            way_refs.extend([n.ref for n in obj.nodes])
            way_node_ptr.append(len(way_refs))
            for t in obj.tags:
                way_tags.append(intern(t.k))
                way_tags.append(intern(t.v))
            way_ptr.append(len(way_tags) // 2)
        elif obj.is_relation():
            b.rel_ids.append(obj.id)
            for m in obj.members:
                b.member_type.append(kinds[m.type])
                b.member_ref.append(m.ref)
                b.member_role.append(intern(m.role))
            b.rel_member_ptr.append(len(b.member_ref))
            for t in obj.tags:
                b.rel_tags.append(intern(t.k))
                b.rel_tags.append(intern(t.v))
            b.rel_tag_ptr.append(len(b.rel_tags) // 2)
    b.osm_base = _snapshot_time(path, osmium)
    return b.finish()


def _snapshot_time(path: Path | str, osmium) -> Optional[str]:
    """The data state of an OSM file: the PBF header's ``osmosis_replication_timestamp`` (or ``timestamp``), else
    the ``<meta osm_base=...>`` of an Overpass XML file; None if it has neither."""
    if Path(path).suffix.lower() == ".pbf":
        try:
            reader = osmium.io.Reader(str(path), osmium.osm.osm_entity_bits.NOTHING)
            try:
                header = reader.header()
                stamp = header.get("osmosis_replication_timestamp") or header.get("timestamp")
            finally:
                reader.close()
            if stamp:
                return stamp
        except Exception as err:  # an unreadable header: no time, the data are still read
            logger.debug("No header time in %s (%s)", path, err)
    return _osm_base_from_header(path)


def _osm_base_from_header(path: Path | str) -> Optional[str]:
    """The ``<meta osm_base=...>`` timestamp of an Overpass XML file (from its first lines), or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "replace")
    except OSError:
        return None
    marker = 'osm_base="'
    i = head.find(marker)
    if i < 0:
        return None
    j = head.find('"', i + len(marker))
    return head[i + len(marker):j] if j > 0 else None


def repair_mojibake(s: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252 and encoded again (``"StraÃŸe"`` -> ``"Straße"``).

    The KITScenes snapshots of 2026-05-03 have all non-ASCII text that way
    (the weekly snapshots are correct UTF-8).
    Bytes cp1252 does not define were lost to U+FFFD in the files; such
    strings are repaired as far as possible (the lost characters stay U+FFFD).
    ASCII strings and strings that are not of that form are returned unchanged.
    """
    if s.isascii():
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except UnicodeEncodeError:
        pass
    except UnicodeDecodeError:
        return s
    raw = bytearray()
    for ch in s:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ch != "\ufffd" and ord(ch) > 0xFF:
                return s  # a character cp1252 cannot have produced: not mojibake
            raw += b"\xef\xbf\xbd" if ch == "\ufffd" else bytes([ord(ch)])
    repaired = raw.decode("utf-8", errors="replace")
    return repaired if repaired.count("\ufffd") < len(s) else s


def looks_double_encoded(strings: list[str], min_share: float = 0.5) -> bool:
    """Whether most non-ASCII *strings* are double encoded (:func:`repair_mojibake` changes them).

    True for the 2026-05-03 Overpass snapshots (nearly all of them), False for
    correctly encoded files such as the weekly PBF snapshots, where a correct
    UTF-8 string almost never also reads as UTF-8 seen through cp1252.
    """
    non_ascii = [s for s in strings if not s.isascii()]
    if not non_ascii:
        return False
    changed = sum(1 for s in non_ascii if repair_mojibake(s) != s)
    return changed >= min_share * len(non_ascii)


def with_repaired_strings(data: OSMData) -> OSMData:
    """*data* with :func:`repair_mojibake` applied to its string table (the arrays are shared)."""
    return dataclasses.replace(data, strings=[repair_mojibake(s) for s in data.strings])
