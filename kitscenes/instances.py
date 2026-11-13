"""Instance masks of the front and rear ring cameras (``processed/seamseg_instances``).

The masks come from seamseg (https://github.com/mapillary/seamseg) trained on
Mapillary Vistas v1.2. Each ``<camera>/<frame>.json`` lists the detected
objects: a Vistas trainId (:data:`VISTAS_CLASSES`), an objectness score, a
bounding box ``(x0, y0, x1, y1)`` in pixels and a COCO compressed-RLE mask at
the full image size. They are model output, not annotations
(:attr:`kitscenes.schema.Provenance.MODEL_DERIVED`). seamseg does no
tracking: instance numbers are per frame and camera.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cached_property
from typing import Optional, Sequence

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.schema import Provenance

#: (trainId, name, is_thing, rgb) of the 65 Mapillary Vistas v1.2 classes, stuff first.
VISTAS_CLASSES: tuple[tuple[int, str, bool, tuple[int, int, int]], ...] = (
    (0, "Curb", False, (196, 196, 196)), (1, "Fence", False, (190, 153, 153)),
    (2, "Guard Rail", False, (180, 165, 180)), (3, "Barrier", False, (90, 120, 150)),
    (4, "Wall", False, (102, 102, 156)), (5, "Bike Lane", False, (128, 64, 255)),
    (6, "Curb Cut", False, (170, 170, 170)), (7, "Parking", False, (250, 170, 160)),
    (8, "Pedestrian Area", False, (96, 96, 96)), (9, "Rail Track", False, (230, 150, 140)),
    (10, "Road", False, (128, 64, 128)), (11, "Service Lane", False, (110, 110, 110)),
    (12, "Sidewalk", False, (244, 35, 232)), (13, "Bridge", False, (150, 100, 100)),
    (14, "Building", False, (70, 70, 70)), (15, "Tunnel", False, (150, 120, 90)),
    (16, "Lane Marking - General", False, (255, 255, 255)), (17, "Mountain", False, (64, 170, 64)),
    (18, "Sand", False, (230, 160, 50)), (19, "Sky", False, (70, 130, 180)),
    (20, "Snow", False, (190, 255, 255)), (21, "Terrain", False, (152, 251, 152)),
    (22, "Vegetation", False, (107, 142, 35)), (23, "Water", False, (0, 170, 30)),
    (24, "Pothole", False, (70, 100, 150)), (25, "On Rails", False, (0, 80, 100)),
    (26, "Car Mount", False, (32, 32, 32)), (27, "Ego Vehicle", False, (120, 10, 10)),
    (28, "Bird", True, (165, 42, 42)), (29, "Ground Animal", True, (0, 192, 0)),
    (30, "Crosswalk - Plain", True, (140, 140, 200)), (31, "Person", True, (220, 20, 60)),
    (32, "Bicyclist", True, (255, 0, 0)), (33, "Motorcyclist", True, (255, 0, 100)),
    (34, "Other Rider", True, (255, 0, 200)), (35, "Lane Marking - Crosswalk", True, (200, 128, 128)),
    (36, "Banner", True, (255, 255, 128)), (37, "Bench", True, (250, 0, 30)),
    (38, "Bike Rack", True, (100, 140, 180)), (39, "Billboard", True, (220, 220, 220)),
    (40, "Catch Basin", True, (220, 128, 128)), (41, "CCTV Camera", True, (222, 40, 40)),
    (42, "Fire Hydrant", True, (100, 170, 30)), (43, "Junction Box", True, (40, 40, 40)),
    (44, "Mailbox", True, (33, 33, 33)), (45, "Manhole", True, (100, 128, 160)),
    (46, "Phone Booth", True, (142, 0, 0)), (47, "Street Light", True, (210, 170, 100)),
    (48, "Pole", True, (153, 153, 153)), (49, "Traffic Sign Frame", True, (128, 128, 128)),
    (50, "Utility Pole", True, (0, 0, 80)), (51, "Traffic Light", True, (250, 170, 30)),
    (52, "Traffic Sign (Back)", True, (192, 192, 192)), (53, "Traffic Sign (Front)", True, (220, 220, 0)),
    (54, "Trash Can", True, (140, 140, 20)), (55, "Bicycle", True, (119, 11, 32)),
    (56, "Boat", True, (150, 0, 255)), (57, "Bus", True, (0, 60, 100)),
    (58, "Car", True, (0, 0, 142)), (59, "Caravan", True, (0, 0, 90)),
    (60, "Motorcycle", True, (0, 0, 230)), (61, "Other Vehicle", True, (128, 64, 64)),
    (62, "Trailer", True, (0, 0, 110)), (63, "Truck", True, (0, 0, 70)),
    (64, "Wheeled Slow", True, (0, 0, 192)),
)
CLASS_NAMES: dict[int, str] = {c[0]: c[1] for c in VISTAS_CLASSES}
#: Class value of pixels no instance covers.
UNLABELLED = 255


def palette() -> np.ndarray:
    """(256, 3) uint8 colours by class id (grey for unlabelled), e.g. ``palette()[label_image]``."""
    colours = np.full((256, 3), 90, np.uint8)
    for train_id, _name, _thing, rgb in VISTAS_CLASSES:
        colours[train_id] = rgb
    return colours


def rle_runs(counts: str) -> np.ndarray:
    """Run lengths of a COCO compressed RLE string (alternating 0/1, starting with 0).

    The string is base-48 LEB128-like: five bits per character, bit 5
    continues the number, bit 4 of the last character sign-extends it, and
    every count after the second is a delta to the one before the last.

    Raises:
        ValueError: For a negative run or a truncated number (a corrupt string).
    """
    return parse_rles([counts])[0]


def parse_rles(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run lengths of several compressed RLE strings at once, without a Python loop over characters.

    Returns:
        ``(runs, offsets)``: all runs concatenated, and (K + 1,) offsets:
        string ``k`` has ``runs[offsets[k]:offsets[k + 1]]``.

    Raises:
        ValueError: For a negative run or a truncated number.
    """
    chars = np.frombuffer("".join(strings).encode("ascii"), np.uint8).astype(np.int64) - 48
    char_ends = np.cumsum([len(c) for c in strings])
    if not len(chars):
        return np.zeros(0, np.int64), np.zeros(len(strings) + 1, np.int64)
    last = (chars & 0x20) == 0  # the last character of a number
    if not last[char_ends[char_ends > 0] - 1].all():
        raise ValueError("truncated number in RLE")
    number_end = np.flatnonzero(last)
    number_start = np.concatenate([[0], number_end[:-1] + 1])
    length = number_end - number_start + 1
    shift = 5 * (np.arange(len(chars)) - np.repeat(number_start, length))
    value = np.add.reduceat((chars & 0x1F) << shift, number_start)
    negative = (chars[number_end] & 0x10) != 0
    value[negative] -= np.left_shift(1, 5 * length[negative])  # sign extension
    # Every number after the second of its string adds the one two places before it:
    # two running sums per string, one over the odd positions, one over the even ones from 2 on.
    string = np.searchsorted(char_ends, number_end, side="right")
    offsets = np.searchsorted(string, np.arange(len(strings) + 1))
    m = np.arange(len(value)) - offsets[string]  # position in its string
    chain = np.where(m == 0, -1, 2 * string + m % 2)
    members = np.flatnonzero(chain >= 0)
    members = members[np.argsort(chain[members], kind="stable")]
    summed = np.cumsum(value[members])
    first = np.ones(len(members), bool)
    first[1:] = chain[members][1:] != chain[members][:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(members)), 0))
    value[members] = summed - (summed - value[members])[start]
    if (value < 0).any():
        raise ValueError("negative run in RLE")
    return value, offsets


def _intervals(runs: np.ndarray, offsets: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(starts, ends, string)`` of the foreground runs in the column-major flat image, clipped to *size*."""
    string = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    ends = np.cumsum(runs)
    ends -= np.concatenate([[0], ends])[offsets[:-1]][string]  # restart the sum per string
    odd = (np.arange(len(runs)) - offsets[string]) % 2 == 1  # the 1-runs
    return np.minimum(ends[odd] - runs[odd], size), np.minimum(ends[odd], size), string[odd]


def _expand(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """All indices of the intervals ``[starts, ends)``."""
    lengths = ends - starts
    return np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(int(lengths.sum()))


def _resample(image: np.ndarray, size: Optional[tuple[int, int]]) -> np.ndarray:
    """*image* at ``size=(width, height)``: the pixel whose centre is nearest each output pixel centre."""
    if size is None or (image.shape[1], image.shape[0]) == tuple(size):
        return image
    width, height = int(size[0]), int(size[1])
    rows = np.minimum(((np.arange(height) + 0.5) * image.shape[0] / height).astype(np.int64), image.shape[0] - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * image.shape[1] / width).astype(np.int64), image.shape[1] - 1)
    return image[rows[:, None], cols[None, :]]


def decode_rle(counts: str, height: int, width: int) -> np.ndarray:
    """A COCO compressed RLE mask as an (height, width) bool array."""
    flat = np.zeros(height * width, bool)
    starts, ends, _ = _intervals(*parse_rles([counts]), flat.size)
    flat[_expand(starts, ends)] = True
    return flat.reshape((height, width), order="F")  # COCO runs go down columns


@dataclasses.dataclass(frozen=True, eq=False)
class InstanceMasks:
    """The detected objects of one camera image.

    Attributes:
        camera: Camera name.
        image_size: ``(width, height)`` the masks refer to.
        classes: (K,) Vistas trainIds (see :data:`CLASS_NAMES`).
        scores: (K,) objectness.
        boxes: (K, 4) ``x0, y0, x1, y1`` in pixels.
        rles: The K COCO RLE strings (decode with :meth:`mask`).
        provenance: Model output.
    """

    __repr__ = compact_repr

    camera: str
    image_size: tuple[int, int]
    classes: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    rles: tuple[str, ...]
    provenance: Provenance = Provenance.MODEL_DERIVED

    def __len__(self) -> int:
        return len(self.classes)

    @classmethod
    def from_json(cls, data: bytes | str, camera: str) -> "InstanceMasks":
        d = json.loads(data)
        size = d.get("image_info", {}).get("original_size")
        if not size or len(size) != 2:
            raise ValueError("instance file has no image_info.original_size")
        height, width = int(size[0]), int(size[1])
        insts = [i for i in d.get("instances", []) if i.get("mask_rle")]
        return cls(
            camera=camera,
            image_size=(width, height),
            classes=np.array([int(i.get("class", UNLABELLED)) for i in insts], np.int64),
            scores=np.array([float(i.get("objectness", 0.0)) for i in insts]),
            boxes=np.array([i.get("bbox", [np.nan] * 4) for i in insts], np.float64).reshape(-1, 4),
            rles=tuple(i["mask_rle"] for i in insts),
        )

    def names(self) -> list[str]:
        return [CLASS_NAMES.get(int(c), f"class {c}") for c in self.classes]

    def mask(self, i: int, size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """(H, W) bool mask of instance *i*; at ``size=(width, height)`` if given (nearest pixel centre,
        e.g. the size of a downscaled image)."""
        width, height = self.image_size
        return _resample(decode_rle(self.rles[i], height, width), size)

    def scaled_boxes(self, size: tuple[int, int]) -> np.ndarray:
        """:attr:`boxes` in an image of ``size=(width, height)`` (e.g. a downscaled one)."""
        sx, sy = size[0] / self.image_size[0], size[1] / self.image_size[1]
        return self.boxes * np.array([sx, sy, sx, sy])

    @cached_property
    def intervals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every mask as foreground pixel intervals: ``(starts, ends, instance)``.

        Indices into the column-major flat image (``col * height + row``),
        ``[starts, ends)``, grouped by instance and in order.
        """
        width, height = self.image_size
        return _intervals(*parse_rles(self.rles), width * height)

    def _painting_order(self, min_score: float) -> np.ndarray:
        """Instances to paint, least confident first (the more confident wins on overlap)."""
        order = np.argsort(self.scores, kind="stable")
        keep = (self.scores[order] >= min_score) & (self.classes[order] >= 0) & (self.classes[order] < UNLABELLED)
        return order[keep]

    def interval_table(self, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The intervals of the instances that :meth:`label_images` paints: ``(starts, ends, rank, instance)``.

        ``rank`` is the painting position (higher wins where masks overlap).
        Small enough to ship to a GPU instead of label images.
        """
        order = self._painting_order(min_score)
        rank = np.full(len(self.classes), -1, np.int64)
        rank[order] = np.arange(len(order))
        starts, ends, instance = self.intervals
        used = rank[instance] >= 0
        return starts[used], ends[used], rank[instance[used]], instance[used]

    def lookup(self, rows: np.ndarray, cols: np.ndarray, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Class and instance at pixels, as in :meth:`label_images` but without drawing them.

        Each pixel is matched against the mask intervals by binary search; the
        cost grows with the pixels asked for, not with the image size.

        Args:
            rows, cols: (N,) integer pixel coordinates inside the image.

        Returns:
            ``(classes, instances)``: (N,) uint8 and uint16 (``i + 1``, 0 for none).
        """
        width, height = self.image_size
        query = np.asarray(cols, np.int64) * height + np.asarray(rows, np.int64)
        classes = np.full(len(query), UNLABELLED, np.uint8)
        instances = np.zeros(len(query), np.uint16)
        starts, ends, rank, instance = self.interval_table(min_score)
        if not len(starts) or not len(query):
            return classes, instances
        order = self._painting_order(min_score)
        by_pixel = np.argsort(query, kind="stable")
        sorted_query = query[by_pixel]
        first, stop = np.searchsorted(sorted_query, starts), np.searchsorted(sorted_query, ends)
        hits = stop - first  # pixels inside each interval
        interval = np.repeat(np.arange(len(starts)), hits)
        pixel = by_pixel[np.repeat(first, hits) + np.arange(len(interval)) - np.repeat(np.cumsum(hits) - hits, hits)]
        best = np.full(len(query), -1, np.int64)
        np.maximum.at(best, pixel, rank[interval])
        found = best >= 0
        winner = order[best[found]]
        classes[found] = self.classes[winner]
        instances[found] = winner + 1
        return classes, instances

    def label_images(self, min_score: float = 0.0, size: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, np.ndarray]:
        """(class image uint8, instance image uint16) at the masks' size, or at ``size=(width, height)``
        (nearest pixel centre, e.g. the size of a downscaled image).

        Instances are painted least confident first, so where two overlap the
        more confident one wins. Uncovered pixels are :data:`UNLABELLED` in
        the class image and 0 in the instance image (instance ``i`` is
        ``i + 1``). Instances below *min_score* are left out. For a few
        pixels, :meth:`lookup` is much cheaper.
        """
        width, height = self.image_size
        classes = np.full(width * height, UNLABELLED, np.uint8)
        ids = np.zeros(width * height, np.uint16)
        starts, ends, instance = self.intervals
        bounds = np.searchsorted(instance, np.arange(len(self.classes) + 1))
        for i in self._painting_order(min_score):
            pixels = _expand(starts[bounds[i]:bounds[i + 1]], ends[bounds[i]:bounds[i + 1]])
            classes[pixels] = self.classes[i]
            ids[pixels] = i + 1
        return (_resample(classes.reshape((height, width), order="F"), size),
                _resample(ids.reshape((height, width), order="F"), size))

    def label_image(self, min_score: float = 0.0, size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """Just the class image of :meth:`label_images`."""
        return self.label_images(min_score, size)[0]

    def select(self, keep: np.ndarray) -> "InstanceMasks":
        """The instances where *keep* (bool or index array) is set."""
        idx = np.flatnonzero(keep) if np.asarray(keep).dtype == bool else np.asarray(keep)
        return dataclasses.replace(
            self, classes=self.classes[idx], scores=self.scores[idx], boxes=self.boxes[idx],
            rles=tuple(self.rles[i] for i in idx),
        )
