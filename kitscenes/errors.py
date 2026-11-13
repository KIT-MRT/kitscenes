"""Exceptions raised by the kitscenes API.

Everything derives from :class:`KITScenesError`, and each class also derives
from the builtin a caller would expect (``KeyError`` for a missing scene,
``FileNotFoundError`` for a missing dataset, ``ValueError`` for broken files),
so existing ``except`` clauses keep working.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

#: The ``missing=`` argument of the accessors: ``"raise"`` an :class:`Unavailable`
#: (the default), or return ``"none"`` (None) instead. Type checkers see an
#: ``Optional`` result only for ``missing="none"``.
Missing = Literal["raise", "none"]


class Availability(str, Enum):
    """Why some data (a sensor, labels, the map, the pose, an aerial crop) is there or not.

    - ``PRESENT``: the data is there.
    - ``NOT_IN_RELEASE``: labels the release does not cover (instance masks
      exist for the front and rear ring cameras only). Never a sensor: every
      scene has all 19.
    - ``NOT_DOWNLOADED``: the release has it, but it is not in this copy (a
      partial download, e.g. a sensor left out); the error names the command
      that gets it.
    - ``DROPPED``: there is data of this kind, but none for this frame (a
      dropped sensor frame, a frame time outside the poses or in a gap
      between them, a frame without aerial crop).
    - ``WITHHELD_BY_SPLIT``: the split does not ship it (maps and labels in
      the test splits, GNSS in the test-e2e splits).
    - ``INVALID``: it is there but unusable (labels that do not fit the
      sweep, poses that cannot be interpolated).
    - ``NOT_INSTALLED``: reading it needs an optional dependency that is
      not installed (lanelet2 for the map); the error names the extra.
    - ``NOT_PREPARED``: something to set up first (aerial crops: attach
      them with ``ds.use_aerial(roi=...)``).
    """

    PRESENT = "present"
    NOT_IN_RELEASE = "not_in_release"
    NOT_DOWNLOADED = "not_downloaded"
    DROPPED = "dropped"
    WITHHELD_BY_SPLIT = "withheld_by_split"
    INVALID = "invalid"
    NOT_INSTALLED = "not_installed"
    NOT_PREPARED = "not_prepared"


#: What each state means, for the error messages.
_EXPLAIN = {
    Availability.NOT_IN_RELEASE: "the release does not have it for this scene",
    Availability.NOT_DOWNLOADED: "it is not in this copy of the dataset",
    Availability.DROPPED: "there is none for this frame",
    Availability.WITHHELD_BY_SPLIT: "the split does not ship it",
    Availability.INVALID: "the data on disk is unusable",
    Availability.NOT_INSTALLED: "an optional dependency is not installed",
    Availability.NOT_PREPARED: "it has not been prepared",
}


class KITScenesError(Exception):
    """Base class of all kitscenes errors."""


class MissingDependencyError(KITScenesError, ImportError):
    """An optional dependency is not installed; the message names the extra that brings it."""


class DatasetNotFound(KITScenesError, FileNotFoundError):
    """The dataset root does not exist or holds no (matching) scenes."""


class SceneNotFound(KITScenesError, KeyError):
    """No scene with this ID in the dataset."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


class DataFormatError(KITScenesError, ValueError):
    """A file does not have the release format (timestamps, calibration, labels)."""


class ChecksumError(DataFormatError):
    """A member of a release zip is corrupt (CRC-32 mismatch with ``verify=True``, or a truncated zip).
    Download the archive again. Not an ``OSError``, so I/O retry loops do not wait on it."""


class CalibrationError(DataFormatError):
    """``calib.json`` is malformed, has an invalid entry, or has none for a
    sensor with data; raised when the scene's rig is loaded
    (:attr:`kitscenes.Scene.rig`)."""


class ImageSizeError(DataFormatError):
    """A camera image does not have the size its calibration says (:attr:`kitscenes.Scene.rig`): e.g. images
    resized by another tool instead of ``kitscenes transcode``, so projections into them would be wrong."""


class Unavailable(KITScenesError, LookupError):
    """Data that is not available: a sensor, labels, the map, the pose or an aerial crop.

    Every accessor raises a subclass of this (unless called with
    ``missing="none"``); ``scene.availability(what, index)`` and
    ``frame.availability(what)`` ask first.

    Attributes:
        what: What is missing: a sensor name, ``"ground_labels/<lidar>"``,
            ``"instances/<camera>"``, ``"map"``, ``"pose"``, ``"aerial"``,
            ``"<gnss device>/<topic>"``.
        index: Frame index, or ``None`` for the whole scene.
        state: Why: an :class:`Availability` other than ``PRESENT``.
        detail: The specifics (which file, which split).
        hint: What gets it, e.g. the download command; empty if nothing can.
    """

    def __init__(self, what: str, index: Optional[int], state: Availability, detail: str = "", hint: str = "") -> None:
        self.what = what
        self.index = index
        self.state = Availability(state)
        self.detail = detail
        self.hint = hint
        where = f" at frame {index}" if index is not None else ""
        message = f"{what}{where} is unavailable ({self.state.value}): {detail or _EXPLAIN.get(self.state, '')}"
        super().__init__(message + (f"; {hint}" if hint else ""))

    def __reduce__(self):
        return type(self), (self.what, self.index, self.state, self.detail, self.hint)


class SensorUnavailable(Unavailable):
    """Data of a sensor (or a GNSS topic) is not available; :attr:`state` says why."""


class LabelsUnavailable(Unavailable):
    """Labels (ground segmentation, instance masks) are not available or do not fit the data."""


class MapUnavailable(Unavailable):
    """A scene has no usable map: ``WITHHELD_BY_SPLIT`` (test splits),
    ``NOT_DOWNLOADED`` (no ``maps/`` in this copy) or ``NOT_INSTALLED``
    (lanelet2 is missing)."""


class PoseUnavailable(Unavailable):
    """No ego pose: ``DROPPED`` (the time is outside the poses or in a gap
    between them), ``INVALID`` (fewer than two poses, or not in time order)
    or ``NOT_DOWNLOADED`` (no ``poses.txt``)."""


class AerialUnavailable(Unavailable):
    """No aerial crop (:mod:`kitscenes.aerial`): ``NOT_PREPARED`` (no crops
    attached: ``ds.use_aerial(roi=...)`` or a prepared cache),
    ``WITHHELD_BY_SPLIT`` (the test-e2e splits: poses relative to the first
    pose, so their scenes cannot be placed on the imagery; the test and
    map_change_test splits are placed with their city's origin) or ``DROPPED`` (no crop for this frame:
    no raw tiles there, no pose, or none in the prepared cache)."""
