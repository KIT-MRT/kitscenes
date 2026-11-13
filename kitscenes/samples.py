"""Training samples: frames as dicts of arrays, and :func:`collate`.

``SampleDataset(ds, SampleSpec(...))`` is a plain sequence (``len`` / ``[]``)
for a torch ``DataLoader``, without making torch a dependency; it pickles to
the dataset root, scene IDs and indices::

    spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=4, lidars=("lidar_top",), future_s=3.0)
    loader = DataLoader(kitscenes.SampleDataset(ds, spec), batch_size=8, num_workers=8, collate_fn=kitscenes.collate)

Keys of a sample: ``meta``, ``pose``, ``ego``, ``camera``, ``lidar``,
``radar``, ``future``, ``past``, ``map``, ``sd_map``, ``navigation``; shapes
and units are in docs/training.md, "Sample layout". With
``SampleSpec(frame="road")`` the ``_base`` keys are ``_road``. Clips
(``clip_length > 1``) are lists of frame samples, the oldest first.
"""

from __future__ import annotations

import dataclasses
import functools
from collections import OrderedDict
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np

from kitscenes.catalog import Dataset, FrameList
from kitscenes.constants import CameraNameLike, LidarNameLike, RadarNameLike, sensor_names
from kitscenes.ego_trajectory import DEFAULT_STEP_S, num_slots
from kitscenes.errors import Availability
from kitscenes.frame import Frame
from kitscenes.image_corrections import stored_correction
from kitscenes.map.roi import Region
from kitscenes.map.sampling import FixedPoints, Sampling
from kitscenes.rig import invert_transform


@dataclasses.dataclass(frozen=True)
class MapLabelSpec:
    """Map labels in the samples: the arguments of :class:`kitscenes.map.MLConverter` (see there).

    :meth:`maptr` is the preset for MapTR-style vector-map models.

    Attributes:
        grouping: Line type grouping, one of :data:`kitscenes.map.GROUPINGS`.
        region: Labelled region around the vehicle (:class:`kitscenes.map.Region`).
        dims: 3 or 2 (the converter's 2D mode).
        sampling, te_sampling: Points per line and per traffic element
            (:class:`~kitscenes.map.FixedPoints`, :class:`~kitscenes.map.Spacing`,
            :class:`~kitscenes.map.Adaptive` or None).
        ignore_elevation: Flatten the map to z = 0.
        cache: Converters (one per scene, with its map) a worker keeps. A
            runtime setting: specs that differ only in it are equal.
        classes: Line classes, ``{name: converter types}`` or a list of
            types (stored as ``((name, types), ...)``); None: every type of
            the grouping (:data:`kitscenes.map.GROUPING_CLASSES`).
        te_classes: The same for traffic elements (None: every type; ``()``: none).
        crosswalks: ``"lines"`` or ``"polygons"`` (closed outlines per crossing).
        max_height_diff: Cut away lines this far above or below the road
            under the vehicle (metres), None for no filter.
        chaining: ``"frame"`` or ``"scene"`` (chains built once per scene:
            stable keys over the scene; needs lanelet2 1.2.3+kitscenes2).
        recipe: The label recipe (:data:`kitscenes.map.LABEL_RECIPES`):
            ``"default"`` or ``"paper-v2.0.0"`` (the labels of the paper runs:
            the converter's own cut, poses only; needs lanelet2
            1.2.3+kitscenes3 and a dataset opened with ``use_imu=False``).
        frame: The ego frame the labels are for: None (default: the sample's,
            ``SampleSpec.frame``; :meth:`labels` without *frame*: the base
            frame), ``"base"`` or ``"road"``. :meth:`maptr` sets ``"road"``,
            the frame of the map benchmark; a ``SampleSpec`` with another
            ``frame`` raises.
    """

    grouping: str = "default"
    region: Region = Region()
    dims: int = 3
    sampling: Sampling = FixedPoints(20)
    te_sampling: Sampling = FixedPoints(20)
    ignore_elevation: bool = False
    cache: int = dataclasses.field(default=8, compare=False)
    classes: Optional[Any] = None
    te_classes: Optional[Any] = None
    crosswalks: Literal["lines", "polygons"] = "lines"
    max_height_diff: Optional[float] = None
    chaining: Literal["frame", "scene"] = "frame"
    frame: Optional[Literal["base", "road"]] = None
    recipe: str = "default"

    def __post_init__(self) -> None:
        from kitscenes.map.labels import GROUPING_CLASSES, GROUPINGS, TE_TYPES, normalize_classes
        from kitscenes.map.recipes import LABEL_RECIPES

        if self.grouping not in GROUPINGS:
            raise ValueError(f"grouping must be one of {GROUPINGS}, got {self.grouping!r}")
        if self.dims not in (2, 3):
            raise ValueError("dims must be 2 or 3")
        if self.crosswalks not in ("lines", "polygons"):
            raise ValueError(f"crosswalks must be 'lines' or 'polygons', got {self.crosswalks!r}")
        if self.chaining not in ("frame", "scene"):
            raise ValueError(f"chaining must be 'frame' or 'scene', got {self.chaining!r}")
        if self.frame not in (None, "base", "road"):
            raise ValueError(f"frame must be None, 'base' or 'road', got {self.frame!r}")
        if self.recipe not in LABEL_RECIPES:
            raise ValueError(f"unknown label recipe {self.recipe!r}; one of {tuple(LABEL_RECIPES)}")
        if self.max_height_diff is not None and not self.max_height_diff > 0:
            raise ValueError("max_height_diff must be > 0 (or None)")
        if self.max_height_diff is not None and self.ignore_elevation:
            raise ValueError("max_height_diff needs the map's heights; it cannot be combined with ignore_elevation")
        for name, available in (("classes", GROUPING_CLASSES[self.grouping]), ("te_classes", TE_TYPES)):
            value = getattr(self, name)
            if value is not None:  # hashable and comparable: ((name, (types, ...)), ...)
                pairs = tuple(normalize_classes(dict(value) if _is_pairs(value) else value, available, name).items())
                object.__setattr__(self, name, pairs)

    @classmethod
    def maptr(cls, *, centerline: bool = False, **overrides) -> "MapLabelSpec":
        """The MapTR preset: classes ``divider`` (lane markings), ``ped_crossing`` and ``boundary``.

        - divider: ``Divider`` of the ``"maptr"`` grouping (all lane and
          bike-lane markings; virtual lane borders are not dividers).
        - ped_crossing: crosswalks as closed polygons clipped by the box,
          one per crossing (``crosswalks="polygons"``).
        - boundary: ``DrivableArea``, the ``drivable_space_border`` lines
          (the closest match to the AV2 and nuScenes boundary).
        - centerline (``centerline=True``, MapTRv2): lane centerlines.

        Defaults: the 60 x 30 m box around the vehicle (``Region()``), 3D
        (``dims=3``; ``dims=2`` for 2D) in the road frame (``frame="road"``,
        the frame of the map benchmark, for ``SampleSpec(frame="road")``;
        ``frame=None`` follows the sample's frame), 20 points per element
        (``FixedPoints(20)``), lines more than 4 m off the road level cut
        away (``max_height_diff=4.0``), no traffic elements. Any field can be
        overridden, e.g. ``MapLabelSpec.maptr(dims=2, chaining="scene")``.
        """
        from kitscenes.map.labels import MAPTR_CLASSES, MAPTRV2_CLASSES

        fields = dict(grouping="maptr", classes=MAPTRV2_CLASSES if centerline else MAPTR_CLASSES, te_classes=(),
                      crosswalks="polygons", max_height_diff=4.0, region=Region(), dims=3, sampling=FixedPoints(20),
                      frame="road")
        fields.update(overrides)
        return cls(**fields)

    @property
    def line_class_names(self) -> tuple[str, ...]:
        """The line classes, the keys of ``sample["map"]["lines"]``."""
        from kitscenes.map.labels import GROUPING_CLASSES

        return GROUPING_CLASSES[self.grouping] if self.classes is None else tuple(name for name, _ in self.classes)

    @property
    def te_class_names(self) -> tuple[str, ...]:
        """The traffic element classes, the keys of ``sample["map"]["traffic_elements"]``."""
        from kitscenes.map.labels import TE_TYPES

        return TE_TYPES if self.te_classes is None else tuple(name for name, _ in self.te_classes)

    def labels(self, at: Frame, *, frame: Optional[Literal["base", "road"]] = None,
               converters: Optional[OrderedDict] = None) -> dict:
        """The map labels around the :class:`~kitscenes.Frame` *at*, exactly as a sample holds them (``sample["map"]``).

        Args:
            at: The frame of the scene to label.
            frame: The coordinate frame of the labels: None (default) for the
                spec's :attr:`frame`, else ``"base"``. ``"road"`` (the frame
                of the map benchmark, :meth:`maptr`) lifts 3D labels by
                ``scene.road_height_m``.
            converters: Keeps the converters of recent scenes between calls
                (by scene ID; at most :attr:`cache`).

        Returns:
            The dict of ``sample["map"]`` (docs/training.md, "Sample layout");
            without a pose or a map ``available`` is False and the arrays are empty.
        """
        if frame is None:
            frame = self.frame or "base"
        if frame not in ("base", "road"):
            raise ValueError(f"frame must be 'base' or 'road', got {frame!r}")
        out = _map_sample(at, self, OrderedDict() if converters is None else converters)
        if frame == "road" and self.dims == 3 and not self.ignore_elevation:
            _lift_polylines(out, at.scene.road_height_m)
        return out

    def converter(self, scene) -> Any:
        """``MLConverter.for_scene(scene, self)`` (raises :class:`MapUnavailable` like ``scene.map``)."""
        from kitscenes.map import MLConverter

        return MLConverter.for_scene(scene, self)


@dataclasses.dataclass(frozen=True)
class SDMapSpec:
    """OpenStreetMap roads in the samples (:mod:`kitscenes.sdmap`): the drivable road pieces around the frame's pose.

    Attributes:
        region: The region around the vehicle (:class:`kitscenes.map.Region`); pieces are clipped to it.
        points: Points per piece (evenly resampled).
        classes: Road classes to keep (:class:`kitscenes.sdmap.RoadClass` values), None for all.
    """

    region: Region = Region()
    points: int = 20
    classes: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not (isinstance(self.points, int) and self.points >= 2):
            raise ValueError("points must be an int >= 2")
        if self.classes is not None:
            from kitscenes.sdmap import RoadClass

            object.__setattr__(self, "classes", tuple(RoadClass(c).value for c in self.classes))


@dataclasses.dataclass(frozen=True)
class NavigationSpec:
    """Driving commands and target points in the samples (:mod:`kitscenes.navigation`).

    Attributes:
        rule: The :class:`~kitscenes.navigation.CommandRule` (e.g. ``kitscenes.navigation.NAVSIM``).
        source: ``"auto"``, ``"path"`` or ``"sd_route"`` (see :meth:`kitscenes.Frame.navigation`).
        targets: ``"goal"`` (one point, the end of the scene's route; the
            default, it does not reveal the future path) or ``"route"`` (route
            points ahead, which do; see :meth:`kitscenes.Frame.navigation`).
        goal: ``"sd_route"`` (default: the end of the driven SD route, a route-level destination)
            or ``"path"`` (the exact last recorded position; see :meth:`kitscenes.Frame.navigation`).
        num_targets: Target points per sample (padded with NaN).
        max_distance_m: How far ahead the target points reach (``"route"`` only).
        spacing_m: A target point every *spacing_m* instead of at the maneuvers and the end (``"route"`` only).
    """

    rule: Any = None
    source: Literal["auto", "path", "sd_route"] = "auto"
    targets: Literal["goal", "route"] = "goal"
    goal: Literal["sd_route", "path"] = "sd_route"
    num_targets: int = 1
    max_distance_m: float = 50.0
    spacing_m: Optional[float] = None

    def __post_init__(self) -> None:
        from kitscenes.navigation import CommandRule

        if not isinstance(self.rule, CommandRule):
            raise TypeError("NavigationSpec(rule=...) takes a kitscenes.navigation.CommandRule, e.g. NAVSIM")
        if not (isinstance(self.num_targets, int) and self.num_targets >= 0):
            raise ValueError("num_targets must be an int >= 0")
        if self.targets not in ("goal", "route"):
            raise ValueError("targets must be 'goal' or 'route'")
        if self.goal not in ("sd_route", "path"):
            raise ValueError("goal must be 'sd_route' or 'path'")


def _is_pairs(value) -> bool:
    """Whether *value* is the stored form ``((name, (types, ...)), ...)``."""
    return isinstance(value, tuple) and all(
        isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], tuple) for p in value)


@dataclasses.dataclass(frozen=True)
class SampleSpec:
    """What a sample holds.

    Attributes:
        cameras: Cameras to include.
        image: ``"decoded"`` (uint8 arrays) or ``"bytes"`` (the full-size
            JPEG, with the full-size ``K``; for decoding on the GPU with
            :func:`kitscenes.decode.decode_cameras`).
        reduce: Decode cameras at 1/2, 1/4 or 1/8 size (``K`` is scaled to
            match). With ``image="bytes"`` the samples keep the full-size
            JPEG and ``K``, and ``decode_cameras(batch, spec)`` decodes at
            1/*reduce* and scales ``K`` (any other size: its ``size`` or
            ``scale``).
        image_correction: Correct the cameras a scene ships a correction for
            (:mod:`kitscenes.image_corrections`). With ``image="bytes"`` the JPEG stays
            as stored: such a camera's entry gets ``"image_correction"``, its
            :class:`~kitscenes.image_corrections.ImageCorrection`, which
            :func:`kitscenes.decode.decode_cameras` applies and removes.
        lidars: Lidars to fuse into one cloud (the sample's frame, measured points only); with
            ``lidar="deskew_inputs"`` one :class:`~kitscenes.motion.DeskewInputs` per lidar with the
            stored rows (``frame.lidar(name, rows=True)``; ``measured`` marks the points).
        lidar: ``"deskewed"`` (CPU, in the worker) or ``"deskew_inputs"`` (for the GPU).
        radars: Radars to include (the sample's frame, compensated).
        future_s: Seconds of future ego trajectory (0 for none), see :meth:`Frame.future`.
        past_s: Seconds of past ego trajectory (0 for none), see :meth:`Frame.past`.
        trajectory_step_s: Spacing of the future and past slots (default 0.1 s, the frame rate).
        map_labels: Map labels around the frame's pose (a :class:`MapLabelSpec`),
            None for none. With ``require=True`` every scene must have a map
            (:class:`~kitscenes.errors.MapUnavailable` otherwise, e.g. in the
            test splits); with ``require=False`` those samples say why in
            ``map["state"]``.
        clip_length: Frames per sample: 1 (default) for single frames, N > 1
            for clips of N frames (a sample is then a list of N frame
            samples, the oldest first).
        clip_stride: Frame-index step between the frames of a clip (2: every
            other frame).
        require: Two things at once. True (default): only frames where every
            requested sensor has data (for clips: every frame of the clip),
            and anything else missing raises (a scene without a map with
            ``map_labels``: ``MapUnavailable``). False: every frame; what is
            missing is marked instead: a camera with ``valid`` False, a lidar
            or radar left out of the cloud, ``map["available"]`` False.
        frame: The ego frame of the sample: ``"base"`` (default) or ``"road"``
            (docs/conventions.md, "The road frame"). Must be
            ``map_labels.frame`` where that is set (``MapLabelSpec.maptr()``:
            ``SampleSpec(frame="road")``).
        sd_map: OpenStreetMap roads around the frame's pose (an
            :class:`SDMapSpec`), None for none. Scenes that cannot be placed
            (test-e2e) say why in ``sd_map["state"]``; no error.
        navigation: The driving command and target points (a
            :class:`NavigationSpec`), None for none; frames without one say why
            in ``navigation["state"]``.
    """

    cameras: tuple[CameraNameLike, ...] = ()
    image: Literal["decoded", "bytes"] = "decoded"
    reduce: int = 1
    lidars: tuple[LidarNameLike, ...] = ()
    lidar: Literal["deskewed", "deskew_inputs"] = "deskewed"
    radars: tuple[RadarNameLike, ...] = ()
    future_s: float = 0.0
    past_s: float = 0.0
    trajectory_step_s: float = DEFAULT_STEP_S
    map_labels: Optional[MapLabelSpec] = None
    clip_length: int = 1
    clip_stride: int = 1
    require: bool = True
    frame: Literal["base", "road"] = "base"
    sd_map: Optional[SDMapSpec] = None
    navigation: Optional[NavigationSpec] = None
    image_correction: bool = True

    def __post_init__(self) -> None:
        if self.sd_map is not None and not isinstance(self.sd_map, SDMapSpec):
            raise TypeError("sd_map must be a kitscenes.SDMapSpec (or None)")
        if self.navigation is not None and not isinstance(self.navigation, NavigationSpec):
            raise TypeError("navigation must be a kitscenes.NavigationSpec (or None)")
        if self.frame not in ("base", "road"):
            raise ValueError(f"frame must be 'base' or 'road', got {self.frame!r}")
        if self.image not in ("decoded", "bytes"):
            raise ValueError(f"image must be 'decoded' or 'bytes', got {self.image!r}")
        if self.lidar not in ("deskewed", "deskew_inputs"):
            raise ValueError(f"lidar must be 'deskewed' or 'deskew_inputs', got {self.lidar!r}")
        if self.reduce not in (1, 2, 4, 8):
            raise ValueError("reduce must be 1, 2, 4 or 8")
        if self.map_labels is not None and not isinstance(self.map_labels, MapLabelSpec):
            raise TypeError("map_labels must be a kitscenes.MapLabelSpec (or None)")
        want = None if self.map_labels is None else self.map_labels.frame
        if want is not None and want != self.frame:
            preset = ", e.g. the map benchmark preset MapLabelSpec.maptr()" if want == "road" else ""
            raise ValueError(
                f'map_labels are for the {want} frame (MapLabelSpec frame="{want}"{preset}), but the sample is in '
                f'the {self.frame} frame: use SampleSpec(frame="{want}"), or frame=None in the MapLabelSpec for '
                "labels in the sample's frame")
        if not (isinstance(self.clip_length, int) and self.clip_length >= 1):
            raise ValueError(f"clip_length must be an int >= 1, got {self.clip_length!r}")
        if not (isinstance(self.clip_stride, int) and self.clip_stride >= 1):
            raise ValueError(f"clip_stride must be an int >= 1, got {self.clip_stride!r}")
        num_slots(self.future_s, self.trajectory_step_s)  # ValueError for a bad horizon or step
        num_slots(self.past_s, self.trajectory_step_s)
        for name in ("cameras", "lidars", "radars"):
            value = getattr(self, name)
            object.__setattr__(self, name, sensor_names(value))  # plain strings (the samples are pickled)

    @property
    def sensors(self) -> tuple[str, ...]:
        return (*self.cameras, *self.lidars, *self.radars)


class SampleDataset(Sequence):
    """Samples of a dataset's frames as dicts of arrays (lists of them for clips).

    Args:
        dataset: The dataset (or a filtered / sliced one).
        spec: What each sample holds.
        frames: Frames to use; by default ``dataset.frames(require=spec.sensors)``
            (or all frames with ``spec.require=False``). With
            ``spec.clip_length > 1`` a clip is every run of frames
            ``i - (N - 1) * stride, ..., i`` of one scene that are all in it.

    Raises:
        SensorUnavailable: If a required sensor is not downloaded (as :meth:`Dataset.frames`).
        MapUnavailable: With ``spec.map_labels`` and ``spec.require``, for the
            first scene without a map (``WITHHELD_BY_SPLIT`` in the test
            splits, ``NOT_DOWNLOADED``, ``NOT_INSTALLED``).
        MissingDependencyError: With ``spec.map_labels``, if lanelet2 has no ML converter.
    """

    def __init__(self, dataset: Dataset, spec: SampleSpec, frames: Optional[FrameList] = None) -> None:
        self.dataset = dataset
        self.spec = spec
        self.frames = frames if frames is not None else dataset.frames(require=spec.sensors if spec.require else ())
        self._clips = _clips(self.frames.refs, spec.clip_length, spec.clip_stride)
        self._converters: OrderedDict = OrderedDict()
        if spec.map_labels is not None:
            _check_map_labels(dataset, self.frames, spec.require, spec.map_labels)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_converters"] = OrderedDict()  # rebuilt per worker
        return state

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, i):
        if isinstance(i, slice):
            sliced = SampleDataset.__new__(SampleDataset)
            sliced.dataset, sliced.spec, sliced._clips = self.dataset, self.spec, self._clips[i]
            sliced._converters = OrderedDict()
            sliced.frames = FrameList(self.dataset, [(scene, k) for scene, ks in sliced._clips for k in ks])
            return sliced
        scene_id, indices = self._clips[i]
        scene = self.dataset.scene(scene_id)
        if self.spec.clip_length == 1:
            return make_sample(scene.frame(indices[0]), self.spec, converters=self._converters)
        return make_clip([scene.frame(k) for k in indices], self.spec, converters=self._converters)

    def __repr__(self) -> str:
        what = "frames" if self.spec.clip_length == 1 else "clips"
        return f"SampleDataset({what}={len(self)}, spec={self.spec})"


def _clips(refs: Sequence[tuple[str, int]], length: int, stride: int) -> list[tuple[str, tuple[int, ...]]]:
    """``(scene_id, frame indices)`` of every clip whose frames are all in *refs*, in the order of their last frame."""
    if length == 1:
        return [(scene_id, (index,)) for scene_id, index in refs]
    present = set(refs)
    out = []
    for scene_id, last in refs:
        indices = tuple(last - (length - 1 - j) * stride for j in range(length))
        if indices[0] >= 0 and all((scene_id, k) in present for k in indices):
            out.append((scene_id, indices))
    return out


def make_clip(frames: Sequence[Frame], spec: SampleSpec, *, converters: Optional[OrderedDict] = None) -> list[dict]:
    """The samples of a clip's frames (the oldest first), each with
    ``pose["T_last_base"]``: maps its base frame into the base frame of the clip's
    last frame (NaN where a pose is missing); ``T_last_road`` with ``spec.frame="road"``."""
    samples = [make_sample(frame, spec, converters=converters) for frame in frames]
    key = f"T_world_{spec.frame}"
    T_world_last = samples[-1]["pose"][key]
    T_last_world = np.linalg.inv(T_world_last) if np.isfinite(T_world_last).all() else np.full((4, 4), np.nan)
    for sample in samples:
        sample["pose"][f"T_last_{spec.frame}"] = T_last_world @ sample["pose"][key]
    return samples


def make_sample(frame: Frame, spec: SampleSpec, *, converters: Optional[OrderedDict] = None) -> dict:
    """The sample of one frame (docs/training.md, "Sample layout").

    *converters* keeps the map-label converters of recent scenes between calls
    (by scene ID; at most ``spec.map_labels.cache``).
    """
    scene = frame.scene
    missing = "raise" if spec.require else "none"
    ego = spec.frame  # "base" or "road"
    T_base_ego = invert_transform(scene.T_road_base) if ego == "road" else np.eye(4)
    pose = frame.pose if frame.availability("pose") is Availability.PRESENT else None
    sample: dict[str, Any] = {
        "meta": {
            "scene": scene.id, "split": scene.split, "index": frame.index, "token": frame.token,
            "t_ns": frame.t_ns, "pose_frame": scene.pose_frame, "frame": ego,
        },
        "pose": {f"T_world_{ego}": pose.T_world_base @ T_base_ego if pose is not None else np.full((4, 4), np.nan)},
    }

    cams = {}
    for name in spec.cameras:
        camera = scene.rig.cameras[name]
        if spec.image == "bytes":  # the full-size JPEG with its K; decode_cameras applies reduce
            image = frame.image_bytes(name, missing=missing)
        else:
            if spec.reduce > 1:
                camera = camera.scaled(1 / spec.reduce)
            image = frame.image(name, reduce=spec.reduce, image_correction=spec.image_correction, missing=missing)
        valid = image is not None
        if not valid:  # require=False: a black image of the size the image would have
            w, h = scene.rig.cameras[name].image_size
            if spec.image == "bytes":
                image = _black_jpeg(w, h)
            else:  # JPEG DCT scaling rounds up
                image = np.zeros((-(-h // spec.reduce), -(-w // spec.reduce), 3), np.uint8)
        cams[name] = {"image": image, "K": camera.K, f"T_cam_{ego}": camera.T_sensor_base @ T_base_ego,
                      "valid": valid}
        corr = stored_correction(scene, name) if valid and spec.image == "bytes" and spec.image_correction else None
        if corr is not None and not corr.is_identity:
            cams[name]["image_correction"] = corr
    sample["camera"] = cams

    if spec.lidars:
        present = [n for n in spec.lidars if spec.require or frame.availability(n) is Availability.PRESENT]
        if spec.lidar == "deskew_inputs":
            sample["lidar"] = {n: frame.lidar_deskew_inputs(n, frame=ego) for n in present}
        else:
            fused = frame.fused_lidar(present, frame=ego)
            source = np.array([spec.lidars.index(n) for n in fused.sensor], dtype=np.uint8)
            sample["lidar"] = {
                "xyz": fused.xyz,
                "reflectivity": np.asarray(fused.attrs.get("reflectivity", np.zeros(len(fused), np.float32))),
                "source": source[fused.source_index] if len(fused) else np.zeros(0, np.uint8),
            }

    if spec.radars:
        xyz, rr, rcs, src = [], [], [], []
        for k, name in enumerate(spec.radars):
            cloud = frame.radar(name, frame=ego, missing=missing)
            if cloud is None:
                continue
            cloud = cloud.compact()
            xyz.append(cloud.xyz)
            rr.append(np.asarray(cloud["range_rate"], np.float32))
            rcs.append(np.asarray(cloud.attrs.get("rcs", np.zeros(len(cloud), np.float32)), np.float32))
            src.append(np.full(len(cloud), k, np.uint8))
        sample["radar"] = {
            "xyz": np.concatenate(xyz) if xyz else np.zeros((0, 3), np.float32),
            "range_rate": np.concatenate(rr) if rr else np.zeros(0, np.float32),
            "rcs": np.concatenate(rcs) if rcs else np.zeros(0, np.float32),
            "source": np.concatenate(src) if src else np.zeros(0, np.uint8),
        }

    from kitscenes.e2e import EGO_WINDOW_S, past_motion

    history = frame.past(EGO_WINDOW_S, frame=ego, missing="none")
    motion = None if history is None else past_motion(history)
    sample["ego"] = {
        "velocity": np.zeros(3) if motion is None else motion["velocity"],
        "acceleration": np.zeros(3) if motion is None else motion["acceleration"],
        "yaw_rate": np.float64(0.0 if motion is None else motion["yaw_rate"]),
        "speed": np.float64(0.0 if motion is None else motion["speed"]),
        "valid": motion is not None,
    }

    for group, horizon_s in (("past", spec.past_s), ("future", spec.future_s)):
        k = num_slots(horizon_s, spec.trajectory_step_s)
        if k == 0:
            continue
        xyz, quat, valid = np.zeros((k, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (k, 1)), np.zeros(k, dtype=bool)
        load = frame.past if group == "past" else frame.future
        traj = load(horizon_s, step_s=spec.trajectory_step_s, frame=ego, missing="none")  # None: no pose at the frame
        if traj is not None:
            valid = traj.valid.copy()
            xyz[valid], quat[valid] = traj.xyz_ego[valid], traj.quat_ego[valid]
        sample[group] = {"xyz_ego": xyz, "quat_ego": quat, "valid": valid}
        if group == "future":
            from kitscenes.e2e import COMMAND_HORIZON_S, driving_command

            at = min(k, num_slots(COMMAND_HORIZON_S, spec.trajectory_step_s))
            sample[group]["command"] = np.int64(driving_command(xyz, valid, step_s=spec.trajectory_step_s))
            sample[group]["command_valid"] = bool(valid[at - 1])

    if spec.sd_map is not None:
        sample["sd_map"] = _sd_map_sample(frame, spec.sd_map, pose, scene.road_height_m if ego == "road" else 0.0)
    if spec.navigation is not None:
        sample["navigation"] = _navigation_sample(frame, spec.navigation, pose)
    if spec.map_labels is not None:
        sample["map"] = spec.map_labels.labels(frame, frame=ego, converters=converters)
    return sample


@functools.lru_cache(maxsize=8)
def _black_jpeg(width: int, height: int) -> bytes:
    from io import BytesIO

    from PIL import Image

    out = BytesIO()
    Image.new("RGB", (width, height)).save(out, format="JPEG")
    return out.getvalue()


def _sd_map_sample(frame: Frame, spec: SDMapSpec, pose, dz: float) -> dict:
    """The roads of :class:`SDMapSpec` around the frame's pose (base frame, plus *dz* for the road frame)."""
    from kitscenes.sdmap import attributes as attrs

    scene = frame.scene
    state = scene.availability("sd_map")
    if state is Availability.PRESENT and pose is None:
        state = frame.availability("pose")
    out = {"available": False, "state": state.value, "xyz": np.zeros((0, 3), np.float32),
           "length": np.zeros(0, np.int32), "way_id": np.zeros(0, np.int64), "road_class": np.zeros(0, np.int8),
           "is_link": np.zeros(0, bool), "one_way": np.zeros(0, bool), "lanes": np.zeros(0, np.int8),
           "speed_limit_mps": np.zeros(0, np.float32)}
    if state is not Availability.PRESENT:
        return out
    elements = scene.sd_map.elements(spec.region.around(pose), ways="roads", nodes="none")
    elements = elements.resampled(spec.points).in_ego(pose, road_height_m=scene.road_height_m)
    xyz, ids, cls, link, oneway, lanes, speed = [], [], [], [], [], [], []
    for way_id, points, tags in zip(elements.way_ids, elements.way_points, elements.way_tags):
        road_class, is_link = attrs.road_class(tags["highway"])
        if spec.classes is not None and road_class.value not in spec.classes:
            continue
        fwd, bwd = attrs.directions(tags)
        v, _ = attrs.speed_limit(tags, fwd)
        n_lanes = attrs.lanes(tags, fwd)[0] if fwd != bwd else attrs.lanes(tags, True)[0] + attrs.lanes(tags, False)[0]
        xyz.append(points + np.array([0.0, 0.0, dz]))
        ids.append(int(way_id))
        cls.append(road_class.rank)
        link.append(is_link)
        oneway.append(fwd != bwd)
        lanes.append(n_lanes)
        speed.append(np.nan if v is None else v / 3.6)  # the tags are in km/h
    if xyz:
        out.update(xyz=np.concatenate(xyz).astype(np.float32), length=np.full(len(xyz), spec.points, np.int32),
                   way_id=np.array(ids, np.int64), road_class=np.array(cls, np.int8), is_link=np.array(link, bool),
                   one_way=np.array(oneway, bool), lanes=np.array(lanes, np.int8),
                   speed_limit_mps=np.array(speed, np.float32))
    out["available"] = True
    return out


def _navigation_sample(frame: Frame, spec: NavigationSpec, pose) -> dict:
    """The command and target points of :class:`NavigationSpec` (UNKNOWN and no targets where there are none)."""
    from kitscenes.errors import Unavailable
    from kitscenes.navigation import Command

    k = spec.num_targets
    out = {"available": False, "state": Availability.PRESENT.value, "command": np.int64(Command.UNKNOWN),
           "target_points": np.full((k, 2), np.nan, np.float32), "target_valid": np.zeros(k, bool)}
    if pose is None:
        out["state"] = frame.availability("pose").value
        return out
    try:
        nav = frame.navigation(spec.rule, source=spec.source, targets=spec.targets, goal=spec.goal,
                               max_distance_m=spec.max_distance_m, spacing_m=spec.spacing_m)
    except Unavailable as err:
        out["state"] = err.state.value
        return out
    m = min(k, len(nav.target_points))
    out["target_points"][:m] = nav.target_points[:m]
    out["target_valid"][:m] = True
    out.update(available=True, command=np.int64(nav.command))
    return out


def _lift_polylines(map_sample: dict, dz: float) -> None:
    """Base-frame map labels of a sample into the road frame, in place: ``z += dz`` (``T_road_base`` is a pure z shift)."""
    groups = [*map_sample["lines"].values(), *map_sample["traffic_elements"].values(), map_sample["lane_segments"]]
    for polylines in groups:
        polylines["xyz"][:, 2] += np.float32(dz)


def _check_map_labels(dataset: Dataset, frames: FrameList, require: bool, spec: MapLabelSpec) -> None:
    """Fail early: no ML converter at all (or none for scene chaining), a recipe that the lanelet2 build or the
    dataset's ego motion does not allow, or (with *require*) a scene without a map."""
    from kitscenes.map import _lanelet2
    from kitscenes.map.labels import require_ml_converter
    from kitscenes.map.recipes import check_label_recipe

    require_ml_converter()
    if not check_label_recipe(spec.recipe)["use_imu"] and dataset.use_imu:
        raise ValueError(f"label recipe {spec.recipe!r} is made with the poses alone: open the dataset with "
                         "kitscenes.open(..., use_imu=False)")
    if spec.chaining == "scene":
        _lanelet2.require_reusable_map_data("MapLabelSpec(chaining=\"scene\"), chains built once per scene,")
    if not require:
        return
    for scene_id in dict.fromkeys(scene_id for scene_id, _ in frames.refs):
        err = dataset.scene(scene_id)._unavailable("map")
        if err is not None:
            raise type(err)(err.what, None, err.state,
                            f"scene {scene_id} ({err.detail}); SampleSpec(map_labels=...) needs a map in every "
                            "scene, or require=False for samples that say why they have none", err.hint)


#: ``sample["map"]["topology"]`` edges and their columns.
TOPOLOGY_COLUMNS = {
    "lane_to_lane": ("lane", "next_lane"),
    "te_to_centerline": ("te_class", "te", "centerline"),
    "te_to_te": ("te_class", "te", "other_te_class", "other_te"),
}
LANE_SEGMENT = "lane_segment"  # the class name in the keys of lane segments


def _map_sample(frame: Frame, spec: MapLabelSpec, converters: OrderedDict) -> dict:
    scene = frame.scene
    state = frame.availability("map")
    if state is Availability.PRESENT:
        state = frame.availability("pose")
    if state is not Availability.PRESENT:
        info = scene.split_info
        return {"available": False, "state": state.value,
                "map_changed": info is not None and info.family == "map_change",
                "lines": {name: _polylines(name, [], [], [], spec.dims) for name in spec.line_class_names},
                "traffic_elements": {name: _polylines(name, [], [], [], spec.dims) for name in spec.te_class_names},
                "lane_segments": _polylines(LANE_SEGMENT, [], [], [], spec.dims),
                "topology": {name: np.zeros((0, len(cols)), np.int64) for name, cols in TOPOLOGY_COLUMNS.items()}}
    converter = converters.get(scene.id)
    if converter is None:
        converter = spec.converter(scene)
        converters[scene.id] = converter
        while len(converters) > max(1, spec.cache):
            converters.popitem(last=False)
    else:
        converters.move_to_end(scene.id)
    labels = converter(frame.pose)
    lines = {}
    for name in spec.line_class_names:
        found = labels.lines.get(name, [])
        pieces = labels.line_pieces.get(name) or [0] * len(found)
        lines[name] = _polylines(name, found, labels.line_ids.get(name, []), pieces, spec.dims)
    elements = {}
    for name in spec.te_class_names:
        found = labels.traffic_elements.get(name, [])
        ids = [(int(i),) for i in labels.traffic_element_ids.get(name, [])]
        count: dict = {}
        pieces = []
        for i in ids:  # a cut element comes as several pieces with its id
            pieces.append(count.get(i, 0))
            count[i] = pieces[-1] + 1
        elements[name] = _polylines(name, found, ids, pieces, spec.dims)
    te_class = {name: k for k, name in enumerate(spec.te_class_names)}
    topology = {
        "lane_to_lane": labels.lane_to_lane,
        "te_to_centerline": [(te_class[t], i, j) for t, i, j in labels.te_to_centerline],
        "te_to_te": [(te_class[a], i, te_class[b], j) for a, i, b, j in labels.te_to_te],
    }
    return {"available": True, "state": Availability.PRESENT.value, "map_changed": bool(labels.map_changed),
            "lines": lines, "traffic_elements": elements,
            "lane_segments": _polylines(LANE_SEGMENT, labels.lane_segments, labels.lane_segment_ids,
                                        labels.lane_segment_pieces, spec.dims),
            "topology": {name: np.array(edges, dtype=np.int64).reshape(-1, len(TOPOLOGY_COLUMNS[name]))
                         for name, edges in topology.items()}}


def _polylines(name: str, lines: list, ids: list, pieces: list, dims: int) -> dict:
    """One class's polylines (N_i, D) concatenated: ``xyz`` (sum N_i, D) float32, ``length`` (L,) int32, the
    member ``ids`` concatenated with ``num_ids`` (L,), ``piece`` (L,) and the hashed ``key`` (L,)."""
    from kitscenes.map._vector import instance_key

    return {
        "xyz": np.concatenate([np.asarray(m, np.float32).reshape(-1, dims) for m in lines]) if lines
        else np.zeros((0, dims), np.float32),
        "length": np.array([len(m) for m in lines], dtype=np.int32),
        "ids": np.array([i for member_ids in ids for i in member_ids], dtype=np.int64),
        "num_ids": np.array([len(member_ids) for member_ids in ids], dtype=np.int32),
        "piece": np.array(pieces, dtype=np.int32),
        "key": np.array([instance_key(name, member_ids, piece) for member_ids, piece in zip(ids, pieces)],
                        dtype=np.int64),
    }


def _collate_polylines(groups: list[dict], tensor) -> dict:
    """Concatenate each class's polylines over the batch, with the sample of each polyline in ``batch``."""
    out = {}
    for name in dict.fromkeys(name for g in groups for name in g):
        present = [(b, g[name]) for b, g in enumerate(groups) if name in g]
        out[name] = {field: tensor(np.concatenate([p[field] for _, p in present])) for field in present[0][1]}
        out[name]["batch"] = tensor(np.concatenate([np.full(len(p["length"]), b, np.int64) for b, p in present]))
    return out


_RAGGED = {("lidar", "xyz"), ("lidar", "reflectivity"), ("lidar", "source"),
           ("radar", "xyz"), ("radar", "range_rate"), ("radar", "rcs"), ("radar", "source")}


def collate(samples: list) -> Union[dict, list[dict]]:
    """Batch samples: fixed-size arrays are stacked; point clouds are concatenated
    with a ``batch`` index per point; images of different sizes, JPEG bytes,
    deskew inputs and ``meta`` become lists.

    Decoded images are stacked channels first, (B, 3, H, W) uint8, the layout
    of :func:`kitscenes.decode.decode_cameras`; a single sample holds
    (H, W, 3) like ``frame.image()``.

    Clips (lists of N frame samples) give a list of N batches, one per clip
    position: ``torch.stack([b["camera"][cam]["image"] for b in batch], 1)``
    is (B, N, 3, H, W).

    Returns torch tensors if torch is installed, else numpy arrays.
    """
    if isinstance(samples[0], list):
        return [collate([clip[t] for clip in samples]) for t in range(len(samples[0]))]
    try:
        import torch

        def tensor(x: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x))
    except ImportError:  # pragma: no cover - exercised without torch
        def tensor(x: np.ndarray):
            return np.ascontiguousarray(x)

    out: dict[str, Any] = {"meta": {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}}
    out["pose"] = {k: tensor(np.stack([s["pose"][k] for s in samples])) for k in samples[0]["pose"]}

    if "camera" in samples[0]:
        cams = {}
        for name in samples[0]["camera"]:
            items = [s["camera"][name] for s in samples]
            images = [item["image"] for item in items]
            same = all(isinstance(im, np.ndarray) and im.shape == images[0].shape for im in images)
            cams[name] = {"image": tensor(np.stack(images).transpose(0, 3, 1, 2)) if same else images}
            for key in items[0]:  # K, T_cam_base (T_cam_road), valid
                if key not in ("image", "image_correction"):
                    cams[name][key] = tensor(np.stack([item[key] for item in items]))
            if any("image_correction" in item for item in items):  # per scene: None where it has none
                cams[name]["image_correction"] = [item.get("image_correction") for item in items]
        out["camera"] = cams

    for group in ("lidar", "radar"):
        if group not in samples[0]:
            continue
        if group == "lidar" and "xyz" not in samples[0]["lidar"]:
            out["lidar"] = [s["lidar"] for s in samples]  # deskew inputs: see motion_torch.collate_deskew_inputs
            continue
        fields = samples[0][group]
        batched = {k: tensor(np.concatenate([s[group][k] for s in samples])) for k in fields if (group, k) in _RAGGED}
        counts = [len(s[group]["xyz"]) for s in samples]
        batched["batch"] = tensor(np.repeat(np.arange(len(samples)), counts))
        out[group] = batched

    out["ego"] = {k: tensor(np.array([s["ego"][k] for s in samples])) for k in samples[0]["ego"]}

    for group in ("past", "future"):
        if group in samples[0]:
            out[group] = {k: tensor(np.stack([s[group][k] for s in samples])) for k in samples[0][group]}

    if "sd_map" in samples[0]:
        sds = [s["sd_map"] for s in samples]
        fields = [f for f in sds[0] if f not in ("available", "state")]
        out["sd_map"] = {"available": tensor(np.array([m["available"] for m in sds], dtype=bool)),
                         "state": [m["state"] for m in sds]}
        out["sd_map"].update({f: tensor(np.concatenate([m[f] for m in sds])) for f in fields})
        out["sd_map"]["batch"] = tensor(np.concatenate([np.full(len(m["length"]), b, np.int64)
                                                        for b, m in enumerate(sds)]))
    if "navigation" in samples[0]:
        navs = [s["navigation"] for s in samples]
        out["navigation"] = {"state": [m["state"] for m in navs]}
        out["navigation"].update({f: tensor(np.stack([np.asarray(m[f]) for m in navs]))
                                  for f in navs[0] if f != "state"})
    if "map" in samples[0]:
        maps = [s["map"] for s in samples]
        out["map"] = {
            "available": tensor(np.array([m["available"] for m in maps], dtype=bool)),
            "state": [m["state"] for m in maps],
            "map_changed": tensor(np.array([m["map_changed"] for m in maps], dtype=bool)),
            "lines": _collate_polylines([m["lines"] for m in maps], tensor),
            "traffic_elements": _collate_polylines([m["traffic_elements"] for m in maps], tensor),
            "lane_segments": _collate_polylines([{LANE_SEGMENT: m["lane_segments"]} for m in maps], tensor)[LANE_SEGMENT],
            "topology": {name: tensor(np.concatenate(
                [np.column_stack([np.full(len(m["topology"][name]), b, np.int64), m["topology"][name]])
                 for b, m in enumerate(maps)])) for name in TOPOLOGY_COLUMNS},
        }
    return out
