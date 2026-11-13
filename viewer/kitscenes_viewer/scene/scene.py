"""KITScenes scene -> what the viewer needs: map, poses, point clouds, images.

Everything public here is already in the viewer's ENU frame (see `enu.py`), so
callers never juggle two coordinate systems. The dataset is read through the
official `kitscenes` package: it owns the layout (folders or packed archives),
the deskew, and the camera colours and seamseg labels of the points
(`kitscenes.painting`); none of that is re-derived here.
"""

from pathlib import Path

import numpy as np

from .enu import LocalFrame, SceneFrame
from .mapfeed import MapConverter

RING_CAMERAS = ("camera_ring_front", "camera_ring_front_left", "camera_ring_front_right",
                "camera_ring_rear_right", "camera_ring_rear", "camera_ring_rear_left")
# The only cameras the dataset ships seamseg masks for.
SEGMENTED_CAMERAS = ("camera_ring_front", "camera_ring_rear")


def open_scene(scene_dir):
    """The scene as a `kitscenes.Scene` (its split is the parent folder's name when that is one)."""
    from kitscenes import Scene
    return Scene(Path(scene_dir))


class NotPlaceable(Exception):
    """A scene whose poses cannot be put on the map; the message says why."""


def scene_origin(scene):
    """(lat, lon) of the origin of the scene's pose frame, as `kitscenes.aerial.placement_origin` finds it.

    That is the scene's `maps/origin.json`, or, for the test and
    map_change_test splits (no maps, poses in their city's map frame), its
    city's origin (`kitscenes.map.origin.CITY_ORIGINS`).

    Raises NotPlaceable when the scene has no such origin by design: the
    test-e2e splits' poses are relative to the scene's first pose, and a
    scene of a split without maps whose city is unknown has nothing to be
    placed with. Raises FileNotFoundError when a scene of a split with maps
    lacks its `maps/origin.json` (an incomplete copy).
    """
    from kitscenes import Unavailable
    from kitscenes.aerial import placement_origin
    try:
        origin = placement_origin(scene)
    except Unavailable as exc:
        info = scene.split_info
        if scene.pose_frame == "first_pose":
            raise NotPlaceable("%s poses are relative to the scene's first pose"
                               % ("the %s split's" % info.name if info else "the scene's")) from None
        if info is not None and not info.has_maps:
            raise NotPlaceable("the %s split ships no map origin and the scene's city is unknown"
                               % info.name) from None
        raise FileNotFoundError(exc.detail or str(exc)) from None
    return float(origin.lat), float(origin.lon)


def scene_poses(scene):
    """The scene's poses as {t: (N,3), q: (N,4) xyzw unit, time: (N,) seconds}.

    From `scene.track`: `poses.txt`, read in one request (no existence check
    first: on a network mount every check is a round trip). Raises
    FileNotFoundError without the file -- the city scan takes that as "not a
    scene, look inside" -- and ValueError for a file without poses.
    """
    from kitscenes import Availability
    track = scene.track
    if not len(track):
        if scene.availability("pose") is Availability.NOT_DOWNLOADED:
            raise FileNotFoundError("no poses.txt in %s" % scene.files.path)
        raise ValueError("no poses in %s" % scene.files.path)
    return {
        "time": track.t_ns.astype(np.float64) / 1e9,
        "t": track.xyz.copy(),
        "q": track.quat / np.linalg.norm(track.quat, axis=1, keepdims=True),
    }


def scene_reference_timestamps(scene):
    """One timestamp per frame, in seconds since the scene start (`scene.timestamps_ns`).

    Raises FileNotFoundError without `timestamp.reference.txt` (every
    release scene has one) and `kitscenes.errors.DataFormatError` (a
    ValueError) for a malformed file, e.g. epoch nanoseconds instead of
    relative seconds.
    """
    return scene.timestamps_ns / 1e9


def track_enu(poses, timestamps, frame):
    """Ego pose at every frame time, in viewer ENU: (positions (N,3), xyzw (N,4)).

    Vectorised, because a city holds tens of thousands of frames. The heading
    is turned by the meridian convergence *where the car is*, which is a
    property of the place, not of the scene.
    """
    from scipy.spatial.transform import Rotation, Slerp
    times = poses["time"]
    t = np.clip(np.asarray(timestamps, dtype=np.float64), times[0], times[-1])
    if len(times) > 1:
        rots = Slerp(times, Rotation.from_quat(poses["q"]))(t)
    else:
        rots = Rotation.from_quat(np.repeat(poses["q"][:1], len(t), axis=0))
    pos = np.column_stack([np.interp(t, times, poses["t"][:, i]) for i in range(3)])
    xy = frame.apply_array(pos[:, :2])
    spin = Rotation.from_euler("z", np.radians(frame.rotation_deg_array(pos[:, :2]))[:, None])
    return np.column_stack([xy, pos[:, 2]]), (spin * rots).as_quat()


def map_bbox_centre(lanelet_map, origin):
    """Centre of a map's lat/lon bbox, which is the anchor JOSM picks for it."""
    import pyproj
    from .enu import utm_epsg
    xy = np.array([[p.x, p.y] for p in lanelet_map.pointLayer], dtype=np.float64)
    if not len(xy):
        return origin
    epsg = utm_epsg(*origin)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    inv = pyproj.Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    e0, n0 = fwd.transform(origin[1], origin[0])
    lon, lat = inv.transform(xy[:, 0] + e0, xy[:, 1] + n0)
    return ((float(np.min(lat)) + float(np.max(lat))) / 2,
            (float(np.min(lon)) + float(np.max(lon))) / 2)


class MapFile:
    """A Lanelet2 map on disk -- a city-wide one, say -- loaded about `origin`.

    The origin only has to be the one the features are then converted with:
    lanelet2 projects lat/lon into that frame on load, and `SceneFrame` takes
    the same frame back out to the viewer's ENU.
    """

    def __init__(self, path, origin):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError("map not found: %s" % self.path)
        self.origin = (float(origin[0]), float(origin[1]))
        self._map = None

    @property
    def lanelet_map(self):
        if self._map is None:
            import lanelet2
            from lanelet2.io import Origin
            from lanelet2.projection import UtmProjector
            self._map = lanelet2.io.load(str(self.path), UtmProjector(Origin(*self.origin)))
        return self._map

    def bbox_centre(self):
        return map_bbox_centre(self.lanelet_map, self.origin)

    def features(self, anchor):
        return MapConverter(SceneFrame(*self.origin, *anchor)).features(self.lanelet_map)


class SceneSource:
    """One recorded scene, in the viewer's frame.

    The scene is placed like the city index places it (`scene_origin`). Its
    map is `map_path` when given, else its own `maps/map.osm`, else -- the
    test and map_change_test splits ship none -- its city's HD map from the
    dataset's `lanelet2_hd_maps/` (`kitscenes.Dataset.city_maps`), else none.
    A scene that cannot be placed (the test-e2e splits: poses relative to the
    first pose) is shown in its own metres (`LocalFrame`), without a map or
    an anchor. `notes` says why whenever the scene does not get its own map.
    """

    def __init__(self, scene_dir, anchor=None, map_path=None):
        given = Path(scene_dir).expanduser()
        self.dir = given.resolve()
        if not self.dir.is_dir():
            raise FileNotFoundError("scene directory not found: %s" % self.dir)
        self.scene = open_scene(self.dir)
        # The name as given, not as resolved: a symlinked scene directory
        # keeps the id (and cache directory) the city index knows it by.
        self.scene_id = given.name or self.dir.name
        # The dataset root as given too: `data` may be a link elsewhere.
        self._given_dir = given.absolute()
        self.notes = []
        try:
            self.origin_lat, self.origin_lon = scene_origin(self.scene)
            self.placed = True
        except NotPlaceable as exc:
            self.origin_lat = self.origin_lon = None
            self.placed = False
            self.notes.append("not placed on the map, %s: shown in its own metres, without a map" % exc)
        self._map_arg = Path(map_path).expanduser().resolve() if map_path else None
        self._projector = None
        self._lanelet_map = None
        self.poses = scene_poses(self.scene)
        self.timestamps = scene_reference_timestamps(self.scene)
        self.frame_count = min(len(self.poses["time"]), len(self.timestamps))
        from kitscenes import LIDAR_NAMES
        self.lidars = [s for s in self.scene.sensors if s in LIDAR_NAMES]
        self.cameras = [c for c in RING_CAMERAS if c in self.scene.sensors]
        self.set_anchor(anchor or self._default_anchor())

    def set_anchor(self, anchor):
        """Point the ENU frame at a new anchor (JOSM's, when it owns the map).

        A scene that is not placed has no anchor and keeps its own metres.
        """
        if not self.placed:
            self.anchor = None
            self.frame = LocalFrame()
            return
        self.anchor = (float(anchor[0]), float(anchor[1]))
        self.frame = SceneFrame(self.origin_lat, self.origin_lon, *self.anchor)

    # -- map ---------------------------------------------------------------
    @property
    def map_path(self):
        """The Lanelet2 map this scene is shown on, or None (see the class docstring); looked up once."""
        if not hasattr(self, "_map_path"):
            self._map_path = self._find_map()
        return self._map_path

    def _find_map(self):
        from kitscenes import Availability
        if not self.placed:
            if self._map_arg:
                self.notes.append("--map %s is not used: the scene cannot be placed on it" % self._map_arg)
            return None
        if self._map_arg:
            return self._map_arg
        if self.scene.availability("map") is Availability.PRESENT:
            return self.dir / "maps" / "map.osm"
        info = self.scene.info
        city = info.city if info is not None else None
        root = self._given_dir.parent.parent.parent if self._given_dir.parent.parent.name == "data" else None
        if city is None or root is None:
            self.notes.append("no map of its own and %s, so no city map either: opened without a map "
                              "(--map PATH gives it one)"
                              % ("its city is unknown" if city is None
                                 else "it is not in a dataset folder (<root>/data/<split>/<id>)"))
            return None
        import kitscenes
        from kitscenes.constants import HD_MAPS_DIR
        from kitscenes.errors import KITScenesError
        try:
            found = kitscenes.open(root).city_maps().get(city)
        except (KITScenesError, OSError) as exc:
            self.notes.append("no map of its own, and the dataset at %s could not be opened for its city map "
                              "(%s): opened without a map (--map PATH gives it one)" % (root, exc))
            return None
        if found is None:
            self.notes.append("no map of its own and no %s/%s/%s.osm: opened without a map "
                              "(--map PATH gives it one)" % (root, HD_MAPS_DIR, city))
            return None
        self.notes.append("no map of its own: shown on the %s city map %s" % (city, found))
        return found

    @property
    def projector(self):
        if self._projector is None:
            from lanelet2.io import Origin
            from lanelet2.projection import UtmProjector
            self._projector = UtmProjector(Origin(self.origin_lat, self.origin_lon))
        return self._projector

    @property
    def lanelet_map(self):
        """The map at `map_path` in the scene's frame; FileNotFoundError when there is none."""
        if self._lanelet_map is None:
            if self.map_path is None:
                raise FileNotFoundError("scene %s has no map: %s" % (self.scene_id, "; ".join(self.notes)))
            import lanelet2
            self._lanelet_map = lanelet2.io.load(str(self.map_path), self.projector)
        return self._lanelet_map

    def _default_anchor(self):
        """Centre of the map's lat/lon bbox (see `map_bbox_centre`); the origin without a map."""
        if not self.placed:
            return None
        if self.map_path is None:
            return self.origin_lat, self.origin_lon
        return map_bbox_centre(self.lanelet_map, (self.origin_lat, self.origin_lon))

    def map_features(self):
        return MapConverter(self.frame).features(self.lanelet_map)

    # -- poses -------------------------------------------------------------
    def pose_at(self, time_s):
        """Ego pose interpolated at `time_s`, held at the ends: (R 3x3, t (3,)) in scene metres."""
        times = self.poses["time"]
        t = float(np.clip(time_s, times[0], times[-1]))
        if len(times) < 2:  # a single pose
            from scipy.spatial.transform import Rotation
            return Rotation.from_quat(self.poses["q"][0]).as_matrix(), self.poses["t"][0].copy()
        T = self.scene.ego(int(round(t * 1e9))).T_world_base
        return T[:3, :3], T[:3, 3]

    def _frame_pose(self, index):
        """(4, 4) base -> scene frame at the time of frame `index`."""
        rot, pos = self.pose_at(float(self.timestamps[index]))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rot, pos
        return T

    # -- point clouds ------------------------------------------------------
    def to_enu(self, xyz):
        """(N,3) scene metres -> viewer ENU. ~0.12 s for a 750 k-point frame."""
        if not len(xyz):
            return xyz
        xy = self.frame.apply_array(xyz[:, :2])
        return np.column_stack([xy[:, 0], xy[:, 1], xyz[:, 2]])

    def frame_points_scene(self, index, lidars=None):
        """All LiDARs of one frame, in the scene's own metric frame.

        Returns (xyz float64 (N,3), intensity float32 (N,), sensor uint8 (N,)):
        the measured points of every sweep, deskewed to the frame time by
        `kitscenes` (`Frame.fused_lidar`), then moved by the frame's pose;
        `sensor` is the dataset's `sensor_id` column, which the browser's
        legend names. This does not depend on the viewer's ENU anchor.
        """
        from kitscenes import transform_points
        frame = self.scene.frame(int(index))
        present = set(frame.sensors("lidar"))
        names = [n for n in (lidars or self.lidars) if n in present]
        if not names:
            return np.zeros((0, 3)), np.zeros(0, np.float32), np.zeros(0, np.uint8)
        cloud = frame.fused_lidar(names)
        xyz = transform_points(self._frame_pose(index), cloud.xyz.astype(np.float64))
        intensity = cloud.attrs.get("reflectivity", np.zeros(len(xyz), np.float32))
        sensor = cloud.attrs.get("sensor_id", np.zeros(len(xyz), np.uint8))
        return xyz, np.asarray(intensity, np.float32), np.asarray(sensor, np.uint8)

    def ego_track(self):
        """Every frame's ego pose in viewer ENU: positions and xyzw quaternions.

        Small enough (a few hundred frames) to travel with the scene
        description, which saves the browser a round trip per frame for the
        camera lock.
        """
        # The same function draws the city's trajectories, so a scene picked
        # there lands exactly on the line it was picked from.
        pos, quat = track_enu(self.poses, self.timestamps[:self.frame_count], self.frame)
        positions = [[round(float(v), 4) for v in row] for row in pos]
        quats = [[round(float(v), 7) for v in row] for row in quat]  # xyzw, as three.js wants
        return positions, quats

    # -- colours and labels from the cameras -------------------------------
    def _base_points(self, index, xyz):
        """Scene-frame points -> the base frame of frame `index`."""
        from kitscenes import invert_transform, transform_points
        return transform_points(invert_transform(self._frame_pose(index)), np.asarray(xyz, np.float64))

    def has_masks(self):
        """Whether the scene has seamseg masks (the class and instance colour modes)."""
        if not hasattr(self, "_has_masks"):
            self._has_masks = self.class_palette() is not None
        return self._has_masks

    def frame_labels(self, index, xyz, cameras=SEGMENTED_CAMERAS):
        """(Vistas class, instance number) per point, from the seamseg masks (`kitscenes.painting.label`).

        Instance numbers are per frame, not tracked across frames: seamseg does
        not track, so they only separate one car from another within a frame.
        Each camera's numbers are offset by 1000 so the two cannot collide.
        """
        from kitscenes.painting import label
        classes, instances, camera = label(self.scene.frame(int(index)), self._base_points(index, xyz),
                                           cameras=cameras)
        numbers = np.where(instances > 0, camera.astype(np.int64) * 1000 + instances, 0)
        return classes, numbers.astype(np.uint16)

    def frame_rgb(self, index, xyz, cameras=None):
        """Camera colour per point; neutral grey where no camera saw it (`kitscenes.painting.paint`).

        All six ring cameras are used, not just the two with masks, so most
        of a frame gets a colour. Images are decoded at half resolution, which
        a JPEG can do from its DCT coefficients at about four times the speed;
        a point needs only one pixel.
        """
        from kitscenes.painting import paint
        rgb, _camera = paint(self.scene.frame(int(index)), self._base_points(index, xyz),
                             cameras=cameras or self.cameras, reduce=2)
        return rgb

    # -- cameras -----------------------------------------------------------
    def camera_info(self):
        """Per ring camera: intrinsics, image size and the mount in the ego frame."""
        out = {}
        for name in self.cameras:
            camera = self.scene.rig.cameras.get(name)
            if camera is None:
                continue
            width, height = camera.image_size
            out[name] = {
                "fx": float(camera.K[0, 0]),
                "fy": float(camera.K[1, 1]),
                "cx": float(camera.K[0, 2]),
                "cy": float(camera.K[1, 2]),
                "width": int(width) or None,
                "height": int(height) or None,
                # Maps camera-frame points into the ego (base) frame (T_<to>_<from>), row-major 4x4. The camera frame
                # is OpenCV: x right, y down, z forward.
                "T_ego_cam": [[float(v) for v in row] for row in camera.T_base_sensor],
            }
        return out

    def image_bytes(self, camera, index):
        """The stored JPEG of a ring camera at frame `index`, or None (folders and packed scenes alike)."""
        try:
            return self.scene.frame(int(index)).image_bytes(camera, missing="none")
        except (IndexError, KeyError, OSError):
            return None

    # -- description for the browser --------------------------------------
    def class_palette(self):
        """Class colours and names for the browser, or None without masks."""
        from kitscenes import Availability
        from kitscenes.instances import CLASS_NAMES, UNLABELLED, palette
        cameras = [c for c in SEGMENTED_CAMERAS if c in self.scene.rig.cameras
                   and self.scene.availability("instances/" + c, 0) is Availability.PRESENT]
        if not cameras:
            return None
        return {
            "palette": palette().tolist(),
            "names": {str(train_id): name for train_id, name in CLASS_NAMES.items()},
            "unlabelled": UNLABELLED,
            "cameras": cameras,
        }

    def describe(self, base_url):
        positions, quats = self.ego_track()
        return {
            "type": "scene",
            "scene": {
                "id": self.scene_id,
                "base": base_url,
                "frames": int(self.frame_count),
                "times": [round(float(t), 4) for t in self.timestamps[:self.frame_count]],
                "lidars": self.lidars,
                "ego": {"pos": positions, "quat": quats},
                "cameras": self.camera_info(),
                # None for a scene that is not placed: its own metres, on no map.
                "anchor": {"lat": self.anchor[0], "lon": self.anchor[1]} if self.anchor else None,
                # The Vistas palette travels with the scene, so the browser
                # has one source of truth for class colours and names.
                "classes": self.class_palette(),
            },
        }
