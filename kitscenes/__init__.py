"""kitscenes: Python API for the KITScenes Multimodal autonomous driving dataset.

Example::

    import kitscenes

    ds = kitscenes.open("/data/kitscenes", splits="val")   # or $KITSCENES_ROOT
    scene = ds[0]
    frame = scene[50]                                      # = scene.frame(50)
    image = frame.image("camera_ring_front")               # (H, W, 3) uint8
    cloud = frame.lidar("lidar_top")                       # PointCloud, base frame, deskewed
    lanes = scene.map.lanes(kitscenes.map.Circle.around(frame.pose, 50.0))

Training: ``kitscenes.SampleDataset(ds, kitscenes.SampleSpec(...))`` with
``collate_fn=kitscenes.collate`` in a torch ``DataLoader``.

Public are the names in ``__all__`` of ``kitscenes`` (and ``kitscenes.open``,
left out of it so that ``from kitscenes import *`` does not shadow the
builtin) and of the subpackages and modules the docs name: ``kitscenes.map``,
``.sdmap``, ``.aerial``, ``.visualization``, ``.navigation``, ``.e2e``,
``.decode``, ``.dali``, ``.mapping``, ``.accumulate``, ``.ego_trajectory``,
``.motion_torch``, ``.splits``, ``.export``, ``.transcode`` and ``.errors``.
Everything else, with or without a leading underscore, may change. lanelet2
is only imported when a map is loaded.
"""

from kitscenes._version import __version__  # the source's (an editable install's metadata can be stale)

__license__ = "Apache-2.0"

from kitscenes.catalog import Dataset, FrameList
from kitscenes.catalog import open as open  # the redundant alias marks it exported for type checkers (not in __all__)
from kitscenes.constants import CAMERA_NAMES, CAMERA_RING_NAMES, LIDAR_NAMES, RADAR_NAMES, CameraName, LidarName, RadarName
from kitscenes.ego_trajectory import EgoState, EgoTrajectory
from kitscenes.errors import (
    AerialUnavailable,
    Availability,
    CalibrationError,
    ChecksumError,
    DataFormatError,
    DatasetNotFound,
    ImageSizeError,
    KITScenesError,
    LabelsUnavailable,
    MapUnavailable,
    MissingDependencyError,
    PoseUnavailable,
    SceneNotFound,
    SensorUnavailable,
    Unavailable,
)
from kitscenes.frame import Frame
from kitscenes.points import PointCloud, RadarCloud
from kitscenes.rig import Camera, Rig, invert_transform, transform_points
from kitscenes.samples import MapLabelSpec, NavigationSpec, SampleDataset, SampleSpec, SDMapSpec, collate
from kitscenes.scene import Scene
from kitscenes.schema import EgoPose, SensorFile
from kitscenes.splits import SPLITS, find_scenes, scene_table

from kitscenes import map  # noqa: E402,F401,A004  (kitscenes.map after "import kitscenes"; no lanelet2 import)
from kitscenes import e2e  # noqa: E402,F401  (kitscenes.e2e after "import kitscenes")
from kitscenes import navigation, sdmap  # noqa: E402,F401  (no OSM file is read on import)

__all__ = [
    # data model (kitscenes.open is the entry point)
    "Dataset",
    "FrameList",
    "Scene",
    "Frame",
    "PointCloud",
    "RadarCloud",
    "Rig",
    "Camera",
    "EgoPose",
    "EgoState",
    "EgoTrajectory",
    "SensorFile",
    # training
    "SampleSpec",
    "MapLabelSpec",
    "SDMapSpec",
    "NavigationSpec",
    "SampleDataset",
    "collate",
    # geometry
    "transform_points",
    "invert_transform",
    # names
    "CameraName",
    "LidarName",
    "RadarName",
    "CAMERA_NAMES",
    "CAMERA_RING_NAMES",
    "LIDAR_NAMES",
    "RADAR_NAMES",
    "SPLITS",
    "scene_table",
    "find_scenes",
    # errors
    "Availability",
    "KITScenesError",
    "Unavailable",
    "SensorUnavailable",
    "LabelsUnavailable",
    "MapUnavailable",
    "PoseUnavailable",
    "AerialUnavailable",
    "MissingDependencyError",
    "DatasetNotFound",
    "SceneNotFound",
    "DataFormatError",
    "CalibrationError",
    "ChecksumError",
    "ImageSizeError",
]
