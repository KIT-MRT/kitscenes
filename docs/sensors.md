# Sensors, calibration, fusion

Cameras, lidars, radars and labels of a frame. Frames, units and missing data: [Conventions](conventions.md). GNSS/INS: [GNSS/INS](gnss_ins.md). Notebooks: `02_sensors_and_calibration.ipynb`, `03_labels.ipynb`.

**Contents:** [A frame](#a-frame) · [Sensor names](#sensor-names) · [Sensor data](#sensor-data) · [Calibration and fusion](#calibration-and-fusion) · [Lidar rows](#lidar-rows) · [Lidar scan layouts](#lidar-scan-layouts) · [Files of a frame](#files-of-a-frame) · [More per frame and scene](#more-per-frame-and-scene)

## A frame

```python
import kitscenes
import matplotlib.pyplot as plt

ds = kitscenes.open(splits="val")              # root from $KITSCENES_ROOT, or kitscenes.open("/data/kitscenes", splits="val")
scene = ds[0]                                  # or ds["<scene uuid>"]; len(ds), iter(ds), ds.scene_ids
print(scene.info)                              # split, city, lidar generation, frames; also scene.num_frames, .duration_s
frame = scene[50]                              # = scene.frame(50): one 10 Hz time step; iter(scene) gives every frame

image = frame.image("camera_ring_front", reduce=4)   # (568, 876, 3) uint8 RGB; reduce=1 for the full 3504 x 2272
cloud = frame.lidar("lidar_top")               # PointCloud in the vehicle frame (x forward, y left, z up), deskewed
xyz = cloud.xyz                                # (N, 3) float32, the measured points
radar = frame.radar("radar_front")             # RadarCloud, Doppler compensated
pose = frame.pose                              # EgoPose: xyz, quat (qx, qy, qz, qw), T_world_base at the frame time
                                               # (raises PoseUnavailable without poses)
world = frame.lidar("lidar_top", frame="world")   # the sweep in the world (map) frame
drive = scene.track.xyz                        # (num poses, 3) the whole drive, world frame
state = scene.ego(frame.t_ns)                  # pose, velocity, acceleration, yaw rate, curvature at any time

fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
left.imshow(image); left.axis("off")
right.scatter(xyz[:, 0], xyz[:, 1], s=0.05, c=xyz[:, 2], vmin=-3, vmax=3)
right.set(xlim=(-50, 50), ylim=(-50, 50), aspect="equal", xlabel="x forward (m)", ylabel="y left (m)")
plt.show()
```

## Sensor names

- `frame.sensors("camera")` (or `"lidar"`, `"radar"`): the sensors with data at this frame.
- `kitscenes.CAMERA_NAMES`, `CAMERA_RING_NAMES`, `LIDAR_NAMES`, `RADAR_NAMES`. `CAMERA_RING_NAMES` is the camera order of every converter: front, front_left, front_right, rear, rear_left, rear_right.
- A wrong name raises a `KeyError` that lists the valid ones.
- Every parameter that takes a sensor name accepts the string or a member of `kitscenes.CameraName`, `LidarName`, `RadarName` (`frame.lidar(kitscenes.LidarName.TOP)`). Members are strings: `LidarName.TOP == "lidar_top"`.
- Signatures use `Literal` names (`kitscenes.constants.CameraNameLike` etc.), so an IDE completes both and a type checker flags a typo or a lidar passed as a camera.
- Names the API returns (dict keys, `cloud.sensor`, sample keys) are always plain strings.

## Sensor data

- `frame.image(camera)`: (H, W, 3) uint8 RGB, decoded from the stored JPEG (`image_bytes()`). `scene.rig.cameras[camera].K` is (3, 3), `image_size` is (width, height).
- `frame.lidar(name)`: a `PointCloud` of the measured points. `xyz` (N, 3) float32 and per-point `attrs` as stored, the same for both lidar generations:
  `reflectivity` float32, `timestamp` float64 (the firing time), `ring` uint8, `sensor_id` uint8, `sensor_specific_data` uint16 (Hesai: bit 8 marks the second return), plus `row` uint32 (the row in the stored file).
- `frame.radar(name)`: a `RadarCloud`. `xyz` plus `range`, `azimuth`, `elevation` (relative to the radar), `range_rate` (ego motion removed), `rcs`, a `*_std` for each of the first four, `timestamp`, `sensor_id`, `detection_id`.
- Typed point columns: `cloud.reflectivity`, `.ring`, `.row`, `.frame_index` (accumulated clouds), `.point_t_ns` (int64 ns since the scene start; `attrs["timestamp"]` is the same in seconds). Radar: `.range`, `.azimuth`, `.elevation`, `.range_rate`, `.rcs`. `cloud.attrs` has every column by name.
- `frame.lidar(name, labels=True)`: the cloud with the ground labels as a bool column, `cloud["ground"]`. `frame.ground_mask(lidar)` is the same mask alone, aligned with `frame.lidar(lidar)`. If the lidar has a sweep but no labels, both raise `LabelsUnavailable`; with `missing="none"` the cloud comes without the column.
- `frame.instances(camera)`: the instance masks of a panoptic model (seamseg; front and rear ring cameras).
- `print()` of poses, trajectories, the rig, instance masks, map elements, patches and labels is one line (arrays as `<dtype (shape)>`).

## Calibration and fusion

```python
frame.sensors("lidar")                    # sensors with data at this frame
cam = scene.rig.cameras["camera_ring_front"]   # K, T_base_sensor, image_size
uv, depth, visible = cam.project(cloud.xyz)   # base-frame points -> pixels

small = frame.image("camera_ring_front", reduce=4)         # JPEG DCT scaling: much faster than full size
uv, _, visible = cam.scaled(1 / 4).project(cloud.xyz)

fused = frame.fused_lidar()               # all LiDARs, base frame
fused.sensor, fused.source_index          # which LiDAR each point came from
labelled = frame.lidar("lidar_top", labels=True)   # labelled["ground"]: (N,) bool; raises LabelsUnavailable
                                          # where there are no labels (INVALID if a label file does not fit its sweep)
ground = labelled.xyz[labelled["ground"]]
traj = frame.future(horizon_s=3.0)        # EgoTrajectory: a pose every 0.1 s ahead (t_ns, xyz_ego: the planning target,
                                          # quat_ego), NaN and not traj.valid past the end of the poses
hist = frame.past(horizon_s=1.0)          # the same backwards, the oldest slot first
world = frame.lidar("lidar_top", frame="world")   # through frame.T_world_base (also fused_lidar(), radar())
road = frame.lidar("lidar_top", frame="road")     # the road frame: base + (0, 0, scene.road_height_m), road at z ~ 0

raw = frame.lidar("lidar_top", frame="sensor", deskew=False, rows=True)   # the stored rows, as measured
raw.layout                                # ScanLayout(model='Hesai OT128', order='column-major', rings=128, columns=3600, ...)
ranges = raw.range_image()                # (returns, rings, columns) metres, NaN where there is no point
refl = raw.range_image("reflectivity")    # any attribute; ring 0 (top) is image row 0
ring, column, ret = raw.scan_position()   # per row, from raw.attrs["row"]
rays = frame.lidar("lidar_top", rows=True, no_returns=True)   # also the fired rays without a return (rays.no_return; xyz 0)
labels = frame.ground_mask("lidar_top", rows=True, no_returns=True)   # aligned with rays; False (no label) for the no-returns
```

The road frame: [Conventions](conventions.md#the-road-frame). Ego trajectories: [End-to-end driving](e2e.md#ego-trajectory-api).

**Ego motion and the IMU.** `scene.ego_motion` is built once per scene, on first use, and is what every cloud, `frame.pose`, `scene.ego()`, `accumulate` and the maps use. `scene.track` (every recorded pose) stays what `poses.txt` says.

```python
ego = scene.ego_motion
ego.source                                # "imu", "poses" or None
ego.reason                                # why not "imu", e.g. "no IMU data"
ego.qa                                    # "imu": gyro bias, pose against gyro residuals, ...

raw = frame.lidar("lidar_top", frame="sensor", deskew=False)
cloud = ego.compensate(raw)               # the same as frame.lidar("lidar_top")

ds = kitscenes.open(splits="val", use_imu=False)   # the poses alone, for everything
scene = kitscenes.Scene(path, calib_file="/path/to/other_calib.json")   # test another calibration
```

| `ego.source` | When | What it means |
| ------------ | ---- | ------------- |
| `"imu"` | The scene has IMU data and `gnss_ins_imu` in `calib.json` is valid (within 5° of the nominal axes); in release v2.0.0 every scene outside the test-e2e splits | Rotations refined with the 50 Hz gyro: changes faster than 1 s come from the gyro, slower ones from `poses.txt`, and the gyro bias is estimated per scene. Positions are those of `poses.txt` |
| `"poses"` | No IMU data (e.g., test-e2e splits), no valid `gnss_ins_imu`, unusable IMU data, or `use_imu=False` | Interpolation between the poses alone |
| `None` | No usable poses | Clouds are only transformed, `motion_compensated=False` and `reason` says why |

Missing IMU data is no error, the scene falls back and logs why at level `INFO`. `motion_compensated` is True for `"imu"` and `"poses"` alike. `ego.qa["flagged"]` means that the poses deviate noticeably from the gyro; the refinement is what corrects that.

**Image corrections:** the stored images of a camera can differ slightly from what `calib.json` describes. In release v2.0.0 every scene ships `calibration/image_corrections.json`, which warps `camera_base_front_right_rect` onto the rectified stereo pair of the calibration. The API corrects the cameras listed there wherever it decodes them: `frame.image()`, `SampleSpec` and `kitscenes.decode.decode_cameras`, at any `reduce` and for transcoded cameras. Other cameras are not touched. `frame.image(camera, image_correction=False)` and `SampleSpec(image_correction=False)` give the stored pixels; `frame.image_bytes()` is always the stored JPEG. `scene.files.image_corrections` holds the corrections, `kitscenes.image_corrections.warp_image` corrects an image you decoded yourself. NVIDIA DALI yields the stored JPEGs: apply `kitscenes.decode.apply_image_corrections` to them.

## Lidar rows

`frame.lidar(name)` returns the measured points in file order; `cloud.row` (`attrs["row"]`) is each point's row in the file. `frame.lidar(name, rows=True)` returns every stored row. Rows at the sensor origin (0, 0, 0) hold no point:

- **No-returns** (reflectivity -1): a fired ray that returned nothing (free space, e.g. sky), with its real firing time. Hesai and Velodyne lidars only: 1–5 % of a Hesai `lidar_top` sweep, 2–16 % of a Velodyne one, 5–40 % of a corner lidar's. `frame.lidar(name, rows=True, no_returns=True)` keeps them (xyz 0, flagged by `cloud.no_return`); `rows=True` alone does not.
- **Slots without a point** (reflectivity 0), in a `rows=True` cloud with `cloud.measured` False:
  - Hesai OT128 (`lidar_top`, 2026 generation): rings 0–23 and 88–127 fire only every second column, so exactly a quarter of its rows are slots that never fired (nominal column time). Plus 1–2.5 % returns closer than 1.4 m, which the extraction zeroes.
  - Velodyne `lidar_top` (2025): a fixed azimuth grid; the 1.4–1.7 % cells no firing fell into have timestamp 0.
- The Seyond lidars (`lidar_front/left/rear/right`) store only returns (unorganized), no rows at the origin.

Rows without a measured point stay at the origin in every frame (deskewing, `transformed`, `frame="world"`); `cloud.compact()` drops them. `frame.lidar()`, `frame.fused_lidar()`, samples, `scene.accumulate`, the voxel maps and the exports (`measured_only=True`) use only measured points.

`frame.ground_mask(name)` takes the same `rows` and `no_returns` as `frame.lidar(name)` and lines up with that cloud. The label files have one row per stored row without the no-returns: the label of a slot without a point means nothing, and the no-returns get False.

## Lidar scan layouts

The Hesai and Velodyne sweeps are organized; `cloud.layout` (a `kitscenes.points.ScanLayout`) says how:

| Lidar | Rings × columns | Order | Row |
|---|---|---|---|
| Hesai OT128 (`lidar_top`, 2026) | 128 × 3600 columns of 0.1° | column-major, two returns per slot | `row = (column * 2 + return) * rings + ring` |
| Hesai PandarXT-32 (`lidar_corner_*`) | 32 × 1500 (sometimes 1504) columns of 0.18°, only the 270° away from the car stored | column-major, two returns per slot | `row = (column * 2 + return) * rings + ring` |
| Velodyne VLS-128 (`lidar_top`, 2025) | 128 × 1708 columns | ring-major, one return | `row = ring * 1708 + column` |

- Ring 0 is the top laser; columns turn clockwise seen from above.
- Both store `pcl_width` × `pcl_height` in the file the same way, so a plain `reshape(pcl_height, pcl_width)` is right only for the Velodyne. Use `cloud.scan_position()` (ring, column, return per row) or `cloud.range_image()`.
- The Seyond lidars have `layout` None.

## Files of a frame

For path-based loaders and exporters, `frame.sensor_file(sensor)` says where the stored file of a camera, lidar or radar frame is, without reading it:

```python
f = frame.sensor_file("camera_ring_front")   # kitscenes.SensorFile
f.path, f.offset, f.size                     # the zip (or, extracted, the .jpg itself), the byte range of the JPEG in it
f.zip_member, f.in_zip                       # "<scene_id>/camera_ring_front/0000000050.jpg" (None if extracted), True
f.read()                                     # the bytes (= frame.image_bytes("camera_ring_front")); plain data, pickles
```

The zips are uncompressed, so `path[offset:offset + size]` is the file as stored. Camera and lidar members start at multiples of 4096 bytes. Any zip reader also finds the file by `zip_member`. `kitscenes.dali.write_camera_indices` builds NVIDIA DALI index files from it.

## More per frame and scene

```python
scene.info                                   # SceneInfo: split, city, lidar, frames, week, light, weather, road types, ... (scene table)
ds.filter(city="Frankfurt", lidar="hesai", weather="rain")   # any column of the scene table, see "Scene table and filters"

masks = frame.instances("camera_ring_front") # seamseg instance masks (front/rear ring cameras; model output)
classes, ids = masks.label_images()          # per-pixel Vistas class and instance id

T = scene.ego(scene.timestamps_ns[:2]).T_world_base   # (2, 4, 4), interpolated; also .velocity, .speed, ...
cam = scene.rig.cameras["camera_ring_front"].cropped(0, 600, 3504, 1200)  # intrinsics of an image crop (3504 x 2272 full)

imu = scene.imu(frame="base")                # 50 Hz gyro/accelerometer; also scene.messages("gnss", "navsatfix")
ins = scene.messages("gnss_ins", "insnavgeod", rate="frame")  # the INS solution assigned to each frame (index = frame)

from kitscenes.painting import paint, label
xyz = frame.fused_lidar().xyz                     # base frame of this frame
rgb, camera = paint(frame, xyz)              # ring camera colour per lidar point
classes, instance, camera = label(frame, xyz)   # seamseg class and instance per point

track = scene.track                          # every recorded pose: t_ns, xyz, quat arrays, origin, utm(), latlon(); track.pose(i)
tracks, failures = ds.read_tracks()          # every scene's track, city, lidar and map origin; no sensor data read

scene.files.path, scene.files.storage        # where the scene's files are; also .calib_file, .rig_original (the release
                                             # calibration), .image_variants (transcoded cameras), .image_corrections

cloud = scene.accumulate(range(0, 100, 5), voxel_size=0.1)   # lidar of many frames in the world frame (device="cuda" optional)
from kitscenes.export import write_pcd, write_ply
write_pcd("scene.pcd", cloud)                                # or write_ply; opens in CloudCompare, Open3D, PCL
```

`paint` and `label` do not check what a camera can see: a point hidden from a camera by something nearer (the road behind a parked car, a wall behind a passing one) takes the colour and class of what is in front of it.

`examples/export_scene.py` writes a whole scene as PCD/PLY point clouds, JPEG images, TUM poses and calibration JSON.
