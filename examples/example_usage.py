"""A tour of the kitscenes API in one script: dataset, scene, frame, calibration, sensor data, poses, map, samples.

Each step prints what it got, so the output doubles as a cheat sheet (test/test_real_data.py runs it as a smoke
test). Needs a val scene and ``kitscenes[map]`` for the map part.

Run::

    KITSCENES_ROOT=/data/kitscenes python examples/example_usage.py
"""

import numpy as np

import kitscenes

# 1. dataset -> scenes
ds = kitscenes.open(splits="val")                    # root from $KITSCENES_ROOT
print(ds)
scene = ds[0]                                        # or ds["<uuid>"]; ds[0:2] is a sub-dataset
print(scene.id, scene.split, scene.num_frames, scene.duration_s, scene.pose_frame)

# 2. scene -> frame
frame = scene.frame(scene.num_frames // 2)
print(frame.t_ns, frame.pose.xyz)            # pose interpolated at the frame time
print(frame.sensors())                               # sensors with data here

# 3. calibration
cam = scene.rig.cameras["camera_ring_front"]
print(cam.K, cam.T_base_sensor, cam.image_size)

# 4. sensor data: decoded on each call, owned by the caller
image = frame.image("camera_ring_front", reduce=2)   # (H/2, W/2, 3) uint8
cloud = frame.lidar("lidar_top")                     # base frame, deskewed
print(cloud, cloud.motion_compensated, cloud.reason)
uv, depth, visible = cam.scaled(0.5).project(cloud.xyz)
print(f"{visible.sum()} lidar points visible in the front camera")
raw = frame.lidar("lidar_top", frame="sensor", deskew=False, rows=True, no_returns=True)   # the stored rows and the rays without a return
print(raw.layout, int(raw.no_return.sum()), "no-returns")   # Hesai column-major, Velodyne ring-major
print(raw.range_image().shape)                       # (returns, rings, columns), metres, NaN where no point
radar = frame.radar("radar_front")
print(radar, np.abs(radar["range_rate"]).min())     # static returns read ~0 m/s

# 5. derived products
fused = frame.fused_lidar()                          # all lidars in one cloud
print(fused.xyz.shape, fused.sensor, fused.provenance)
traj = frame.future(horizon_s=3.0)                   # a pose every 0.1 s; xyz_ego is the planning target
print(traj.t_ns[-1] - traj.anchor_t_ns, traj.xyz_ego[-1], traj.valid.all())   # not valid: past the poses
hist = frame.past(horizon_s=1.0)                     # the same backwards, the oldest first
state = scene.ego(frame.t_ns)                        # pose and motion at any time
print(state.speed, state.yaw_rate, state.curvature)
world = frame.lidar("lidar_top", frame="world")      # the sweep in the world frame

# 6. map (train/val splits; raises MapUnavailable in the test splits)
from kitscenes.map import Region, project_to_camera

try:
    smap = scene.map
    region = Region(ahead=60, behind=20, left=20, right=20).around(frame.pose)   # a map-frame Box
    lanes = smap.lanes(region, kinds=["road"])
    matches = smap.match(frame.pose)                 # lane under the vehicle, heading-aware
    print(len(lanes), [(m.lane_id, round(m.heading_error, 3)) for m in matches])
    print({lane.id: lane.speed_limit_kph for lane in lanes[:3]})
    patch = smap.query(region).in_ego(frame.pose)    # all elements in the vehicle frame
    print(len(patch.lines), len(patch.stop_lines), len(patch.traffic_signs))
    edges = np.concatenate([lane.left_points for lane in lanes])
    uv, depth, visible = project_to_camera(edges, frame.pose, cam)
    print(f"{visible.sum()} lane-edge points visible in the front camera")
except kitscenes.MapUnavailable as err:
    print("no map:", err.state, err)                # e.g. Availability.WITHHELD_BY_SPLIT

# 7. training: samples as dicts of arrays for a torch DataLoader (collate_fn=kitscenes.collate)
spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=4, lidars=("lidar_top",), future_s=3.0, past_s=1.0)
samples = kitscenes.SampleDataset(ds, spec)          # frames where every sensor of the spec has data
batch = kitscenes.collate([samples[0], samples[1]])  # what DataLoader(samples, collate_fn=kitscenes.collate) yields
print(len(samples), batch["camera"]["camera_ring_front"]["image"].shape, batch["lidar"]["xyz"].shape)
print(batch["future"]["xyz_ego"].shape, batch["past"]["valid"].shape)   # (2, 30, 3), (2, 10)

clips = kitscenes.SampleDataset(ds, kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=8, clip_length=3))
clip_batch = kitscenes.collate([clips[0], clips[1]])  # one batch per clip position, the current frame last
print(len(clip_batch), clip_batch[0]["pose"]["T_last_base"].shape)
