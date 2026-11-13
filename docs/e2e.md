# End-to-end driving (test-e2e)

The `test-e2e` split (107 scenes) and `map_change_test-e2e` (5 scenes) are for end-to-end driving: predict where the ego vehicle drives.

**Contents:** [Protocol](#protocol) · [What the scenes ship](#what-the-scenes-ship) · [Submission and metrics](#submission-and-metrics) · [Ego trajectory API](#ego-trajectory-api) · [Driving command and ego status](#driving-command-and-ego-status)

## Protocol

As on the [dataset card](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal):

- **Input:** the first 4 s of a scenario, 41 frames at 10 Hz.
- **Task:** predict the ego trajectory of the next 3 s or 5 s, open loop.
- The future is withheld: a test-e2e scene ends after the 41 input frames.
- Train on `train` / `val`, where `frame.future()` gives the target.

## What the scenes ship

- All cameras, lidars and radars of the 41 frames, the calibration and the poses.
- Poses relative to the first pose instead of the map frame (`scene.pose_frame`, [Conventions](conventions.md#poses)): the first position is 0, the orientation keeps the world yaw.
- No maps, no labels, no GNSS/INS (`WITHHELD_BY_SPLIT`, [Conventions](conventions.md#missing-data)), so no IMU refinement of the ego motion either (`scene.ego_motion.source` is `"poses"`). The scene table gives their city but no recording order and no road types ([Scene table](scene_table.md#the-test-e2e-splits)).
- `scene.num_frames` is 41. `scene.info.num_frames` (the scene table) counts the frames of the full recording.

```python
ds = kitscenes.open(splits=["test-e2e", "map_change_test-e2e"])
scene = ds[0]
last = scene[scene.num_frames - 1]               # the last input frame (index 40)
hist = last.past(horizon_s=4.0)                  # the ego history, in the base frame of the last frame
```

## Submission and metrics

Scoring is `kitscenes-e2e eval` (`kitscenes_e2e.eval`, numpy only; kitscenes-e2e README section "Evaluation"). `kitscenes-e2e` also builds the test-e2e samples, a local validation protocol on `val` ("val-e2e": windows of 4 s input and 5 s ground truth) and kinematic baselines.

A submission is one JSON file (format `kitscenes-e2e/planning-pred`, version 1):

```json
{"format": "kitscenes-e2e/planning-pred", "version": 1, "step_s": 0.5,
 "meta": {"method": "MyPlanner"},
 "samples": {"<token>": [[x, y, yaw], [x, y, yaw], ...]}}
```

- One sample per scene, anchored at the last input frame (index 40). Its key is `frame.token`, `"<scene id>_0040"` ([Conventions](conventions.md#time)).
- Row k (k = 1..T) is the pose at `anchor + k * step_s`: x, y in metres in the **base frame of the anchor** (x forward, y left; not the road frame), optionally the yaw in radians relative to the anchor heading. The third column is always yaw, never z.
- `step_s` is 0.5 (6 or 10 rows) or 0.1 (30 or 50 rows; every 5th is scored). The horizon is 3 s or 5 s, the same for all samples of a file.
- Every scene needs a prediction. Unknown tokens, NaN and wrong shapes are errors. `.npz` works too.
- `kitscenes-e2e eval --pred submission.json` checks the format.

```python
import json

samples = {}
for scene in kitscenes.open(splits="test-e2e"):
    anchor = scene[scene.num_frames - 1]            # frame 40
    xy = my_planner(anchor)                          # (6, 2): 0.5 s .. 3.0 s, base frame of the anchor
    samples[anchor.token] = xy.tolist()
with open("submission.json", "w") as f:
    json.dump({"format": "kitscenes-e2e/planning-pred", "version": 1, "step_s": 0.5,
               "meta": {"method": "my_planner"}, "samples": samples}, f)
```

Metrics, per sample and then averaged, always at 2 Hz (d at t: the xy distance to the ground truth at time t):

| Key | Definition |
|---|---|
| `L2_at@t` (t = 1, 2, 3, 5 s) | d at t (UniAD's "L2 at t") |
| `L2_avg@t` | mean of d from 0.5 s up to t (ST-P3 / VAD "L2 at t", most nuScenes tables) |
| `ADE@3s`, `ADE@5s` / `FDE@3s`, `FDE@5s` | = `L2_avg` / `L2_at` at 3 s and 5 s |
| `MR@3s`, `MR@5s` | share of samples with FDE > 2 m |
| `heading_at@t`, `heading_avg@t` | yaw error in degrees, if submission and ground truth carry yaw |

The two "L2 at t" conventions differ a lot, so always quote the key. A 3 s submission has no 5 s keys.

## Ego trajectory API

- `frame.future(horizon_s, step_s=0.1, frame="base")`: an `EgoTrajectory` with K = `round(horizon_s / step_s)` slots. Slot k (0-based) is at `t_ns + (k + 1) * step`; the frame itself is not a slot.
- The benchmark sampling directly: `frame.future(3.0, step_s=0.5)` gives the 6 rows of a 3 s submission, in its layout.
- `frame.past(horizon_s)`: the same backwards, the oldest slot first (slot k at `t_ns - (K - k) * step`).
- Fields: `t_ns`, `xyz_ego` (the planning target, relative to the frame's base frame), `quat_ego`, `xyz_world`, `valid`, `index`, `anchor_t_ns`.
- Slots the poses do not cover (past the end of the scene, in a gap) are NaN and not `valid`. Both raise `PoseUnavailable` exactly when `frame.pose` does.
- `frame="road"`: the trajectory of the road frame's origin ([Conventions](conventions.md#the-road-frame)).
- `scene.ego(t_ns)` / `frame.ego()`: pose, velocity, acceleration, yaw rate and curvature at any time (interpolated, [Conventions](conventions.md#time)). They read up to 0.2 s after that time; `frame.ego(causal=True)` uses the past only.
- In samples: `SampleSpec(future_s=, past_s=, trajectory_step_s=)`, see [Training](training.md#sample-layout). Invalid slots there hold xyz 0 and the identity quaternion, not NaN: mask with `valid`.
- `kitscenes.ego_trajectory.relative_trajectory(trajectory, anchor_t_ns, t_ns)` does the same for any times.

## Driving command and ego status

Experimental: the benchmark's navigation input is not final; the test-e2e scenes will ship their commands.

`kitscenes.e2e` has the two extra inputs that UniAD- and VAD-style planners take.

```python
from kitscenes import e2e
from kitscenes.navigation import Command

frame = kitscenes.open(splits="val")[0][50]
future = frame.future(3.0, step_s=0.5)                  # 6 slots, 0.5 s .. 3.0 s
command = e2e.driving_command(future.xyz_ego, future.valid, step_s=0.5)   # Command.LEFT 0, STRAIGHT 1, RIGHT 2, UNKNOWN 3
command.onehot("vad")                                   # (right, left, straight): the order of UniAD and VAD
ego = frame.ego(causal=True)                            # EgoState from the past only: velocity, acceleration, yaw_rate, speed
```

**Ego status** (`frame.ego(causal=True)`, `e2e.past_motion`, `sample["ego"]`):

- A least-squares quadratic over the last 0.5 s of poses and the frame's own, evaluated at the frame time. It uses the past only.
- Velocity and acceleration (3,) in m/s and m/s², yaw rate in rad/s (positive turning left), speed in m/s, in the axes of the trajectory's frame. `e2e.past_motion(frame.past(e2e.EGO_WINDOW_S))` returns them as a dict, or None with fewer than two valid past slots; `frame.ego(causal=True)` as an `EgoState` with the frame's pose (NaN motion then; roll and pitch rate are always NaN).
- In samples: `sample["ego"]` with `valid` False with less than 0.2 s of past poses (the scene start; the values are then 0). In road-frame samples it is the motion of the road frame's origin.
- The quadratic lags behind sharp changes. In a tight turn it misreads the arc: 0.6 m/s² along x at 10 m/s on a 20 m radius.
- `frame.ego()` / `scene.ego(t_ns)` use central differences and so read 0.1 s (0.2 s for the acceleration) after the frame. Don't feed them to a planner: use `frame.ego(causal=True)`.

**Driving command** (`e2e.driving_command`, `sample["future"]["command"]`):

- The UniAD / VAD rule: the lateral offset y (left positive) of the last valid slot up to 3 s. `y >= 2 m` is LEFT, `y <= -2 m` RIGHT, else STRAIGHT; UNKNOWN without a valid slot. `horizon_s=` and `threshold_m=` change the 3 s and 2 m.
- The result is a `kitscenes.navigation.Command` (LEFT 0, STRAIGHT 1, RIGHT 2, UNKNOWN 3), the same integers as `sample["navigation"]["command"]`. `Command.onehot("vad")` gives the one-hot order of UniAD and VAD (right, left, straight).
- In samples (with `future_s > 0`): `future["command"]` () int64 and `future["command_valid"]` (), valid where the slot at 3 s (or at `future_s` if shorter) is.
- **It leaks the answer**: the command is derived from the ground-truth future, as in UniAD and VAD on nuScenes. A planner conditioned on it is told part of where the vehicle went.
- **The test-e2e scenes will ship their commands** (experimental, not final). For training, compute them with `kitscenes.navigation` ([SD maps and navigation](sdmap.md#command-rules)) or `e2e.driving_command`.
