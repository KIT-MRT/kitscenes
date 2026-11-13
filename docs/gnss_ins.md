# GNSS/INS: native rate and frame sync

Two receivers:

- `gnss`: Septentrio mosaic, 19 topics (`navsatfix`, `pvtgeodetic`, `pose`, ...; 10 Hz, status 1 Hz);
- `gnss_ins`: AsteRx with IMU, 26 topics (`imu`, `insnavgeod`, `localization`, ...; IMU and INS 50 Hz).

45 message topics in all (`scene.message_topics(device)`). Each topic is one Parquet file with **every message at the receiver's rate**. The messages that belong to the frames are marked in it, so there is no separate "synchronized" copy:

```
<scene>/gnss/<topic>.parquet       19 topics of the Septentrio mosaic (navsatfix, pvtgeodetic, pose, ...; 10 Hz, status 1 Hz)
<scene>/gnss_ins/<topic>.parquet   26 topics of the AsteRx INS (imu, insnavgeod, localization, ...; IMU and INS 50 Hz)
```

## Columns

| Column | Meaning |
|---|---|
| `index` | message number |
| `t_ns` | the message's `header.stamp`, ns since the scene start |
| `t_sync_ns` | the time the extraction synchronized the message to, on the clock of the frame times (`timestamp.reference.txt`; equal to `t_ns` in the release) |
| **`frame_index`** | int32: the frame this message belongs to, the one whose `t_sync_ns` equals that frame's reference time; null for the messages between frames |
| message fields | dotted columns (`orientation.x`, `angular_velocity.z`), lists as list columns (covariances), NaN where the receiver sends none |

The first rows of `gnss_ins/imu.parquet` of a scene whose frames start at `0.0461568`, `0.1461568`, `0.2461568` s:

| index | t_ns | t_sync_ns | frame_index | angular_velocity.z |
|---|---|---|---|---|
| 0 | 46156800 | 46156800 | **0** | 0.0390 |
| 1 | 66156800 | 66156800 | | 0.0356 |
| 2–4 | ... | ... | | ... |
| 5 | 146156800 | 146156800 | **1** | 0.0284 |
| 6–9 | ... | ... | | ... |
| 10 | 246156800 | 246156800 | **2** | 0.0150 |

## Reading

```python
imu = scene.messages("gnss_ins", "imu")                    # every message (50 Hz), pyarrow.Table
per_frame = scene.messages("gnss_ins", "imu", rate="frame")  # one row per frame, index = frame number
# without the API: pyarrow.parquet.read_table(".../gnss_ins/imu.parquet"), then the rows with a frame_index
#                  (pandas: df[df.frame_index.notna()])
```

- `scene.messages(device, topic)` returns the table as stored (`pyarrow`). An unknown topic raises `KeyError`.
- `rate="frame"` gives only the rows with a `frame_index`, one per frame, with `index` = the frame index. Topics slower than the frames (1–2 Hz status and covariance messages) have a `frame_index` on every n-th frame only.
- `scene.imu(frame="base")`: the 50 Hz gyro/accelerometer.
- Use `t_ns` for the measurement time and to interpolate between messages. `frame_index` says which message the dataset assigns to a frame. In dataset 1.x the folders `gnss/<topic>/<frame>.yaml` held these, and `async/` all messages.

## Notes

- GPS week numbers and times of week are zeroed.
- The test-e2e splits ship no GNSS (`WITHHELD_BY_SPLIT`).
- A few rows have a null `index` (15) or `t_sync_ns` (11) where the extraction lost one of them.
