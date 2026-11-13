"""GNSS and INS messages (``gnss/``, ``gnss_ins/``) as tables.

A scene has two receivers: ``gnss`` (Septentrio mosaic) and ``gnss_ins``
(Septentrio AsteRx with IMU). Each message topic (``navsatfix``, ``imu``,
``insnavgeod``, ``pose``, ...) is one Parquet file, ``<device>/<topic>.parquet``,
with one row per message at the receiver's native rate (IMU and INS at 50 Hz,
most others at 10 Hz, status messages at 1 Hz), sorted by message number.
Columns:

- ``index`` (int64): the message number within the topic, 0..N-1;
- ``t_ns`` (int64): the message time (``header.stamp``) in nanoseconds since
  the scene start, the clock of the frames and poses;
- ``t_sync_ns`` (int64): the time the extraction synchronized the message to,
  on the clock of the frame times (``timestamp.reference.txt``), exact; in the
  release it equals the message's ``t_ns``;
- ``frame_index`` (int32, null for most rows): the frame this message is
  assigned to: the message whose ``t_sync_ns`` equals the frame's time
  (``scene.timestamps_ns[frame]``). Topics slower than the frames have it on
  every n-th frame only;
- then the message's fields: nested fields as dotted columns
  (``orientation.x``, ``header.stamp``), lists as list columns (covariances),
  NaN where the receiver sends no value, a string column where a field has
  mixed types.

``rate="native"`` (:func:`read_messages`) gives the table as stored;
``rate="frame"`` only the rows with a ``frame_index``, with ``index`` set to
the frame index (one row per frame that has a message). GPS week numbers and
times of week are zeroed in the release. The test-e2e splits ship no GNSS.

A few rows have a null ``index`` (15 messages the extraction kept only as a
frame's message) or a null ``t_sync_ns`` (11 messages without a synchronized
time). How the release tools rebuilt such topics is recorded in the file's
schema metadata (``repair``); the API does not need it.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

    from kitscenes._storage import Storage

DEVICES = ("gnss", "gnss_ins")

#: Nominal rotation from the IMU's axes to the base frame's: x_imu = -y_base,
#: y_imu = -x_base, z_imu = -z_base (z down). From IMU refinement.
#: Matches the data: on 16 release scenes with GNSS the IMU z rate is the
#: negated pose yaw rate (correlation -0.94 to -1.00) and gravity reads -9.8
#: m/s² on z. A valid gnss_ins_imu rotation in calib.json refines it
#: (:attr:`kitscenes.rig.Rig.imu_to_base`).
IMU_TO_BASE = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
RATES = ("native", "frame")
#: ``kitscenes_format`` in the schema metadata of a topic file.
FORMAT = "gnss_topic_v1"
#: The columns before the message fields.
KEY_COLUMNS = ("index", "t_ns", "t_sync_ns", "frame_index")
TOPIC_EXTENSION = ".parquet"


def topic_member(device: str, topic: str) -> str:
    """Member name of a topic file, e.g. ``gnss_ins/imu.parquet``.

    Raises:
        KeyError: For an unknown device.
    """
    if device not in DEVICES:
        raise KeyError(f"Unknown device {device!r}; expected one of {DEVICES}")
    return f"{device}/{topic}{TOPIC_EXTENSION}"


def check_rate(rate: str) -> None:
    if rate not in RATES:
        raise ValueError(f"rate must be one of {RATES}, got {rate!r}")


def list_topics(storage: "Storage", device: str) -> list[str]:
    """The topics of *device* in this copy of the scene, sorted (empty if none)."""
    topic_member(device, "")  # KeyError for an unknown device
    names = (m.rsplit("/", 1)[1] for m in storage.list(device))
    return sorted(n[: -len(TOPIC_EXTENSION)] for n in names if n.endswith(TOPIC_EXTENSION))


def read_messages(storage: "Storage", device: str, topic: str, rate: str = "native") -> "pa.Table":
    """One topic as a table (see the module docs): every message, or one per frame.

    Raises:
        KeyError: For an unknown device.
        ValueError: For an unknown rate.
        FileNotFoundError: If the scene has no such topic file.
        kitscenes.DataFormatError: If the file is not a GNSS topic file.
    """
    import pyarrow.parquet as pq

    from kitscenes.errors import DataFormatError

    member = topic_member(device, topic)
    check_rate(rate)
    data = storage.read_bytes(member)
    table = pq.ParquetFile(io.BytesIO(data)).read(use_threads=False)
    meta = table.schema.metadata or {}
    if meta.get(b"kitscenes_format") != FORMAT.encode() or table.column_names[: len(KEY_COLUMNS)] != list(KEY_COLUMNS):
        raise DataFormatError(f"{storage.describe(member)} is not a GNSS topic file ({FORMAT})")
    return frame_rows(table) if rate == "frame" else table


def frame_rows(table: "pa.Table") -> "pa.Table":
    """The rows of a topic table that are assigned to a frame, by frame, with ``index`` = the frame index."""
    import pyarrow.compute as pc

    table = table.filter(pc.is_valid(table.column("frame_index")))
    table = table.take(pc.sort_indices(table.column("frame_index")))
    return table.set_column(0, "index", pc.cast(table.column("frame_index"), "int64"))


def vectors(table: "pa.Table", prefix: str) -> np.ndarray:
    """(N, 3) array from the ``prefix.x``, ``.y``, ``.z`` columns."""
    return np.column_stack([table.column(f"{prefix}.{axis}").to_numpy(zero_copy_only=False) for axis in "xyz"]).astype(np.float64)
