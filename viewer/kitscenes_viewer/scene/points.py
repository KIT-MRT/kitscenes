"""Binary point-cloud encoding for the browser.

JSON is out of the question here: a frame is ~740 k points, which is ~9 MB as
raw floats and several times that as text. The browser gets an ArrayBuffer it
can hand to a `THREE.BufferGeometry` with no parsing beyond a few typed-array
views.

    offset 0   8B   magic "LL2PC1\\0\\0"
           8   4B   uint32 count
          12   4B   uint32 flags   bit0 intensity, bit1 sensor, bit2 class,
                                 bit3 camera rgb, bit4 instance
          16  24B   float64 ox, oy, oz   cloud origin in viewer ENU
          40   ..   float32 xyz[3 * count], relative to that origin
             count  uint8 intensity (optional)
             count  uint8 sensor id (optional)
             count  uint8 Vistas class (optional; 255 = unlabelled)
           3*count  uint8 r, g, b from the cameras (optional)
           2*count  uint16 instance number (optional; 0 = no instance)

Positions are float32 *relative to an origin* because ENU metres in these maps
reach several kilometres, where float32 steps to ~0.5 mm; relative to the
cloud's own origin the step is ~1 um and nothing is lost.
"""

import struct

import numpy as np

MAGIC = b"LL2PC1\0\0"
HEADER_SIZE = 40
FLAG_INTENSITY = 1
FLAG_SENSOR = 2
FLAG_CLASS = 4
FLAG_RGB = 8
# Instance numbers are per frame, not tracked between frames, so only a single
# frame's cloud ever carries them -- never the aggregate.
FLAG_INSTANCE = 16
# reflectivity is nominally 0..1 with 2.0 for retro-reflectors (sign posts);
# the whole range is kept so those still stand out.
INTENSITY_FULL_SCALE = 2.0


def encode_points(xyz, intensity=None, sensor=None, origin=None, classes=None, rgb=None,
                  instances=None):
    """(N,3) ENU points -> bytes in the format above."""
    xyz = np.asarray(xyz, dtype=np.float64)
    count = int(xyz.shape[0])
    if origin is None:
        origin = xyz.mean(axis=0) if count else np.zeros(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    flags = 0
    if intensity is not None:
        flags |= FLAG_INTENSITY
    if sensor is not None:
        flags |= FLAG_SENSOR
    if classes is not None:
        flags |= FLAG_CLASS
    if rgb is not None:
        flags |= FLAG_RGB
    if instances is not None:
        flags |= FLAG_INSTANCE

    parts = [struct.pack("<8sII3d", MAGIC, count, flags, *origin),
             (xyz - origin).astype(np.float32).tobytes()]
    if intensity is not None:
        scaled = np.clip(np.asarray(intensity, dtype=np.float32) / INTENSITY_FULL_SCALE, 0.0, 1.0)
        parts.append((scaled * 255.0 + 0.5).astype(np.uint8).tobytes())
    if sensor is not None:
        parts.append(np.asarray(sensor, dtype=np.uint8).tobytes())
    if classes is not None:
        parts.append(np.asarray(classes, dtype=np.uint8).tobytes())
    if rgb is not None:
        parts.append(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3).tobytes())
    if instances is not None:
        parts.append(np.asarray(instances, dtype="<u2").tobytes())
    return b"".join(parts)


def decode_points(blob):
    """Inverse: (xyz, intensity, sensor, origin, classes, rgb, instances)."""
    magic, count, flags = struct.unpack_from("<8sII", blob, 0)
    if magic != MAGIC:
        raise ValueError("not a point cloud blob: %r" % magic)
    origin = np.array(struct.unpack_from("<3d", blob, 16))
    end = HEADER_SIZE + count * 12
    xyz = np.frombuffer(blob, dtype=np.float32, count=count * 3,
                        offset=HEADER_SIZE).reshape(count, 3).astype(np.float64) + origin
    intensity = sensor = classes = rgb = instances = None
    if flags & FLAG_INTENSITY:
        intensity = np.frombuffer(blob, dtype=np.uint8, count=count, offset=end)
        end += count
    if flags & FLAG_SENSOR:
        sensor = np.frombuffer(blob, dtype=np.uint8, count=count, offset=end)
        end += count
    if flags & FLAG_CLASS:
        classes = np.frombuffer(blob, dtype=np.uint8, count=count, offset=end)
        end += count
    if flags & FLAG_RGB:
        rgb = np.frombuffer(blob, dtype=np.uint8, count=count * 3,
                            offset=end).reshape(count, 3)
        end += count * 3
    if flags & FLAG_INSTANCE:
        instances = np.frombuffer(blob, dtype="<u2", count=count, offset=end)
    return xyz, intensity, sensor, origin, classes, rgb, instances
