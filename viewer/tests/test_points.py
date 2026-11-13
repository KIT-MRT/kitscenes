"""The wire format the browser parses in js/render/points.js."""

import struct
import unittest

import numpy as np

from kitscenes_viewer.scene.points import (FLAG_CLASS, FLAG_INSTANCE, FLAG_INTENSITY, FLAG_RGB,
                              FLAG_SENSOR, HEADER_SIZE, MAGIC, decode_points,
                              encode_points)


class EncodeTest(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[1000.5, -2000.25, 140.0],
                             [1001.5, -2000.25, 141.5],
                             [999.0, -1999.0, 139.25]])
        self.intensity = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.sensor = np.array([0, 3, 6], dtype=np.uint8)

    def test_round_trip(self):
        blob = encode_points(self.xyz, self.intensity, self.sensor)
        xyz, intensity, sensor, origin, _c, _r, _i = decode_points(blob)
        np.testing.assert_allclose(xyz, self.xyz, atol=1e-4)
        np.testing.assert_array_equal(sensor, self.sensor)
        # 0, 1 and 2 of the 0..2 reflectivity range land on 0, half and full.
        self.assertEqual(0, intensity[0])
        self.assertEqual(128, intensity[1])
        self.assertEqual(255, intensity[2])
        np.testing.assert_allclose(origin, self.xyz.mean(axis=0))

    def test_header_layout(self):
        blob = encode_points(self.xyz, self.intensity, self.sensor)
        magic, count, flags = struct.unpack_from("<8sII", blob, 0)
        self.assertEqual(MAGIC, magic)
        self.assertEqual(3, count)
        self.assertEqual(FLAG_INTENSITY | FLAG_SENSOR, flags)
        self.assertEqual(HEADER_SIZE + 3 * 12 + 3 + 3, len(blob))

    def test_optional_channels(self):
        blob = encode_points(self.xyz)
        _, intensity, sensor, _, _, _, _ = decode_points(blob)
        self.assertIsNone(intensity)
        self.assertIsNone(sensor)
        self.assertEqual(HEADER_SIZE + 3 * 12, len(blob))

    def test_positions_keep_millimetres_far_from_the_origin(self):
        """float32 relative to the cloud origin, which is the point of it."""
        far = self.xyz + np.array([6_000.0, 6_000.0, 0.0])  # ENU metres get this big
        xyz, *_ = decode_points(encode_points(far))
        self.assertLess(np.abs(xyz - far).max(), 1e-4)

    def test_empty_cloud(self):
        blob = encode_points(np.zeros((0, 3)))
        xyz, *_ = decode_points(blob)
        self.assertEqual(0, len(xyz))

    def test_instances_round_trip_behind_the_other_channels(self):
        """A frame may carry all five; the aggregate never carries instances."""
        classes = np.array([58, 255, 48], np.uint8)
        rgb = np.arange(9, dtype=np.uint8).reshape(3, 3)
        instances = np.array([0, 1, 60000], np.uint16)  # a byte would not hold the last
        blob = encode_points(self.xyz, self.intensity, self.sensor,
                             classes=classes, rgb=rgb, instances=instances)
        _, _, _, _, got_classes, got_rgb, got_instances = decode_points(blob)
        np.testing.assert_array_equal(got_classes, classes)
        np.testing.assert_array_equal(got_rgb, rgb)
        np.testing.assert_array_equal(got_instances, instances)
        flags = struct.unpack_from("<I", blob, 12)[0]
        self.assertEqual(FLAG_INTENSITY | FLAG_SENSOR | FLAG_CLASS | FLAG_RGB | FLAG_INSTANCE,
                         flags)
        self.assertEqual(HEADER_SIZE + 3 * 12 + 3 + 3 + 3 + 9 + 6, len(blob))

    def test_no_instances_by_default(self):
        blob = encode_points(self.xyz, self.intensity)
        self.assertIsNone(decode_points(blob)[6])
        self.assertEqual(0, struct.unpack_from("<I", blob, 12)[0] & FLAG_INSTANCE)

    def test_explicit_origin_is_kept(self):
        blob = encode_points(self.xyz, origin=[1000.0, -2000.0, 140.0])
        xyz, _, _, origin, _, _, _ = decode_points(blob)
        np.testing.assert_allclose(origin, [1000.0, -2000.0, 140.0])
        np.testing.assert_allclose(xyz, self.xyz, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
