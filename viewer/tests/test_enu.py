"""The coordinate chain. If these fail, the map and the point clouds drift."""

import math
import unittest

from kitscenes_viewer.scene.enu import EARTH_R, SceneFrame, enu_to_lat_lon, round_coord, utm_epsg

FRANKFURT = (50.110423, 8.682138)  # the example scene's maps/origin.json


def enu(lat, lon, lat0, lon0):
    """Viewer ENU metres of a lat/lon about the anchor (lat0, lon0): the page's own formula."""
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_R
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


class UtmZoneTest(unittest.TestCase):
    def test_zone_matches_lanelet2(self):
        self.assertEqual("EPSG:32632", utm_epsg(*FRANKFURT))
        self.assertEqual("EPSG:32632", utm_epsg(49.01439, 8.41722))    # Karlsruhe
        self.assertEqual("EPSG:32633", utm_epsg(52.52, 13.405))        # Berlin
        self.assertEqual("EPSG:32719", utm_epsg(-33.45, -70.66))       # southern hemisphere


class EquirectTest(unittest.TestCase):
    def test_round_trip(self):
        lat, lon = 50.1009, 8.6745
        x, y = enu(lat, lon, *FRANKFURT)
        back = enu_to_lat_lon(x, y, *FRANKFURT)
        self.assertAlmostEqual(lat, back[0], places=9)
        self.assertAlmostEqual(lon, back[1], places=9)

    def test_round_coord_is_half_away_from_zero(self):
        self.assertEqual(0.002, round_coord(0.0015))
        self.assertEqual(-0.002, round_coord(-0.0015))
        self.assertEqual(1.235, round_coord(1.2345))


class SceneFrameTest(unittest.TestCase):
    """The frame must reproduce lanelet2's own inverse, not merely come close."""

    @classmethod
    def setUpClass(cls):
        try:
            import lanelet2  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("lanelet2 not installed: %s" % exc)

    def _lanelet2_reverse(self, x, y):
        from lanelet2.core import BasicPoint3d
        from lanelet2.io import Origin
        from lanelet2.projection import UtmProjector
        projector = UtmProjector(Origin(*FRANKFURT))
        gps = projector.reverse(BasicPoint3d(float(x), float(y), 0.0))
        return gps.lat, gps.lon

    def test_matches_the_lanelet2_projector(self):
        anchor = (50.1004768, 8.6745568)
        frame = SceneFrame(FRANKFURT[0], FRANKFURT[1], *anchor)
        worst = 0.0
        for dx in (-800, -200, 0, 350, 900):
            for dy in (-1200, -300, 0, 400, 1100):
                lat, lon = self._lanelet2_reverse(dx, dy)
                want = enu(lat, lon, *anchor)
                got = frame.apply(dx, dy)
                worst = max(worst, math.hypot(want[0] - got[0], want[1] - got[1]))
        self.assertLess(worst, 1e-6, "drifted %.6f mm from lanelet2's chain" % (worst * 1000))

    def test_array_and_single_point_agree(self):
        import numpy as np
        frame = SceneFrame(FRANKFURT[0], FRANKFURT[1], 50.1004768, 8.6745568)
        xy = np.array([[-500.0, -1000.0], [0.0, 0.0], [250.0, 75.5]])
        batch = frame.apply_array(xy)
        for row, point in zip(xy, batch):
            single = frame.apply(row[0], row[1])
            self.assertAlmostEqual(single[0], point[0], places=9)
            self.assertAlmostEqual(single[1], point[1], places=9)

    def test_reports_the_grid_convergence(self):
        frame = SceneFrame(FRANKFURT[0], FRANKFURT[1], 50.1004768, 8.6745568)
        # ~0.25 deg at this longitude: 4.3 m per km, which is why the viewer
        # cannot simply treat scene metres as ENU metres.
        self.assertAlmostEqual(0.25, frame.rotation_deg_array([[-480.0, -1000.0]])[0], places=2)


if __name__ == "__main__":
    unittest.main()
