"""The viewer server's page: the KITScenes look only with --standalone."""

import unittest

from kitscenes_viewer.server import index_page


class IndexPageTest(unittest.TestCase):
    def test_without_the_flag_the_page_is_served_as_it_is(self):
        page = index_page(False).decode("utf-8")
        self.assertIn("<title>Lanelet2 3D Viewer</title>", page)
        self.assertIn("<body>", page)
        self.assertNotIn("standalone", page)

    def test_standalone_is_the_kitscenes_viewer_from_the_first_paint(self):
        page = index_page(True).decode("utf-8")
        self.assertIn("<title>KITScenes viewer</title>", page)
        self.assertIn('<body class="standalone">', page)
        self.assertIn('class="brandName">KITScenes viewer</span>', page)
        self.assertNotIn("Lanelet2 3D", page)


if __name__ == "__main__":
    unittest.main()
