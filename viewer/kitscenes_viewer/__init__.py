"""The KITScenes 3D viewer: a scene or a city of scenes in the browser.

Three parts, each usable alone:

* ``server.py`` serves the page (``static/``) and fans scene messages out to
  every open tab. Standard library only, so the JOSM Lanelet2 plugin runs it
  on the system Python and streams its map into it.
* ``kitscenes_viewer.scene`` reads the dataset through ``kitscenes`` and feeds
  the viewer: map, trajectories, lidar frames, camera images and whole-scene
  clouds.
* ``kitscenes-viewer <scene | dataset root>`` starts both.
"""

__version__ = "0.2.0"
