"""Progress bars without tqdm's monitor thread.

The crop pool forks its workers; tqdm's monitor (started by any bar, even a
disabled one, and kept until the last bar closes) would make that a fork
of a multi-threaded process.
"""

from __future__ import annotations

import tqdm


class ProgressBar(tqdm.tqdm):
    monitor_interval = 0
