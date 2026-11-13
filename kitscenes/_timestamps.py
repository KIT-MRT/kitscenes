"""Exact parsing of the release's timestamp format.

Every timestamp in a released scene (``timestamp.reference.txt``, the first
column of ``poses.txt``) is a decimal number of **seconds relative to the
scene start** (the scene's earliest measurement), with 9 decimals, e.g.
``"0.046204976"``; the frame times are exact and the poses' times equal them.
(Releases before v2.0.0 had frame times rounded to 4 decimals, ``"2.0460"``.)
The API works in integer nanoseconds. Going through ``float`` truncates:
``int(float("2.0460") * 1e9)`` is ``2045999999``, which breaks exact joins on
timestamps. This module parses the decimal string exactly instead.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np

from kitscenes.errors import DataFormatError

_NS_PER_S = Decimal(10) ** 9
_INT64_MAX = np.iinfo(np.int64).max


def seconds_to_ns(text: str) -> int:
    """Convert a decimal seconds string to integer nanoseconds, exactly.

    Digits beyond nanoseconds are rounded half to even.

    Raises:
        DataFormatError: If *text* is not a finite decimal number, or the
            result does not fit into int64 (e.g. an epoch-nanosecond integer,
            which the release format does not use).
    """
    try:
        value = Decimal(text.strip())
    except InvalidOperation:
        raise DataFormatError(f"Not a decimal number of seconds: {text!r}") from None
    if not value.is_finite():
        raise DataFormatError(f"Not a finite number of seconds: {text!r}")
    ns = int((value * _NS_PER_S).to_integral_value())
    if abs(ns) > _INT64_MAX:
        raise DataFormatError(
            f"Timestamp {text!r} is out of range; released timestamps are seconds "
            "relative to the scene start, not epoch nanoseconds."
        )
    return ns


def parse_timestamps(text: str, source: str = "<timestamps>") -> np.ndarray:
    """Parse timestamp text (one value in seconds per line) as int64 nanoseconds.

    Blank lines are skipped.

    Raises:
        DataFormatError: If a line is not a valid timestamp (see :func:`seconds_to_ns`).
    """
    values = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(seconds_to_ns(line))
        except DataFormatError as err:
            raise DataFormatError(f"{source}:{line_no}: {err}") from None
    return np.array(values, dtype=np.int64)
