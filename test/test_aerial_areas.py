"""kitscenes.aerial._areas: the 10 km area of a map origin."""

from __future__ import annotations

from kitscenes.aerial._areas import bucket_area_id
from kitscenes.aerial._geo import latlon_to_epsg25832

FRANKFURT = (50.110423, 8.682138)
FRANKFURT_NEARBY = (50.110500, 8.682300)
KARLSRUHE = (49.006889, 8.403653)


def test_bucket_area_id() -> None:
    area = bucket_area_id(*FRANKFURT)
    e, n = latlon_to_epsg25832(*FRANKFURT)
    assert area == f"area_{int(e // 10_000) * 10_000}_{int(n // 10_000) * 10_000}"
    assert bucket_area_id(*FRANKFURT_NEARBY) == area and bucket_area_id(*KARLSRUHE) != area
    assert bucket_area_id(*FRANKFURT, bucket_size_m=1000.0) == f"area_{int(e // 1000) * 1000}_{int(n // 1000) * 1000}"
