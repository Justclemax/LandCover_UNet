"""Geographic helpers for the EPSG:4326 scene grid (same formulas as serving/src/geo.rs)."""

import math

import numpy as np

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQUATOR = 111_320.0


def row_area_ha(meta: dict) -> np.ndarray:
    """Area of one pixel of each row, in hectares."""
    a, _, _, _, e, f = meta["transform"]
    lat = np.radians(f + (np.arange(meta["height"]) + 0.5) * e)
    return a * M_PER_DEG_LON_EQUATOR * np.cos(lat) * (-e) * M_PER_DEG_LAT / 10_000


def bbox_to_window(meta: dict, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """[west, south, east, north] -> (r0, r1, c0, c1), clipped to the scene."""
    west, south, east, north = bbox
    a, _, c, _, e, f = meta["transform"]
    eps = 1e-6
    c0 = max(math.floor((west - c) / a + eps), 0)
    c1 = min(math.ceil((east - c) / a - eps), meta["width"])
    r0 = max(math.floor((north - f) / e + eps), 0)
    r1 = min(math.ceil((south - f) / e - eps), meta["height"])
    if c0 >= c1 or r0 >= r1:
        raise ValueError("la bbox ne recoupe pas la scène")
    return r0, r1, c0, c1
