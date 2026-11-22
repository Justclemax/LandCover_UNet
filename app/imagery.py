"""On-demand imagery for any place on Earth, from Microsoft Planetary Computer (free, no account).

Builds the 8 model channels on an EPSG:4326 grid with the training resolution (~30 m):
- Landsat 8/9 Collection 2 Level-2, chosen year: per-pixel median of the cloud-free
  observations (QA_PIXEL fill, dilated cloud, cirrus, cloud and shadow bits masked),
  surface reflectance = DN x 2.75e-5 - 0.2 (same units as the training scene);
- ALOS PALSAR-2 yearly mosaic (2015-2021 on Planetary Computer, the latest year is used):
  gamma0 dB = 10 log10(DN^2) - 83, averaged in linear power onto the grid.

The result goes to the Rust API (`/v1/predict/raster`): no model runs here.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from rasterio.windows import from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_ASSETS = ["blue", "green", "red", "nir08", "swir16", "swir22"]  # = B2..B7
CLOUD_BITS = 0b11111  # fill, dilated cloud, cirrus, cloud, cloud shadow
PALSAR_YEAR = 2021
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_HTTP_MULTIPLEX": "YES", "VSI_CACHE": "TRUE", "GDAL_HTTP_MAX_RETRY": "3"}


@dataclass
class Composite:
    x: np.ndarray  # (8, H, W) float32, NaN = missing
    bbox: list[float]  # snapped [west, south, east, north]
    scenes: int  # Landsat scenes used
    clear_share: float  # pixels with at least one cloud-free observation
    observations: float  # mean number of cloud-free observations per pixel
    palsar_year: int | None


def grid(bbox: list[float], res: float) -> tuple[list[float], int, int]:
    """Snap the box to the pixel grid of resolution `res` (degrees)."""
    w, s, e, n = bbox
    width, height = max(1, math.ceil((e - w) / res)), max(1, math.ceil((n - s) / res))
    return [w, n - height * res, w + width * res, n], width, height


def _catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC, modifier=pc.sign_inplace)


def _read_landsat(item, bbox, width, height, transform) -> np.ndarray | None:
    """(6, H, W) reflectance with clouds masked, or None if nothing usable."""
    out = np.full((len(LANDSAT_ASSETS), height, width), np.nan, dtype=np.float32)

    def warp(href, resampling):
        with rio.open(href) as src, WarpedVRT(src, crs="EPSG:4326", transform=transform, width=width,
                                              height=height, resampling=resampling, nodata=0) as vrt:
            return vrt.read(1)

    with rio.Env(**GDAL_ENV):
        qa = warp(item.assets["qa_pixel"].href, Resampling.nearest)
        clear = (qa != 0) & ((qa & CLOUD_BITS) == 0)
        if clear.mean() < 0.02:
            return None
        for i, name in enumerate(LANDSAT_ASSETS):
            dn = warp(item.assets[name].href, Resampling.bilinear).astype(np.float32)
            out[i] = np.where(clear & (dn > 0), dn * 2.75e-5 - 0.2, np.nan)
    return out


def _read_palsar(cat, bbox, width, height, transform) -> tuple[np.ndarray, int | None]:
    out = np.full((2, height, width), np.nan, dtype=np.float32)
    items = list(cat.search(collections=["alos-palsar-mosaic"], bbox=bbox, datetime=str(PALSAR_YEAR)).items())
    if not items:
        return out, None
    with rio.Env(**GDAL_ENV):
        for i, band in enumerate(["HH", "HV"]):
            power = np.zeros((height, width), dtype=np.float64)
            weight = np.zeros((height, width), dtype=np.float64)
            for item in items:  # a box can straddle several 1 x 1 degree tiles
                with rio.open(item.assets[band].href) as src:
                    win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
                    try:  # the search also returns tiles that only touch the box along an edge
                        win = win.intersection(rio.windows.Window(0, 0, src.width, src.height))
                    except rio.errors.WindowError:
                        continue
                    dn = src.read(1, window=win).astype(np.float32)
                    p = np.where(dn > 1, (dn.astype(np.float64) ** 2) * 10 ** (-8.3), np.nan).astype(np.float32)
                    dst = np.full((height, width), np.nan, dtype=np.float32)
                    reproject(p, dst, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=np.nan,
                              dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                              resampling=Resampling.average)
                ok = np.isfinite(dst)
                power[ok] += dst[ok]
                weight[ok] += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                out[i] = np.where(weight > 0, 10 * np.log10(power / np.maximum(weight, 1)), np.nan)
    return out, PALSAR_YEAR


def build(bbox: list[float], res: float, year: int = 2023, max_scenes: int = 16,
          max_cloud: float = 70.0) -> Composite:
    """Cloud-free composite of `bbox` for `year`, on the model grid."""
    box, width, height = grid(bbox, res)
    transform = from_origin(box[0], box[3], res, res)
    cat = _catalog()
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=box, datetime=f"{year}-01-01/{year}-12-31",
                            query={"eo:cloud_cover": {"lt": max_cloud},
                                   "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:max_scenes]
    with ThreadPoolExecutor(max_workers=8) as pool:
        stacks = [s for s in pool.map(lambda it: _read_landsat(it, box, width, height, transform), items)
                  if s is not None]
        radar_future = pool.submit(_read_palsar, cat, box, width, height, transform)
        radar, palsar_year = radar_future.result()
    if stacks:
        cube = np.stack(stacks)  # (scenes, 6, H, W)
        n_obs = np.isfinite(cube[:, 0]).sum(axis=0)
        with np.errstate(all="ignore"):
            optical = np.nanmedian(cube, axis=0).astype(np.float32)
    else:
        optical = np.full((len(LANDSAT_ASSETS), height, width), np.nan, dtype=np.float32)
        n_obs = np.zeros((height, width))
    x = np.concatenate([optical, radar]).astype(np.float32)
    return Composite(x=x, bbox=box, scenes=len(stacks), clear_share=float((n_obs > 0).mean()),
                     observations=float(n_obs.mean()), palsar_year=palsar_year)


def rgb_png(x: np.ndarray, stretch: tuple[float, float] = (0.0, 0.12)) -> bytes:
    """Natural colour preview (B4, B3, B2) as PNG bytes, transparent where missing."""
    import io

    from PIL import Image

    rgb = np.stack([x[2], x[1], x[0]], axis=-1)
    alpha = np.isfinite(rgb).all(axis=-1)
    rgb = np.nan_to_num(np.clip((rgb - stretch[0]) / (stretch[1] - stretch[0]), 0, 1) ** (1 / 1.4) * 255)
    rgba = np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
