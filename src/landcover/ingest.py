"""Data ingestion: download the public rasters, align them on the Landsat grid, remap the classes.

Output (data/processed/, gitignored):
- stack.npy   float32 (C, H, W): reflectance (6 Landsat bands) + radar gamma0 in dB (HH, HV), NaN = missing
- labels.npy  uint8 (H, W): class index 1..K, 0 = nodata or ignored class
- meta.json   grid (transform, CRS, bounds, shape), channels and the kept classes

    uv run python -m landcover.ingest [--force]
"""

import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from landcover import config as cfg


def download(force: bool = False) -> dict[str, Path]:
    """Fetch the three rasters once (about 215 MB) and check their size."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, size) in cfg.SOURCES.items():
        path = cfg.RAW_DIR / name
        if force or not path.exists() or path.stat().st_size != size:
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(f"{cfg.BASE_URL}/{name}", timeout=120) as r, tmp.open("wb") as f:  # noqa: S310
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise RuntimeError(f"{name}: taille inattendue ({tmp.stat().st_size} au lieu de {size} octets)")
            tmp.replace(path)
        paths[key] = path
    return paths


def read_landsat(path: Path) -> tuple[np.ndarray, dict]:
    with rio.open(path) as src:
        idx = [src.descriptions.index(b) + 1 for b in cfg.LANDSAT_BANDS]
        raw = src.read(idx).astype(np.float32)
        grid = {"transform": src.transform, "crs": src.crs, "width": src.width, "height": src.height,
                "bounds": tuple(src.bounds)}
    raw[raw == cfg.NODATA] = np.nan
    return raw / cfg.REFLECTANCE_SCALE, grid


def read_radar_on_grid(path: Path, grid: dict) -> np.ndarray:
    """PALSAR-2 gamma0 (stored as dB x 1000, 25 m) averaged onto the Landsat grid (30 m).

    The average is taken in linear power, then converted back to dB.
    """
    out = np.full((len(cfg.RADAR_BANDS), grid["height"], grid["width"]), np.nan, dtype=np.float32)
    with rio.open(path) as src:
        for i, band in enumerate(cfg.RADAR_BANDS):
            raw = src.read(src.descriptions.index(band) + 1).astype(np.float32)
            power = np.where(raw == cfg.NODATA, np.nan, 10 ** (raw / cfg.RADAR_DB_SCALE / 10)).astype(np.float32)
            reproject(power, out[i], src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=grid["transform"], dst_crs=grid["crs"], dst_nodata=np.nan,
                      resampling=Resampling.average)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 * np.log10(out)


def read_labels(path: Path, grid: dict) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Remap KLHK codes to 1..K (classes with enough pixels), everything else to IGNORE."""
    with rio.open(path) as src:
        codes = src.read(1)
        if (src.width, src.height) != (grid["width"], grid["height"]) or src.transform != grid["transform"]:
            raise RuntimeError("La carte d'occupation du sol n'est pas sur la grille Landsat")
    counts = dict(zip(*np.unique(codes, return_counts=True), strict=True))
    labels = np.full(codes.shape, cfg.IGNORE, dtype=np.uint8)
    kept, ignored = [], []
    for c in cfg.all_classes():
        n = int(counts.get(c.code, 0))
        entry = {"code": c.code, "label": c.label, "label_fr": c.label_fr, "color": c.color, "pixels": n}
        if n >= cfg.MIN_CLASS_PIXELS:
            entry["index"] = len(kept) + 1
            labels[codes == c.code] = entry["index"]
            kept.append(entry)
        elif n > 0:
            ignored.append(entry)
    return labels, kept, ignored


@dataclass
class Scene:
    x: np.ndarray  # (C, H, W) float32, NaN = missing
    y: np.ndarray  # (H, W) uint8, 0 = ignore
    meta: dict

    @property
    def classes(self) -> list[dict]:
        return self.meta["classes"]

    @property
    def valid(self) -> np.ndarray:
        """Pixels where every input channel is available."""
        return np.isfinite(self.x).all(axis=0)


def build(force_download: bool = False) -> Scene:
    paths = download(force_download)
    landsat, grid = read_landsat(paths["landsat"])
    radar = read_radar_on_grid(paths["palsar"], grid)
    labels, kept, ignored = read_labels(paths["landcover"], grid)
    x = np.concatenate([landsat, radar]).astype(np.float32)
    labels[~np.isfinite(x).all(axis=0)] = cfg.IGNORE
    meta = {
        "crs": grid["crs"].to_string(), "transform": list(grid["transform"])[:6], "bounds": grid["bounds"],
        "width": grid["width"], "height": grid["height"], "channels": cfg.CHANNELS,
        "classes": kept, "ignored_classes": ignored,
        "valid_share": float(np.isfinite(x).all(axis=0).mean()),
        "labelled_share": float((labels != cfg.IGNORE).mean()),
    }
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cfg.PROCESSED_DIR / "stack.npy", x)
    np.save(cfg.PROCESSED_DIR / "labels.npy", labels)
    (cfg.PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return Scene(x, labels, meta)


def load() -> Scene:
    """Load the processed scene, building it first if needed."""
    if not (cfg.PROCESSED_DIR / "meta.json").exists():
        return build()
    return Scene(np.load(cfg.PROCESSED_DIR / "stack.npy", mmap_mode="r"),
                 np.load(cfg.PROCESSED_DIR / "labels.npy"),
                 json.loads((cfg.PROCESSED_DIR / "meta.json").read_text()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download the rasters")
    s = build(ap.parse_args().force)
    print(f"Scène {s.meta['width']} x {s.meta['height']}, {len(s.classes)} classes, "
          f"{s.meta['labelled_share']:.1%} de pixels étiquetés, ignorées : "
          f"{[c['label_fr'] for c in s.meta['ignored_classes']]}")
    for i, ch in enumerate(cfg.CHANNELS):
        v = s.x[i][np.isfinite(s.x[i])]
        print(f"  {ch}: p2={np.percentile(v, 2):.3f} p98={np.percentile(v, 98):.3f} manquants={1 - v.size / s.x[i].size:.2%}")
