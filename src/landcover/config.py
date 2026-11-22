"""Single source of truth: paths, data sources, channels, classes, split and training constants."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("LANDCOVER_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RUST_DIR = MODELS_DIR / "rust"
REPORTS_DIR = ROOT / "reports"
LAYERS_DIR = REPORTS_DIR / "layers"
CLASSES_FILE = DATA_DIR / "lc_classes.json"

SEED = 42

# Public Cloud Storage files used by the reference project (github.com/ramiqcom/deep-learning-lc-unet, MIT).
# Kerinci, Sumatra: Landsat 8/9 2023 composite, ALOS PALSAR-2 2023 mosaic, KLHK land cover map.
BASE_URL = "https://storage.googleapis.com/gee-ramiqcom-bucket/youtube/lc_python"
SOURCES = {
    "landsat": ("Landsat_Kerinci_2023_v1.tif", 143_392_947),
    "palsar": ("Palsar_Kerinci_2023_v1.tif", 70_783_263),
    "landcover": ("LC_Image_v1.tif", 867_019),
}
NODATA = -9999

# Model inputs: six Landsat surface-reflectance bands + two radar backscatter bands (dB).
LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]  # blue, green, red, NIR, SWIR1, SWIR2
RADAR_BANDS = ["HH", "HV"]
CHANNELS = LANDSAT_BANDS + RADAR_BANDS
REFLECTANCE_SCALE = 1e4  # the GEE export stores reflectance x 10 000 (same as the reference notebook)
RADAR_DB_SCALE = 1e3  # PALSAR-2 gamma0 is stored as dB x 1000

# Classes with fewer labelled pixels than this are ignored (mining has 40 pixels in the scene).
MIN_CLASS_PIXELS = 1_000
IGNORE = 0  # label value for nodata and ignored classes; real classes are 1..K

# Spatial split: the scene is cut into BLOCK x BLOCK pixel blocks, each block goes to one split.
# Overlapping random patches (as in the reference notebook) leak test pixels into training.
BLOCK = 256
SPLIT_SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}

# Training and tiled inference.
PATCH = 256  # training crop (the reference notebook used 128)
TILE = 256  # inference tile fed to the ONNX model (Rust and Python)
TILE_MARGIN = 32  # border discarded on each side of a tile, so the stride is TILE - 2 * TILE_MARGIN
BASE_FILTERS = 24

# Map overlays: natural color composite stretch (reflectance).
RGB_BANDS = ("B4", "B3", "B2")
RGB_STRETCH = (0.0, 0.12)

FRENCH = {
    2001: "Forêt primaire de terre ferme", 2002: "Forêt secondaire de terre ferme",
    2004: "Mangrove primaire", 2005: "Forêt marécageuse primaire", 2006: "Forêt de plantation",
    2007: "Fourré sec", 2010: "Plantation agro-industrielle", 2012: "Zone bâtie", 2014: "Sol nu",
    3000: "Savane et prairie", 5001: "Eau libre", 20041: "Mangrove secondaire",
    20051: "Forêt marécageuse secondaire", 20071: "Fourré humide", 20091: "Agriculture sèche",
    20092: "Agriculture sèche mixte", 20093: "Rizière", 20094: "Aquaculture", 20121: "Port",
    20122: "Zone de transmigration", 20141: "Mine", 50011: "Marais",
}


@dataclass(frozen=True)
class LandClass:
    code: int  # KLHK code in the source raster
    label: str  # English label from the reference project
    label_fr: str
    color: str  # hex, from the reference palette


def all_classes() -> list[LandClass]:
    raw = json.loads(CLASSES_FILE.read_text())
    return [LandClass(int(v), lab, FRENCH[int(v)], "#" + col.lower())
            for v, lab, col in zip(raw["values"], raw["label"], raw["palette"], strict=True)]
