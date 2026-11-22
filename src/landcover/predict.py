"""Backend analysis for the map app: full-scene prediction, map layers and area statistics.

    uv run python -W ignore -m landcover.predict

Writes
- reports/prediction.tif         GeoTIFF (uint8 classes with colour table) for QGIS & co
- app/static/layers/*.png|.jpg   overlays served by Streamlit (natural colour, prediction,
                                 reference, disagreement, confidence)
- reports/areas.csv              hectares per class, predicted vs reference map
- reports/summary.json           headline numbers shown in the app
"""

import json

import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image
from rasterio.transform import Affine

from landcover import config as cfg
from landcover.geo import row_area_ha
from landcover.ingest import load
from landcover.tiling import predict_window, torch_runner
from landcover.train import device, load_model

APP_LAYERS = cfg.ROOT / "app" / "static" / "layers"
DISAGREE = (214, 40, 160)  # magenta: never used by the land cover palette


def rgb_image(x: np.ndarray, valid: np.ndarray) -> Image.Image:
    lo, hi = cfg.RGB_STRETCH
    idx = [cfg.CHANNELS.index(b) for b in cfg.RGB_BANDS]
    rgb = np.stack([np.asarray(x[i]) for i in idx], axis=-1)
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** (1 / 1.4)  # gentle gamma for the dark forest
    rgb = np.nan_to_num(rgb * 255).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb, "RGB")


def class_image(classes: np.ndarray, colors: list[str], alpha: int = 255) -> Image.Image:
    """Palette PNG: index 0 transparent, 1..K the class colours (small files)."""
    img = Image.fromarray(classes.astype(np.uint8), "P")
    pal = [0, 0, 0] + [v for c in colors for v in bytes.fromhex(c[1:])]
    img.putpalette(pal + [0] * (768 - len(pal)))
    img.info["transparency"] = bytes([0] + [alpha] * len(colors))
    return img


def mask_image(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, 230)
    return Image.fromarray(rgba, "RGBA")


def confidence_image(conf: np.ndarray, valid: np.ndarray, levels: int = 12) -> Image.Image:
    """Low confidence opaque orange, high confidence transparent: where the model hesitates.

    Quantised to a few alpha levels in a palette PNG (a few hundred kB instead of 5 MB).
    """
    q = np.where(valid, np.clip((1 - conf) * 1.6, 0, 1), 0)
    idx = np.round(q * (levels - 1)).astype(np.uint8)
    img = Image.fromarray(idx, "P")
    img.putpalette([255, 176, 32] * levels + [0] * (768 - 3 * levels))
    img.info["transparency"] = bytes(int(i / (levels - 1) * 230) for i in range(levels))
    return img


def areas(pred: np.ndarray, ref: np.ndarray, classes: list[dict], meta: dict) -> pd.DataFrame:
    ha = row_area_ha(meta)[:, None]
    rows = []
    for c in classes:
        k = c["index"]
        rows.append({"index": k, "code": c["code"], "label_fr": c["label_fr"], "color": c["color"],
                     "pred_ha": float((ha * (pred == k)).sum()), "ref_ha": float((ha * (ref == k)).sum())})
    df = pd.DataFrame(rows)
    df["pred_share"] = df.pred_ha / df.pred_ha.sum()
    df["ref_share"] = df.ref_ha / df.ref_ha.sum()
    return df.round({"pred_ha": 1, "ref_ha": 1}).sort_values("pred_ha", ascending=False)


def write_geotiff(classes: np.ndarray, meta: dict, colors: list[str], path) -> None:
    profile = {"driver": "GTiff", "width": meta["width"], "height": meta["height"], "count": 1, "dtype": "uint8",
               "crs": meta["crs"], "transform": Affine(*meta["transform"]), "nodata": 0, "compress": "deflate",
               "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rio.open(path, "w", **profile) as dst:
        dst.write(classes, 1)
        dst.write_colormap(1, {i + 1: tuple(bytes.fromhex(c[1:])) for i, c in enumerate(colors)})


def run() -> dict:
    scene = load()
    metrics = json.loads((cfg.MODELS_DIR / "metrics.json").read_text())
    fill = np.array(metrics["normalisation"]["mean"], dtype=np.float32)
    h, w = scene.y.shape
    pred, conf = predict_window(scene.x, fill, torch_runner(load_model(), device()), 0, h, 0, w)
    valid = scene.valid
    colors = [c["color"] for c in scene.classes]
    APP_LAYERS.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rgb_image(scene.x, valid).save(APP_LAYERS / "satellite.jpg", quality=82, optimize=True)
    class_image(pred, colors).save(APP_LAYERS / "prediction.png", optimize=True)
    class_image(scene.y, colors).save(APP_LAYERS / "reference.png", optimize=True)
    labelled = (scene.y > 0) & (pred > 0)
    mask_image(labelled & (pred != scene.y), DISAGREE).save(APP_LAYERS / "disagreement.png", optimize=True)
    confidence_image(conf, valid).save(APP_LAYERS / "confidence.png", optimize=True)
    write_geotiff(pred, scene.meta, colors, cfg.REPORTS_DIR / "prediction.tif")

    df = areas(pred, scene.y, scene.classes, scene.meta)
    df.to_csv(cfg.REPORTS_DIR / "areas.csv", index=False)
    forest = {"Forêt primaire de terre ferme", "Forêt secondaire de terre ferme", "Forêt de plantation"}
    is_forest = df.label_fr.isin(forest)
    summary = {
        "scene": {"name": "Kerinci, Sumatra (Indonésie)", "year": 2023, "bounds": scene.meta["bounds"],
                  "width": w, "height": h, "area_km2": float(row_area_ha(scene.meta).sum() * w / 100)},
        "forest_share_pred": float(df.pred_share[is_forest].sum()),
        "forest_share_ref": float(df.ref_share[is_forest].sum()),
        "agreement_all_pixels": float((pred[labelled] == scene.y[labelled]).mean()),
        "mean_confidence": float(conf[pred > 0].mean()),
        "test": metrics["summary"],
        "model": {"parameters": metrics["model"]["parameters"], "trained_at": metrics["metadata"]["trained_at"],
                  "fast": metrics["metadata"]["fast"], "epochs": metrics["metadata"]["epochs_run"]},
        "layers": sorted(p.name for p in APP_LAYERS.iterdir()),
    }
    (cfg.REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    s = run()
    print(f"Forêt : {s['forest_share_pred']:.1%} prédite, {s['forest_share_ref']:.1%} dans la référence ; "
          f"accord global {s['agreement_all_pixels']:.1%}, couches : {', '.join(s['layers'])}")
