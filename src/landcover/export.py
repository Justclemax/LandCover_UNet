"""Export the trained U-Net for the Rust server (models/rust/).

- model.onnx   fixed input (1, C, TILE, TILE) raw values -> (1, K, TILE, TILE) logits
- bundle.json  format "landcover-bundle/1": channels, fill values, classes, tiling, scene grid, metrics
- scene.bin    the scene as little-endian float32, pixel-interleaved (H, W, C), NaN = missing
               (about 240 MB, gitignored; rebuilt from the public rasters by `ingest` + `export`)
- parity.json  windows predicted by onnxruntime with landcover.tiling, which Rust must reproduce

    uv run python -W ignore -m landcover.export [--no-scene | --scene-only]
"""

import argparse
import hashlib
import json

import numpy as np

from landcover import config as cfg
from landcover.ingest import load

FORMAT = "landcover-bundle/1"
# (r0, r1, c0, c1): scene corner (edge padding), an interior window, a tiny window, a wide strip.
PARITY_WINDOWS = [(0, 150, 0, 170), (1200, 1420, 900, 1180), (2000, 2007, 1500, 1505), (2600, 2751, 2400, 2692)]


def export_onnx(model, path) -> None:
    import torch

    dummy = torch.zeros(1, len(cfg.CHANNELS), cfg.TILE, cfg.TILE)
    torch.onnx.export(model, (dummy,), str(path), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)


def write_scene(x: np.ndarray, path) -> str:
    """Pixel-interleaved float32 so a window row is one contiguous read. Returns its SHA-256."""
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for r in range(x.shape[1]):
            row = np.ascontiguousarray(np.asarray(x[:, r, :], dtype="<f4").T)  # (W, C)
            b = row.tobytes()
            h.update(b)
            f.write(b)
    return h.hexdigest()


def scene_only(check: bool = True) -> str:
    """Rebuild scene.bin from the public rasters (no torch needed), optionally checking it is
    byte-identical to the scene the committed model was exported with."""
    scene = load()
    sha = write_scene(scene.x, cfg.RUST_DIR / "scene.bin")
    if check:
        want = json.loads((cfg.RUST_DIR / "bundle.json").read_text())["scene"]["sha256"]
        if sha != want:
            raise SystemExit(f"scene.bin différent de celui du bundle ({sha[:12]} au lieu de {want[:12]})")
    return sha


def run(write_scene_file: bool = True) -> dict:
    from landcover.tiling import onnx_runner, predict_window, torch_runner
    from landcover.train import load_model

    scene = load()
    metrics = json.loads((cfg.MODELS_DIR / "metrics.json").read_text())
    fill = np.array(metrics["normalisation"]["mean"], dtype=np.float32)
    model = load_model()
    cfg.RUST_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = cfg.RUST_DIR / "model.onnx"
    export_onnx(model, onnx_path)

    # ONNX must agree with PyTorch before anything is handed to Rust.
    run_onnx, run_torch = onnx_runner(onnx_path), torch_runner(model, "cpu")
    parity = []
    for r0, r1, c0, c1 in PARITY_WINDOWS:
        cls_o, conf_o = predict_window(scene.x, fill, run_onnx, r0, r1, c0, c1)
        cls_t, _ = predict_window(scene.x, fill, run_torch, r0, r1, c0, c1)
        agree = float((cls_o == cls_t).mean())
        if agree < 0.999:
            raise RuntimeError(f"ONNX et PyTorch divergent sur {r0, r1, c0, c1} : {agree:.4f}")
        parity.append({"window": [r0, r1, c0, c1], "classes": cls_o.ravel().tolist(),
                       "confidence": np.round(conf_o.ravel(), 6).tolist(), "torch_agreement": agree})

    scene_sha = None
    if write_scene_file:
        scene_sha = write_scene(scene.x, cfg.RUST_DIR / "scene.bin")
    m = scene.meta
    bundle = {
        "format": FORMAT,
        "model": {"file": "model.onnx", "architecture": "U-Net", "parameters": metrics["model"]["parameters"],
                  "trained_at": metrics["metadata"]["trained_at"], "fast": metrics["metadata"]["fast"],
                  "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()},
        "channels": cfg.CHANNELS,
        "fill": fill.tolist(),
        "tile": cfg.TILE, "tile_margin": cfg.TILE_MARGIN,
        "classes": [{k: c[k] for k in ("index", "code", "label", "label_fr", "color")} for c in scene.classes],
        "scene": {"file": "scene.bin", "layout": "HWC", "dtype": "f32le", "width": m["width"], "height": m["height"],
                  "crs": m["crs"], "transform": m["transform"], "bounds": m["bounds"], "sha256": scene_sha,
                  "name": "Kerinci, Sumatra (Indonésie)", "year": 2023},
        "metrics": metrics["summary"],
    }
    (cfg.RUST_DIR / "bundle.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    (cfg.RUST_DIR / "parity.json").write_text(json.dumps(parity))
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-scene", action="store_true", help="skip writing scene.bin")
    ap.add_argument("--scene-only", action="store_true", help="only rebuild scene.bin and check its SHA-256")
    a = ap.parse_args()
    if a.scene_only:
        print(f"scene.bin reconstruit et vérifié ({scene_only()[:12]})")
        raise SystemExit(0)
    b = run(not a.no_scene)
    print(f"Bundle écrit dans {cfg.RUST_DIR} ({len(b['classes'])} classes, modèle {b['model']['sha256'][:12]})")
