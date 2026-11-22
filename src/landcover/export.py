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


def fingerprint(x: np.ndarray) -> list[dict]:
    """Per-channel missing count, mean and std (float64). Unlike a SHA-256 it survives the
    last-bit differences of numpy's SIMD maths between CPUs (NEON on a Mac, AVX on Linux)."""
    out = []
    for c in range(x.shape[0]):
        v = np.asarray(x[c], dtype=np.float64)
        ok = np.isfinite(v)
        out.append({"missing": int((~ok).sum()), "mean": float(v[ok].mean()), "std": float(v[ok].std())})
    return out


def same_scene(a: list[dict], b: list[dict], rtol: float = 1e-5) -> bool:
    return len(a) == len(b) and all(
        x["missing"] == y["missing"] and np.isclose(x["mean"], y["mean"], rtol=rtol)
        and np.isclose(x["std"], y["std"], rtol=rtol) for x, y in zip(a, b, strict=True))


def scene_only(check: bool = True) -> list[dict]:
    """Rebuild scene.bin from the public rasters (no torch needed) and check it matches, within
    float rounding, the scene the committed model was exported with."""
    scene = load()
    write_scene(scene.x, cfg.RUST_DIR / "scene.bin")
    fp = fingerprint(scene.x)
    if check:
        want = json.loads((cfg.RUST_DIR / "bundle.json").read_text())["scene"]["fingerprint"]
        if not same_scene(fp, want):
            raise SystemExit(f"scene.bin ne correspond pas au bundle :\n{fp}\nattendu :\n{want}")
    return fp


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
                  "fingerprint": fingerprint(scene.x),
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
        scene_only()
        print("scene.bin reconstruit et vérifié (empreinte par canal)")
        raise SystemExit(0)
    b = run(not a.no_scene)
    print(f"Bundle écrit dans {cfg.RUST_DIR} ({len(b['classes'])} classes, modèle {b['model']['sha256'][:12]})")
