"""Train the U-Net, select it on validation mIoU, evaluate it on the test blocks, compare baselines.

    uv run python -W ignore -m landcover.train            # full run (~25 min on Apple silicon)
    uv run python -W ignore -m landcover.train --fast     # smoke run (~1 min), artifacts flagged "fast"

Writes models/unet.pt, models/metrics.json, models/split.npy.
"""

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import torch
from torch import nn

from landcover import config as cfg
from landcover.ingest import Scene, load
from landcover.metrics import confusion, scores
from landcover.model import UNet, n_params
from landcover.split import SPLITS, channel_stats, make_split, pixel_split
from landcover.tiling import predict_window, torch_runner


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class PatchSampler:
    """Random PATCH x PATCH crops centred on training pixels, rare classes over-sampled.

    A class is drawn with probability proportional to frequency ** power, then a training pixel
    of that class is drawn as the crop centre. Labels outside the training blocks are
    replaced by -1 (ignored by the loss), so validation and test labels never leak.
    """

    def __init__(self, scene: Scene, split_map: np.ndarray, rng: np.random.Generator, patch: int = cfg.PATCH,
                 power: float = 0.5):
        self.x, self.rng, self.patch = scene.x, rng, patch
        half = patch // 2
        h, w = scene.y.shape
        target = scene.y.astype(np.int64) - 1
        target[split_map != 0] = -1
        self.target = target
        inner = np.zeros((h, w), dtype=bool)
        inner[half:h - half, half:w - half] = True
        self.centres, freq = [], []
        for k in range(len(scene.classes)):
            idx = np.flatnonzero((target == k) & inner)
            self.centres.append(idx)
            freq.append(len(idx))
        freq = np.array(freq, dtype=float)
        p = np.where(freq > 0, freq**power, 0)
        self.class_p = p / p.sum()
        self.width = w
        self.fill = None

    def batch(self, n: int, fill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.patch // 2
        xs, ys = [], []
        for k in self.rng.choice(len(self.centres), size=n, p=self.class_p):
            i = self.rng.choice(self.centres[k])
            r, c = divmod(int(i), self.width)
            xp = np.asarray(self.x[:, r - half:r + half, c - half:c + half], dtype=np.float32)
            xp = np.where(np.isfinite(xp), xp, fill.reshape(-1, 1, 1))
            yp = self.target[r - half:r + half, c - half:c + half]
            # Dihedral augmentation: 8 flips/rotations.
            rot, flip = int(self.rng.integers(4)), bool(self.rng.integers(2))
            xp, yp = np.rot90(xp, rot, axes=(1, 2)), np.rot90(yp, rot)
            if flip:
                xp, yp = xp[:, :, ::-1], yp[:, ::-1]
            xs.append(np.ascontiguousarray(xp))
            ys.append(np.ascontiguousarray(yp))
        return np.stack(xs), np.stack(ys)


def class_weights(scene: Scene, split_map: np.ndarray, power: float) -> torch.Tensor:
    k = len(scene.classes)
    counts = np.bincount(scene.y[(split_map == 0) & (scene.y > 0)], minlength=k + 1)[1:].astype(float)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0) ** power
    return torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)


def evaluate(pred: np.ndarray, scene: Scene, split_map: np.ndarray, split: int) -> dict:
    y = np.where(split_map == split, scene.y, 0)
    return scores(confusion(y, pred, len(scene.classes)))


def baselines(scene: Scene, split_map: np.ndarray, rng: np.random.Generator) -> dict:
    """Majority class and a per-pixel random forest on the same 8 channels (no spatial context)."""
    from sklearn.ensemble import RandomForestClassifier

    k = len(scene.classes)
    train = (split_map == 0) & (scene.y > 0) & scene.valid
    majority = np.bincount(scene.y[train]).argmax()
    out = {"majority": evaluate(np.where(scene.y > 0, majority, 0).astype(np.uint8), scene, split_map, 2)}
    idx = rng.choice(np.flatnonzero(train), size=min(150_000, int(train.sum())), replace=False)
    feats = np.asarray(scene.x).reshape(len(cfg.CHANNELS), -1)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=cfg.SEED,
                                class_weight="balanced_subsample")
    rf.fit(feats[:, idx].T, scene.y.ravel()[idx])
    test = np.flatnonzero((split_map.ravel() == 2) & scene.valid.ravel())
    pred = np.zeros(scene.y.size, dtype=np.uint8)
    pred[test] = rf.predict(feats[:, test].T)
    out["random_forest_pixel"] = evaluate(pred.reshape(scene.y.shape), scene, split_map, 2)
    out["random_forest_pixel"]["fit_seconds"] = round(time.time() - t0, 1)
    del k
    return out


def train(fast: bool = False, epochs: int = 40, steps: int = 120, batch: int = 8, lr: float = 1e-3,
          patience: int = 8, eval_every: int = 2, patch: int = 256, sample_power: float = 0.5,
          weight_power: float = 0.25) -> dict:
    torch.manual_seed(cfg.SEED)
    rng = np.random.default_rng(cfg.SEED)
    if fast:
        epochs, steps, eval_every = 3, 20, 1
    scene = load()
    k = len(scene.classes)
    blocks = make_split(scene.y, k)
    split_map = pixel_split(blocks, scene.y.shape)
    mean, std = channel_stats(scene.x, (split_map == 0) & scene.valid)
    dev = device()
    model = UNet(len(cfg.CHANNELS), k, cfg.BASE_FILTERS, mean=torch.from_numpy(mean),
                 std=torch.from_numpy(std)).to(dev)
    sampler = PatchSampler(scene, split_map, rng, patch, sample_power)
    weights = class_weights(scene, split_map, weight_power).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    h, w = scene.y.shape
    best, best_miou, bad, history = None, -1.0, 0, []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for _ in range(steps):
            xb, yb = sampler.batch(batch, mean)
            xb, yb = torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        row = {"epoch": epoch, "loss": total / steps, "seconds": round(time.time() - t0, 1)}
        if epoch % eval_every == 0 or epoch == epochs:
            pred, _ = predict_window(scene.x, mean, torch_runner(model, dev), 0, h, 0, w)
            val = evaluate(pred, scene, split_map, 1)
            row |= {"val_miou": val["mean_iou"], "val_oa": val["overall_accuracy"]}
            if val["mean_iou"] > best_miou:
                best_miou, bad = val["mean_iou"], 0
                best = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
                row["best"] = True
            else:
                bad += 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if bad >= patience:
            break
    train_seconds = time.time() - t_start

    model.load_state_dict(best)
    model.eval()
    pred, conf = predict_window(scene.x, mean, torch_runner(model, dev), 0, h, 0, w)
    results = {s: evaluate(pred, scene, split_map, i) for i, s in enumerate(SPLITS)}
    base = baselines(scene, split_map, rng)
    test = results["test"]
    metrics = {
        "metadata": {"trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "fast": fast,
                     "device": dev, "epochs_run": len(history), "train_seconds": round(train_seconds),
                     "torch": torch.__version__},
        "model": {"architecture": "U-Net", "base_filters": cfg.BASE_FILTERS, "parameters": n_params(model),
                  "channels": cfg.CHANNELS, "tile": cfg.TILE, "tile_margin": cfg.TILE_MARGIN},
        "training": {"epochs_max": epochs, "steps_per_epoch": steps, "batch": batch, "lr": lr, "patch": patch,
                     "sample_power": sample_power, "weight_power": weight_power, "patience": patience},
        "classes": scene.classes,
        "split": {"block": cfg.BLOCK, "blocks": {s: int((blocks == i).sum()) for i, s in enumerate(SPLITS)},
                  "labelled_pixels": {s: results[s]["pixels"] for s in SPLITS}},
        "normalisation": {"mean": mean.tolist(), "std": std.tolist()},
        "history": history,
        "unet": results,
        "baselines": base,
        "summary": {"test_overall_accuracy": test["overall_accuracy"], "test_mean_iou": test["mean_iou"],
                    "test_macro_f1": test["macro_f1"], "test_kappa": test["kappa"],
                    "rf_test_mean_iou": base["random_forest_pixel"]["mean_iou"],
                    "mean_confidence": float(conf[pred > 0].mean())},
    }
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_classes": k, "base": cfg.BASE_FILTERS,
                "channels": cfg.CHANNELS}, cfg.MODELS_DIR / "unet.pt")
    np.save(cfg.MODELS_DIR / "split_blocks.npy", blocks)
    (cfg.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def load_model(path=None) -> UNet:
    ck = torch.load(path or cfg.MODELS_DIR / "unet.pt", map_location="cpu", weights_only=True)
    model = UNet(len(ck["channels"]), ck["n_classes"], ck["base"])
    model.load_state_dict(ck["state_dict"])
    return model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--sample-power", type=float, default=0.5, help="class sampling ~ frequency ** p")
    ap.add_argument("--weight-power", type=float, default=0.25, help="loss weight ~ (1 / frequency) ** p")
    a = ap.parse_args()
    m = train(a.fast, a.epochs, a.steps, a.batch, a.lr, patch=a.patch, sample_power=a.sample_power,
              weight_power=a.weight_power)
    s = m["summary"]
    print(f"Test : OA {s['test_overall_accuracy']:.3f}, mIoU {s['test_mean_iou']:.3f}, F1 macro "
          f"{s['test_macro_f1']:.3f}, kappa {s['test_kappa']:.3f} (forêt aléatoire pixel : mIoU {s['rf_test_mean_iou']:.3f})")
