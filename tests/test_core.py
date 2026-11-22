"""Unit tests that need no data download: tiling, split, metrics, geo, model, ONNX export."""

import numpy as np
import pytest
import torch

from landcover import config as cfg
from landcover.geo import bbox_to_window, row_area_ha
from landcover.metrics import confusion, scores
from landcover.model import UNet
from landcover.split import channel_stats, make_split, pixel_split
from landcover.tiling import predict_window, read_tile, softmax_max

K = 3


def pixel_runner(batch: np.ndarray) -> np.ndarray:
    """A 'model' whose logits depend only on each pixel: class = argmax of the first K channels."""
    return batch[:, :K].copy()


def test_read_tile_fills_outside_and_nan():
    x = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    x[1, 0, 0] = np.nan
    fill = np.array([-1.0, -2.0], dtype=np.float32)
    t = read_tile(x, fill, -1, -1, size=4)
    assert t[0, 0, 0] == -1 and t[1, 0, 3] == -2  # outside the scene
    assert t[1, 1, 1] == -2  # NaN replaced by that channel's fill
    assert t[0, 1, 1] == x[0, 0, 0] and t[0, 3, 3] == x[0, 2, 2]


def test_softmax_max_first_maximum_on_ties():
    logits = np.zeros((1, 3, 1, 1), dtype=np.float32)
    arg, prob = softmax_max(logits)
    assert arg[0, 0, 0] == 0 and prob[0, 0, 0] == pytest.approx(1 / 3)


def test_predict_window_is_window_independent():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(K, 400, 380)).astype(np.float32)
    x[:, 10, 20] = np.nan
    fill = np.zeros(K, dtype=np.float32)
    full, conf = predict_window(x, fill, pixel_runner, 0, 400, 0, 380)
    part, _ = predict_window(x, fill, pixel_runner, 100, 177, 50, 333)
    assert (part == full[100:177, 50:333]).all()
    assert (full[~np.isnan(x).any(axis=0)] == x.argmax(axis=0)[~np.isnan(x).any(axis=0)] + 1).all()
    assert full[10, 20] == 0 and conf[10, 20] == 0
    assert ((conf >= 1 / K - 1e-6) | (full == 0)).all() and (conf <= 1).all()


def test_split_assigns_blocks_with_target_shares():
    rng = np.random.default_rng(1)
    labels = rng.integers(1, 4, size=(cfg.BLOCK * 6, cfg.BLOCK * 5)).astype(np.uint8)
    blocks = make_split(labels, 3, tries=20)
    assert blocks.shape == (6, 5)
    counts = np.bincount(blocks.ravel(), minlength=3)
    assert counts[1] == round(30 * 0.15) and counts[2] == round(30 * 0.15)
    full = pixel_split(blocks, (cfg.BLOCK * 6 - 10, cfg.BLOCK * 5 - 3))
    assert full.shape == (cfg.BLOCK * 6 - 10, cfg.BLOCK * 5 - 3)
    assert (full[: cfg.BLOCK, : cfg.BLOCK] == blocks[0, 0]).all()


def test_channel_stats_ignores_nan_and_mask():
    x = np.array([[[1, 3], [np.nan, 100]]], dtype=np.float32)
    mean, std = channel_stats(x, np.array([[True, True], [True, False]]))
    assert mean[0] == pytest.approx(2) and std[0] == pytest.approx(1)


def test_metrics():
    y = np.array([1, 1, 2, 2, 3, 0])
    p = np.array([1, 2, 2, 2, 3, 3])
    cm = confusion(y, p, 3)
    assert cm.sum() == 5 and cm[0, 1] == 1
    s = scores(cm)
    assert s["overall_accuracy"] == pytest.approx(0.8)
    assert s["per_class"]["iou"][0] == pytest.approx(0.5)
    perfect = scores(confusion(y, np.where(y > 0, y, 1), 3))
    assert perfect["mean_iou"] == 1 and perfect["kappa"] == pytest.approx(1)


def test_geo_matches_rust_formulas():
    meta = {"transform": [0.01, 0.0, 101.0, 0.0, -0.01, -1.0], "width": 100, "height": 50}
    assert bbox_to_window(meta, (101.105, -1.2, 101.2, -1.1)) == (10, 20, 10, 20)  # same case as geo.rs
    assert bbox_to_window(meta, (100, -3, 103, 0)) == (0, 50, 0, 100)
    with pytest.raises(ValueError):
        bbox_to_window(meta, (105, -1.2, 106, -1.1))
    assert row_area_ha(meta)[20] == pytest.approx(123.0, abs=1)


def test_unet_shapes_and_normalisation():
    mean, std = torch.arange(8.0), torch.full((8,), 2.0)
    m = UNet(8, 5, base=4, mean=mean, std=std).eval()
    out = m(torch.zeros(2, 8, 64, 48))
    assert out.shape == (2, 5, 64, 48)
    assert torch.equal(m.mean.flatten(), mean)


def test_onnx_export_matches_torch(tmp_path, monkeypatch):
    import onnxruntime as ort

    from landcover import export

    monkeypatch.setattr(cfg, "TILE", 32)
    m = UNet(len(cfg.CHANNELS), 4, base=4).eval()
    export.export_onnx(m, tmp_path / "m.onnx")
    x = np.random.default_rng(2).normal(size=(1, len(cfg.CHANNELS), 32, 32)).astype(np.float32)
    got = ort.InferenceSession(str(tmp_path / "m.onnx")).run(None, {"input": x})[0]
    with torch.inference_mode():
        want = m(torch.from_numpy(x)).numpy()
    assert np.abs(got - want).max() < 1e-4
