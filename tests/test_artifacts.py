"""Consistency of the committed artifacts (skipped when they are absent)."""

import json

import pytest

from landcover import config as cfg

METRICS = cfg.MODELS_DIR / "metrics.json"
BUNDLE = cfg.RUST_DIR / "bundle.json"
SUMMARY = cfg.REPORTS_DIR / "summary.json"
needs = pytest.mark.skipif(not (METRICS.exists() and BUNDLE.exists() and SUMMARY.exists()),
                           reason="artefacts absents : lancer train, export et predict")


@needs
def test_metrics_are_sane():
    m = json.loads(METRICS.read_text())
    s = m["summary"]
    assert 0 < s["test_mean_iou"] <= s["test_overall_accuracy"] <= 1
    assert set(m["unet"]) == {"train", "val", "test"}
    assert m["split"]["blocks"]["train"] > m["split"]["blocks"]["test"] > 0


@needs
def test_bundle_matches_trained_model():
    """A stale bundle (exported before the last training) would serve another model."""
    m, b = json.loads(METRICS.read_text()), json.loads(BUNDLE.read_text())
    assert b["model"]["trained_at"] == m["metadata"]["trained_at"]
    assert [c["code"] for c in b["classes"]] == [c["code"] for c in m["classes"]]
    assert b["fill"] == m["normalisation"]["mean"]
    assert b["channels"] == cfg.CHANNELS


@needs
def test_map_layers_match_model():
    s, m = json.loads(SUMMARY.read_text()), json.loads(METRICS.read_text())
    assert s["model"]["trained_at"] == m["metadata"]["trained_at"]
    for name in ("satellite.jpg", "prediction.png", "reference.png", "disagreement.png", "confidence.png"):
        assert (cfg.ROOT / "app" / "static" / "layers" / name).exists(), name
