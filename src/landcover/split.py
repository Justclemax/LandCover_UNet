"""Spatial train/val/test split by blocks, and per-channel normalisation statistics.

The reference notebook samples overlapping patches anywhere in the scene, then splits them
at random: neighbouring (even identical) pixels end up in train and test, which inflates
the score. Here every BLOCK x BLOCK block belongs to exactly one split, and labels outside
the training blocks are never used for training.
"""

import numpy as np

from landcover import config as cfg

SPLITS = ("train", "val", "test")


def block_grid(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return -(-h // cfg.BLOCK), -(-w // cfg.BLOCK)


def _score(assign: np.ndarray, block_counts: np.ndarray) -> float:
    """Worst ratio between a class's share in a split and the split's target share."""
    total = block_counts.sum(axis=0)
    worst = np.inf
    for s, name in enumerate(SPLITS):
        share = block_counts[assign == s].sum(axis=0) / np.maximum(total, 1)
        worst = min(worst, float((share / cfg.SPLIT_SHARES[name]).min()))
    return worst


def make_split(labels: np.ndarray, n_classes: int, tries: int = 400, seed: int = cfg.SEED) -> np.ndarray:
    """Block assignment (ny, nx) with values 0/1/2 = train/val/test.

    Tries `tries` seeded random assignments with the target shares and keeps the one where
    every class is best represented in every split (rare classes sit in few blocks).
    """
    ny, nx = block_grid(labels.shape)
    counts = np.zeros((ny * nx, n_classes + 1), dtype=np.int64)
    for by in range(ny):
        for bx in range(nx):
            blk = labels[by * cfg.BLOCK:(by + 1) * cfg.BLOCK, bx * cfg.BLOCK:(bx + 1) * cfg.BLOCK]
            counts[by * nx + bx] = np.bincount(blk.ravel(), minlength=n_classes + 1)
    counts = counts[:, 1:]  # drop IGNORE
    labelled = counts.sum(axis=1) > 0
    n = int(labelled.sum())
    n_val, n_test = round(n * cfg.SPLIT_SHARES["val"]), round(n * cfg.SPLIT_SHARES["test"])
    rng = np.random.default_rng(seed)
    best, best_score = None, -np.inf
    idx = np.flatnonzero(labelled)
    for _ in range(tries):
        assign = np.zeros(ny * nx, dtype=np.int8)
        perm = rng.permutation(idx)
        assign[perm[:n_val]] = 1
        assign[perm[n_val:n_val + n_test]] = 2
        sc = _score(assign, counts)
        if sc > best_score:
            best, best_score = assign, sc
    return best.reshape(ny, nx)


def pixel_split(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Expand the block assignment to a (H, W) int8 map."""
    full = np.repeat(np.repeat(blocks, cfg.BLOCK, axis=0), cfg.BLOCK, axis=1)
    return full[:shape[0], :shape[1]]


def channel_stats(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each channel over the masked (training) pixels, ignoring NaN."""
    mean, std = [], []
    for c in range(x.shape[0]):
        v = np.asarray(x[c])[mask]
        v = v[np.isfinite(v)]
        mean.append(v.mean())
        std.append(v.std())
    return np.array(mean, dtype=np.float32), np.maximum(np.array(std, dtype=np.float32), 1e-6)
