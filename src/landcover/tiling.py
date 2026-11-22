"""Tiled inference over any window of the scene. The Rust server implements the same algorithm.

A window [r0, r1) x [c0, c1) is covered by TILE x TILE tiles with a stride of
TILE - 2 * TILE_MARGIN. Each tile is read from the scene with TILE_MARGIN pixels of real
context on every side (pixels outside the scene and missing pixels are filled with the
channel mean, i.e. 0 after normalisation), and only its central stride x stride core is
kept. So a window gives the same classes whatever its size, except within TILE_MARGIN
pixels of the scene edge.

Output: class index 1..K (0 where an input channel is missing) and the softmax
probability of that class.
"""

from collections.abc import Callable

import numpy as np

from landcover import config as cfg

RunFn = Callable[[np.ndarray], np.ndarray]  # (N, C, T, T) float32 -> (N, K, T, T) logits


def read_tile(x: np.ndarray, fill: np.ndarray, r: int, c: int, size: int = cfg.TILE) -> np.ndarray:
    """(C, size, size) tile whose top-left pixel is (r, c) in scene coordinates."""
    n_ch, h, w = x.shape
    out = np.broadcast_to(fill.reshape(-1, 1, 1), (n_ch, size, size)).copy()
    rr0, rr1, cc0, cc1 = max(r, 0), min(r + size, h), max(c, 0), min(c + size, w)
    if rr0 < rr1 and cc0 < cc1:
        part = np.asarray(x[:, rr0:rr1, cc0:cc1], dtype=np.float32)
        part = np.where(np.isfinite(part), part, fill.reshape(-1, 1, 1))
        out[:, rr0 - r:rr1 - r, cc0 - c:cc1 - c] = part
    return out


def softmax_max(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, h, w) -> argmax (first maximum on ties) and its softmax probability."""
    arg = logits.argmax(axis=1)
    top = np.take_along_axis(logits, arg[:, None], axis=1)[:, 0]
    prob = 1.0 / np.exp(logits - top[:, None]).sum(axis=1)
    return arg, prob.astype(np.float32)


def predict_window(x: np.ndarray, fill: np.ndarray, run: RunFn, r0: int, r1: int, c0: int, c1: int,
                   batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    m, stride = cfg.TILE_MARGIN, cfg.TILE - 2 * cfg.TILE_MARGIN
    h, w = r1 - r0, c1 - c0
    classes = np.zeros((h, w), dtype=np.uint8)
    conf = np.zeros((h, w), dtype=np.float32)
    origins = [(oy, ox) for oy in range(0, h, stride) for ox in range(0, w, stride)]
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        tiles = np.stack([read_tile(x, fill, r0 + oy - m, c0 + ox - m) for oy, ox in chunk])
        arg, prob = softmax_max(run(tiles)[:, :, m:m + stride, m:m + stride])
        for (oy, ox), a, p in zip(chunk, arg, prob, strict=True):
            hh, ww = min(stride, h - oy), min(stride, w - ox)
            classes[oy:oy + hh, ox:ox + ww] = a[:hh, :ww] + 1
            conf[oy:oy + hh, ox:ox + ww] = p[:hh, :ww]
    valid = np.isfinite(np.asarray(x[:, r0:r1, c0:c1])).all(axis=0)
    classes[~valid] = 0
    conf[~valid] = 0
    return classes, conf


def torch_runner(model, device: str) -> RunFn:
    import torch

    model = model.to(device).eval()

    def run(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(batch).to(device)).float().cpu().numpy()

    return run


def onnx_runner(path) -> RunFn:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def run(batch: np.ndarray) -> np.ndarray:
        return np.concatenate([sess.run(None, {name: t[None]})[0] for t in batch])

    return run
