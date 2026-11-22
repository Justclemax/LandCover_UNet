//! ONNX U-Net executed by tract (pure Rust) and the tiled inference of `landcover/tiling.py`.

use std::path::Path;

use tract_onnx::prelude::*;

use crate::bundle::Bundle;
use crate::source::{FileScene, Source};

type Plan = TypedRunnableModel<TypedModel>;
/// Core of one tile: origin (row, col) in the window, classes and confidences (stride x stride).
type TileCore = (usize, usize, Vec<u8>, Vec<f32>);

/// Inclusive-exclusive pixel window `[r0, r1) x [c0, c1)` of a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub r0: usize,
    pub r1: usize,
    pub c0: usize,
    pub c1: usize,
}

impl Window {
    pub fn height(&self) -> usize {
        self.r1 - self.r0
    }
    pub fn width(&self) -> usize {
        self.c1 - self.c0
    }
    pub fn pixels(&self) -> usize {
        self.height() * self.width()
    }
}

/// Class index 1..K per pixel (0 = missing input) and the softmax probability of that class.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub window: Window,
    pub classes: Vec<u8>,
    pub confidence: Vec<f32>,
}

pub struct Engine {
    plan: Plan,
    pub bundle: Bundle,
    pub scene: FileScene,
    threads: usize,
}

impl Engine {
    pub fn load(dir: &Path, threads: usize) -> Result<Self, String> {
        let bundle = Bundle::from_file(&dir.join("bundle.json"))?;
        let s = &bundle.scene;
        let scene = FileScene::open(&dir.join(&s.file), s.width, s.height, bundle.channels.len())
            .map_err(|e| format!("scène : {e}"))?;
        let shape = [1, bundle.channels.len(), bundle.tile, bundle.tile];
        let plan = tract_onnx::onnx()
            .model_for_path(dir.join(&bundle.model.file))
            .and_then(|m| m.with_input_fact(0, f32::fact(shape).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("modèle ONNX : {e}"))?;
        let engine = Self {
            plan,
            bundle,
            scene,
            threads: threads.max(1),
        };
        // Fail at start-up, not on the first request, if the graph output is not (1, K, T, T).
        let tile = vec![0.0f32; engine.tile_len()];
        let out = engine.run_tile(&tile)?;
        if out.len() != engine.bundle.classes.len() * engine.bundle.tile * engine.bundle.tile {
            return Err(format!(
                "sortie du modèle de {} valeurs, incompatible avec {} classes",
                out.len(),
                engine.bundle.classes.len()
            ));
        }
        Ok(engine)
    }

    fn tile_len(&self) -> usize {
        self.bundle.channels.len() * self.bundle.tile * self.bundle.tile
    }

    fn run_tile(&self, chw: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.bundle.tile;
        let input = Tensor::from_shape(&[1, self.bundle.channels.len(), t, t], chw)
            .map_err(|e| e.to_string())?;
        let out = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| e.to_string())?;
        Ok(out[0]
            .as_slice::<f32>()
            .map_err(|e| e.to_string())?
            .to_vec())
    }

    /// (C, T, T) tile with top-left pixel `(r, c)` in source coordinates (may be negative or
    /// past the edge): outside pixels and NaN are replaced by the channel fill value.
    fn read_tile<S: Source>(
        &self,
        src: &S,
        r: i64,
        c: i64,
        out: &mut [f32],
        row: &mut Vec<f32>,
    ) -> Result<(), String> {
        let (t, n_ch) = (self.bundle.tile, self.bundle.channels.len());
        for (ch, &f) in self.bundle.fill.iter().enumerate() {
            out[ch * t * t..(ch + 1) * t * t].fill(f);
        }
        let (h, w) = (src.height() as i64, src.width() as i64);
        let (cc0, cc1) = (c.max(0), (c + t as i64).min(w));
        if cc0 >= cc1 {
            return Ok(());
        }
        row.resize((cc1 - cc0) as usize * n_ch, 0.0);
        for rr in r.max(0)..(r + t as i64).min(h) {
            src.read_row(rr as usize, cc0 as usize, cc1 as usize, row)
                .map_err(|e| format!("lecture : {e}"))?;
            let y = (rr - r) as usize;
            for (i, px) in row.chunks_exact(n_ch).enumerate() {
                let x = (cc0 - c) as usize + i;
                for (ch, &v) in px.iter().enumerate() {
                    if v.is_finite() {
                        out[ch * t * t + y * t + x] = v;
                    }
                }
            }
        }
        Ok(())
    }

    /// Classes and confidence of the tile core (stride x stride), row-major.
    fn core(&self, logits: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let (t, m, s, k) = (
            self.bundle.tile,
            self.bundle.tile_margin,
            self.bundle.stride(),
            self.bundle.classes.len(),
        );
        let mut classes = Vec::with_capacity(s * s);
        let mut conf = Vec::with_capacity(s * s);
        for y in m..m + s {
            for x in m..m + s {
                let at = |j: usize| logits[j * t * t + y * t + x];
                let (mut best, mut top) = (0usize, at(0));
                for j in 1..k {
                    if at(j) > top {
                        best = j;
                        top = at(j);
                    }
                }
                let sum: f32 = (0..k).map(|j| (at(j) - top).exp()).sum();
                classes.push(best as u8 + 1);
                conf.push(1.0 / sum);
            }
        }
        (classes, conf)
    }

    /// Tiled prediction of a window of `src`, tiles spread over `self.threads` threads.
    pub fn predict<S: Source>(&self, src: &S, win: Window) -> Result<Prediction, String> {
        if win.r1 > src.height() || win.c1 > src.width() || win.r0 >= win.r1 || win.c0 >= win.c1 {
            return Err("fenêtre hors de la source".into());
        }
        let (m, s) = (self.bundle.tile_margin as i64, self.bundle.stride());
        let origins: Vec<(usize, usize)> = (0..win.height())
            .step_by(s)
            .flat_map(|oy| (0..win.width()).step_by(s).map(move |ox| (oy, ox)))
            .collect();
        let per_thread = origins.len().div_ceil(self.threads.min(origins.len()));
        let cores: Vec<Result<Vec<TileCore>, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = origins
                .chunks(per_thread)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut tile = vec![0.0f32; self.tile_len()];
                        let mut row = Vec::new();
                        let mut done = Vec::with_capacity(chunk.len());
                        for &(oy, ox) in chunk {
                            let (r, c) = ((win.r0 + oy) as i64 - m, (win.c0 + ox) as i64 - m);
                            self.read_tile(src, r, c, &mut tile, &mut row)?;
                            let (cls, conf) = self.core(&self.run_tile(&tile)?);
                            done.push((oy, ox, cls, conf));
                        }
                        Ok(done)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("thread d'inférence interrompu".into()))
                })
                .collect()
        });

        let (h, w) = (win.height(), win.width());
        let mut classes = vec![0u8; h * w];
        let mut confidence = vec![0.0f32; h * w];
        for part in cores {
            for (oy, ox, cls, conf) in part? {
                let (hh, ww) = (s.min(h - oy), s.min(w - ox));
                for y in 0..hh {
                    let dst = (oy + y) * w + ox;
                    classes[dst..dst + ww].copy_from_slice(&cls[y * s..y * s + ww]);
                    confidence[dst..dst + ww].copy_from_slice(&conf[y * s..y * s + ww]);
                }
            }
        }
        // Pixels with a missing input channel get class 0.
        let n_ch = src.channels();
        let mut row = vec![0.0f32; w * n_ch];
        for y in 0..h {
            src.read_row(win.r0 + y, win.c0, win.c1, &mut row)
                .map_err(|e| format!("lecture : {e}"))?;
            for (x, px) in row.chunks_exact(n_ch).enumerate() {
                if px.iter().any(|v| !v.is_finite()) {
                    classes[y * w + x] = 0;
                    confidence[y * w + x] = 0.0;
                }
            }
        }
        Ok(Prediction {
            window: win,
            classes,
            confidence,
        })
    }
}
