//! Geographic helpers for an EPSG:4326 north-up grid: bbox <-> pixel window, pixel areas,
//! class areas and the PNG overlay sent to map clients.

use serde::Serialize;

use crate::bundle::{Bundle, SceneInfo};
use crate::engine::{Prediction, Window};

/// Mean Earth radius-based metres per degree (WGS84 approximations, < 0.5 % error here).
const M_PER_DEG_LAT: f64 = 110_574.0;
const M_PER_DEG_LON_EQUATOR: f64 = 111_320.0;

/// Pixel window covering `[west, south, east, north]`, clipped to the scene.
pub fn bbox_to_window(scene: &SceneInfo, bbox: [f64; 4]) -> Result<Window, String> {
    let [west, south, east, north] = bbox;
    if bbox.iter().any(|v| !v.is_finite()) {
        return Err("bbox contient une valeur non finie".into());
    }
    if !(-180.0..=180.0).contains(&west)
        || !(-180.0..=180.0).contains(&east)
        || !(-90.0..=90.0).contains(&south)
        || !(-90.0..=90.0).contains(&north)
    {
        return Err("bbox doit être en degrés [ouest, sud, est, nord]".into());
    }
    if west >= east || south >= north {
        return Err("bbox doit vérifier ouest < est et sud < nord".into());
    }
    let [a, _, c, _, e, f] = scene.transform;
    let col = |x: f64| (x - c) / a;
    let row = |y: f64| (y - f) / e;
    // EPS absorbs decimal-to-binary error, so a bbox on pixel edges maps to exactly those pixels.
    const EPS: f64 = 1e-6;
    let c0 = (col(west) + EPS).floor().max(0.0);
    let c1 = (col(east) - EPS).ceil().min(scene.width as f64);
    let r0 = (row(north) + EPS).floor().max(0.0);
    let r1 = (row(south) - EPS).ceil().min(scene.height as f64);
    if c0 >= c1 || r0 >= r1 {
        return Err("la bbox ne recoupe pas la scène".into());
    }
    Ok(Window {
        r0: r0 as usize,
        r1: r1 as usize,
        c0: c0 as usize,
        c1: c1 as usize,
    })
}

/// `[west, south, east, north]` of a pixel window.
pub fn window_bounds(scene: &SceneInfo, w: Window) -> [f64; 4] {
    let [a, _, c, _, e, f] = scene.transform;
    [
        c + w.c0 as f64 * a,
        f + w.r1 as f64 * e,
        c + w.c1 as f64 * a,
        f + w.r0 as f64 * e,
    ]
}

/// Area of one pixel of scene row `r`, in hectares.
pub fn pixel_area_ha(scene: &SceneInfo, r: usize) -> f64 {
    let [a, _, _, _, e, f] = scene.transform;
    let lat = (f + (r as f64 + 0.5) * e).to_radians();
    a * M_PER_DEG_LON_EQUATOR * lat.cos() * (-e) * M_PER_DEG_LAT / 10_000.0
}

#[derive(Debug, Serialize)]
pub struct ClassArea {
    pub index: u8,
    pub label_fr: String,
    pub color: String,
    pub pixels: u64,
    pub hectares: f64,
    pub share: f64,
}

/// Hectares per predicted class (classes with at least one pixel), largest first.
/// `scene_rows` is false for client images, whose pixels have no known position.
pub fn class_areas(bundle: &Bundle, p: &Prediction, scene_rows: bool) -> Vec<ClassArea> {
    let k = bundle.classes.len();
    let (mut pixels, mut ha) = (vec![0u64; k + 1], vec![0f64; k + 1]);
    let w = p.window.width();
    for (y, row) in p.classes.chunks_exact(w).enumerate() {
        let area = if scene_rows {
            pixel_area_ha(&bundle.scene, p.window.r0 + y)
        } else {
            0.0
        };
        for &c in row {
            pixels[c as usize] += 1;
            ha[c as usize] += area;
        }
    }
    let total: u64 = pixels[1..].iter().sum();
    let mut out: Vec<ClassArea> = bundle
        .classes
        .iter()
        .filter(|c| pixels[c.index as usize] > 0)
        .map(|c| ClassArea {
            index: c.index,
            label_fr: c.label_fr.clone(),
            color: c.color.clone(),
            pixels: pixels[c.index as usize],
            hectares: (ha[c.index as usize] * 10.0).round() / 10.0,
            share: pixels[c.index as usize] as f64 / total.max(1) as f64,
        })
        .collect();
    out.sort_by_key(|a| std::cmp::Reverse(a.pixels));
    out
}

/// RGBA PNG of the class map (transparent where class 0).
pub fn render_png(bundle: &Bundle, p: &Prediction, opacity: u8) -> Result<Vec<u8>, String> {
    let palette = bundle.palette();
    let mut rgba = Vec::with_capacity(p.classes.len() * 4);
    for &c in &p.classes {
        if c == 0 {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let [r, g, b] = palette[c as usize - 1];
            rgba.extend_from_slice(&[r, g, b, opacity]);
        }
    }
    let mut out = Vec::new();
    {
        let mut enc =
            png::Encoder::new(&mut out, p.window.width() as u32, p.window.height() as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&rgba).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> SceneInfo {
        SceneInfo {
            file: "scene.bin".into(),
            layout: "HWC".into(),
            dtype: "f32le".into(),
            width: 100,
            height: 50,
            crs: "EPSG:4326".into(),
            transform: [0.01, 0.0, 101.0, 0.0, -0.01, -1.0],
            bounds: [101.0, -1.5, 102.0, -1.0],
            name: "test".into(),
            year: 2023,
        }
    }

    #[test]
    fn bbox_round_trip() {
        let s = scene();
        let w = bbox_to_window(&s, [101.105, -1.2, 101.2, -1.1]).unwrap();
        assert_eq!(
            w,
            Window {
                r0: 10,
                r1: 20,
                c0: 10,
                c1: 20
            }
        );
        let b = window_bounds(&s, w);
        assert!((b[0] - 101.1).abs() < 1e-9 && (b[3] + 1.1).abs() < 1e-9);
    }

    #[test]
    fn bbox_is_clipped_and_validated() {
        let s = scene();
        assert_eq!(
            bbox_to_window(&s, [100.0, -3.0, 103.0, 0.0]).unwrap(),
            Window {
                r0: 0,
                r1: 50,
                c0: 0,
                c1: 100
            }
        );
        assert!(bbox_to_window(&s, [105.0, -1.2, 106.0, -1.1]).is_err());
        assert!(bbox_to_window(&s, [101.2, -1.2, 101.1, -1.1]).is_err());
        assert!(bbox_to_window(&s, [f64::NAN, -1.2, 101.1, -1.1]).is_err());
        assert!(bbox_to_window(&s, [101.0, -100.0, 101.1, -1.1]).is_err());
    }

    #[test]
    fn pixel_area_near_equator() {
        // 0.01 degree at ~1.2 degrees south: about 1 113 m x 1 106 m = 123 ha.
        let ha = pixel_area_ha(&scene(), 20);
        assert!((ha - 123.0).abs() < 1.0, "{ha}");
    }
}
