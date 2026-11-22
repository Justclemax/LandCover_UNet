//! `bundle.json` written by `landcover.export` (format `landcover-bundle/1`), validated on load.

use std::path::Path;

use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "landcover-bundle/1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bundle {
    pub format: String,
    pub model: ModelInfo,
    pub channels: Vec<String>,
    pub fill: Vec<f32>,
    pub tile: usize,
    pub tile_margin: usize,
    pub classes: Vec<LandClass>,
    pub scene: SceneInfo,
    pub metrics: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub file: String,
    pub architecture: String,
    pub parameters: u64,
    pub trained_at: String,
    pub fast: bool,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandClass {
    pub index: u8,
    pub code: u32,
    pub label: String,
    pub label_fr: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneInfo {
    pub file: String,
    pub layout: String,
    pub dtype: String,
    pub width: usize,
    pub height: usize,
    pub crs: String,
    /// GDAL-style affine transform `[a, b, c, d, e, f]`: x = c + col * a, y = f + row * e.
    pub transform: [f64; 6],
    pub bounds: [f64; 4],
    pub name: String,
    pub year: u32,
}

impl Bundle {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let b: Bundle =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        b.validate()?;
        Ok(b)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT {
            return Err(format!(
                "format {:?} inattendu (attendu {FORMAT})",
                self.format
            ));
        }
        if self.channels.is_empty() || self.fill.len() != self.channels.len() {
            return Err("fill doit avoir une valeur par canal".into());
        }
        if self.fill.iter().any(|v| !v.is_finite()) {
            return Err("fill contient une valeur non finie".into());
        }
        if self.tile == 0 || !self.tile.is_multiple_of(16) || 2 * self.tile_margin >= self.tile {
            return Err("tile doit être un multiple de 16 et tile_margin < tile / 2".into());
        }
        for (i, c) in self.classes.iter().enumerate() {
            if c.index as usize != i + 1 {
                return Err(format!(
                    "classe {i} : index {} au lieu de {}",
                    c.index,
                    i + 1
                ));
            }
            if parse_color(&c.color).is_none() {
                return Err(format!(
                    "classe {} : couleur {:?} invalide",
                    c.index, c.color
                ));
            }
        }
        let s = &self.scene;
        if s.layout != "HWC" || s.dtype != "f32le" {
            return Err("la scène doit être en HWC f32le".into());
        }
        if s.crs != "EPSG:4326"
            || s.transform[1] != 0.0
            || s.transform[3] != 0.0
            || s.transform[0] <= 0.0
            || s.transform[4] >= 0.0
        {
            return Err("la scène doit être en EPSG:4326, sans rotation, nord en haut".into());
        }
        if s.width == 0 || s.height == 0 {
            return Err("scène vide".into());
        }
        Ok(())
    }

    pub fn stride(&self) -> usize {
        self.tile - 2 * self.tile_margin
    }

    pub fn palette(&self) -> Vec<[u8; 3]> {
        self.classes
            .iter()
            .map(|c| parse_color(&c.color).unwrap_or([0, 0, 0]))
            .collect()
    }
}

/// `#rrggbb` -> RGB.
pub fn parse_color(s: &str) -> Option<[u8; 3]> {
    let h = s.strip_prefix('#')?;
    if h.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(h, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_parse() {
        assert_eq!(parse_color("#006400"), Some([0, 100, 0]));
        assert_eq!(parse_color("006400"), None);
        assert_eq!(parse_color("#00640"), None);
        assert_eq!(parse_color("#zz6400"), None);
    }
}
