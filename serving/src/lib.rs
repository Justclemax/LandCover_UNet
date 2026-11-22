//! HTTP API (axum) around the tract U-Net engine.
//!
//! Routes (JSON in, JSON out, errors as `{"error": {"code", "message"}}`):
//! - `GET  /health`              liveness, no auth
//! - `GET  /v1/model`            classes, palette, scene bounds, limits, test metrics
//! - `GET  /v1/openapi.json`     OpenAPI 3.1 description
//! - `POST /v1/predict/bbox`     classify a lon/lat box of the Kerinci scene -> PNG overlay + areas
//! - `POST /v1/predict/image`    classify a client image (H x W x C values, `null` = missing)
//! - `POST /v1/predict/raster`   same as binary float32 (up to 512 x 512), e.g. imagery fetched anywhere on Earth
//!
//! Safety limits: optional API key (`x-api-key`, constant-time comparison), body size limit,
//! window/image size limits, a semaphore bounding concurrent inferences (503 when the queue
//! wait expires) and a request timeout (504). Inference runs on blocking threads.

pub mod bundle;
pub mod engine;
pub mod geo;
pub mod source;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::engine::{Engine, Window};
use crate::source::MemImage;

pub const OPENAPI: &str = include_str!("../openapi.json");

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_window_pixels: usize,
    pub max_image_pixels: usize,
    pub max_raster_pixels: usize,
    pub max_body_bytes: usize,
    pub queue_wait: Duration,
    pub request_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_window_pixels: 1_200_000,
            max_image_pixels: 256 * 256,
            max_raster_pixels: 512 * 512,
            max_body_bytes: 12 << 20,
            queue_wait: Duration::from_secs(10),
            request_timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub permits: Arc<Semaphore>,
    pub api_key: Option<Arc<str>>,
    pub limits: Limits,
}

impl AppState {
    pub fn new(
        engine: Engine,
        max_concurrent: usize,
        api_key: Option<String>,
        limits: Limits,
    ) -> Self {
        Self {
            engine: Arc::new(engine),
            permits: Arc::new(Semaphore::new(max_concurrent.max(1))),
            api_key: api_key.filter(|k| !k.is_empty()).map(Into::into),
            limits,
        }
    }
}

// ----------------------------------------------------------------------------- errors
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    TooLarge(String),
    #[error("clé d'API absente ou invalide (en-tête x-api-key)")]
    Unauthorized,
    #[error("serveur occupé, réessayez dans quelques secondes")]
    Busy,
    #[error("délai de traitement dépassé")]
    Timeout,
    #[error("erreur interne : {0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ApiError::Invalid(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
            ApiError::TooLarge(_) => (StatusCode::PAYLOAD_TOO_LARGE, "too_large"),
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            ApiError::Busy => (StatusCode::SERVICE_UNAVAILABLE, "busy"),
            ApiError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "timeout"),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
        };
        if status.is_server_error() {
            tracing::error!(%status, error = %self);
        }
        let mut resp = (
            status,
            Json(json!({"error": {"code": code, "message": self.to_string()}})),
        )
            .into_response();
        if matches!(self, ApiError::Busy) {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        }
        resp
    }
}

fn parse<T: for<'de> Deserialize<'de>>(body: &Bytes) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::BadRequest(format!("JSON invalide : {e}")))
}

// ----------------------------------------------------------------------------- inference plumbing
/// Wait for an inference slot, run `f` on a blocking thread, bound the total time.
async fn infer<T, F>(state: &AppState, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&Engine) -> Result<T, String> + Send + 'static,
{
    let permit = tokio::time::timeout(
        state.limits.queue_wait,
        state.permits.clone().acquire_owned(),
    )
    .await
    .map_err(|_| ApiError::Busy)?
    .map_err(|_| ApiError::Busy)?;
    let engine = state.engine.clone();
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit; // released when the computation ends, even after a timeout
        f(&engine)
    });
    match tokio::time::timeout(state.limits.request_timeout, task).await {
        Err(_) => Err(ApiError::Timeout),
        Ok(Err(e)) => Err(ApiError::Internal(e.to_string())),
        Ok(Ok(r)) => r.map_err(ApiError::Internal),
    }
}

// ----------------------------------------------------------------------------- handlers
async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn model_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let b = &state.engine.bundle;
    let l = &state.limits;
    Json(json!({
        "model": {"architecture": b.model.architecture, "parameters": b.model.parameters,
                  "trained_at": b.model.trained_at, "fast": b.model.fast, "sha256": b.model.sha256,
                  "runtime": "tract (Rust)"},
        "channels": b.channels,
        "classes": b.classes,
        "scene": {"name": b.scene.name, "year": b.scene.year, "crs": b.scene.crs, "bounds": b.scene.bounds,
                  "width": b.scene.width, "height": b.scene.height,
                  "resolution_deg": b.scene.transform[0]},
        "tiling": {"tile": b.tile, "margin": b.tile_margin},
        "limits": {"max_window_pixels": l.max_window_pixels, "max_image_pixels": l.max_image_pixels,
                   "max_raster_pixels": l.max_raster_pixels,
                   "max_body_bytes": l.max_body_bytes, "auth": state.api_key.is_some()},
        "metrics": b.metrics,
    }))
}

async fn openapi() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/json")], OPENAPI)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BboxRequest {
    /// `[west, south, east, north]` in degrees (EPSG:4326).
    bbox: [f64; 4],
    #[serde(default)]
    include_classes: bool,
    #[serde(default = "default_opacity")]
    opacity: f32,
}

fn default_opacity() -> f32 {
    0.85
}

async fn predict_bbox(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let req: BboxRequest = parse(&body)?;
    if !(0.0..=1.0).contains(&req.opacity) {
        return Err(ApiError::Invalid("opacity doit être entre 0 et 1".into()));
    }
    let win =
        geo::bbox_to_window(&state.engine.bundle.scene, req.bbox).map_err(ApiError::Invalid)?;
    if win.pixels() > state.limits.max_window_pixels {
        return Err(ApiError::TooLarge(format!(
            "zone de {} pixels, maximum {} (environ {:.0} km²) : réduisez la bbox",
            win.pixels(),
            state.limits.max_window_pixels,
            state.limits.max_window_pixels as f64 * 0.09 / 100.0
        )));
    }
    let started = Instant::now();
    let opacity = (req.opacity * 255.0).round() as u8;
    let (pred, png) = infer(&state, move |e| {
        let p = e.predict(&e.scene, win)?;
        let png = geo::render_png(&e.bundle, &p, opacity)?;
        Ok((p, png))
    })
    .await?;
    let b = &state.engine.bundle;
    let mean_conf = mean_confidence(&pred.classes, &pred.confidence);
    let mut out = json!({
        "bounds": geo::window_bounds(&b.scene, win),
        "window": {"row": win.r0, "col": win.c0, "height": win.height(), "width": win.width()},
        "png_base64": base64::engine::general_purpose::STANDARD.encode(png),
        "areas": geo::class_areas(b, &pred, &|y| geo::pixel_area_ha(&b.scene, win.r0 + y)),
        "mean_confidence": mean_conf,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    if req.include_classes {
        out["classes"] = json!(pred.classes);
    }
    Ok(Json(out))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageRequest {
    width: usize,
    height: usize,
    /// Pixel-interleaved values (row by row, then column, then channel), `null` = missing.
    data: Vec<Option<f32>>,
}

async fn predict_image(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let req: ImageRequest = parse(&body)?;
    let n_ch = state.engine.bundle.channels.len();
    if req.width == 0 || req.height == 0 {
        return Err(ApiError::Invalid(
            "width et height doivent être positifs".into(),
        ));
    }
    let pixels = req
        .width
        .checked_mul(req.height)
        .ok_or_else(|| ApiError::TooLarge("image trop grande".into()))?;
    if pixels > state.limits.max_image_pixels {
        return Err(ApiError::TooLarge(format!(
            "image de {pixels} pixels, maximum {}",
            state.limits.max_image_pixels
        )));
    }
    if req.data.len() != pixels * n_ch {
        return Err(ApiError::Invalid(format!(
            "data doit contenir width x height x {n_ch} valeurs ({}), reçu {}",
            pixels * n_ch,
            req.data.len()
        )));
    }
    let data: Vec<f32> = req
        .data
        .into_iter()
        .map(|v| v.filter(|x| x.is_finite()).unwrap_or(f32::NAN))
        .collect();
    let img = MemImage {
        data,
        width: req.width,
        height: req.height,
        channels: n_ch,
    };
    let win = Window {
        r0: 0,
        r1: req.height,
        c0: 0,
        c1: req.width,
    };
    let started = Instant::now();
    let pred = infer(&state, move |e| e.predict(&img, win)).await?;
    Ok(Json(json!({
        "width": req.width,
        "height": req.height,
        "classes": pred.classes,
        "areas": geo::class_areas(&state.engine.bundle, &pred, &|_| 0.0),
        "mean_confidence": mean_confidence(&pred.classes, &pred.confidence),
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    })))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RasterQuery {
    width: usize,
    height: usize,
    west: Option<f64>,
    south: Option<f64>,
    east: Option<f64>,
    north: Option<f64>,
    #[serde(default)]
    include_classes: bool,
    opacity: Option<f32>,
}

/// Binary image: body = height x width x C little-endian float32 (pixel-interleaved, NaN =
/// missing), size in the query string. With west/south/east/north the areas are in hectares.
async fn predict_raster(
    State(state): State<AppState>,
    query: Result<Query<RasterQuery>, QueryRejection>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Query(q) =
        query.map_err(|e| ApiError::BadRequest(format!("paramètres invalides : {e}")))?;
    let n_ch = state.engine.bundle.channels.len();
    let opacity = q.opacity.unwrap_or(0.85);
    if !(0.0..=1.0).contains(&opacity) {
        return Err(ApiError::Invalid("opacity doit être entre 0 et 1".into()));
    }
    if q.width == 0 || q.height == 0 {
        return Err(ApiError::Invalid(
            "width et height doivent être positifs".into(),
        ));
    }
    let pixels = q
        .width
        .checked_mul(q.height)
        .ok_or_else(|| ApiError::TooLarge("image trop grande".into()))?;
    if pixels > state.limits.max_raster_pixels {
        return Err(ApiError::TooLarge(format!(
            "image de {pixels} pixels, maximum {}",
            state.limits.max_raster_pixels
        )));
    }
    if body.len() != pixels * n_ch * 4 {
        return Err(ApiError::Invalid(format!(
            "le corps doit faire width x height x {n_ch} x 4 = {} octets, reçu {}",
            pixels * n_ch * 4,
            body.len()
        )));
    }
    let bbox = match (q.west, q.south, q.east, q.north) {
        (Some(w), Some(s), Some(e), Some(n)) => {
            if ![w, s, e, n].iter().all(|v| v.is_finite())
                || w >= e
                || s >= n
                || s < -90.0
                || n > 90.0
            {
                return Err(ApiError::Invalid(
                    "bbox invalide (ouest < est, sud < nord, en degrés)".into(),
                ));
            }
            Some([w, s, e, n])
        }
        (None, None, None, None) => None,
        _ => {
            return Err(ApiError::Invalid(
                "donnez les quatre bornes west, south, east, north ou aucune".into(),
            ));
        }
    };
    let (words, _) = body.as_chunks::<4>();
    let data: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
    let img = MemImage {
        data,
        width: q.width,
        height: q.height,
        channels: n_ch,
    };
    let win = Window {
        r0: 0,
        r1: q.height,
        c0: 0,
        c1: q.width,
    };
    let started = Instant::now();
    let alpha = (opacity * 255.0).round() as u8;
    let (pred, png) = infer(&state, move |e| {
        let p = e.predict(&img, win)?;
        let png = geo::render_png(&e.bundle, &p, alpha)?;
        Ok((p, png))
    })
    .await?;
    let (w, h) = (q.width, q.height);
    let areas = match bbox {
        Some(b) => geo::class_areas(&state.engine.bundle, &pred, &|y| {
            geo::grid_row_area_ha(b, w, h, y)
        }),
        None => geo::class_areas(&state.engine.bundle, &pred, &|_| 0.0),
    };
    let mut out = json!({
        "width": w,
        "height": h,
        "bounds": bbox,
        "png_base64": base64::engine::general_purpose::STANDARD.encode(png),
        "areas": areas,
        "mean_confidence": mean_confidence(&pred.classes, &pred.confidence),
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    if q.include_classes {
        out["classes"] = json!(pred.classes);
    }
    Ok(Json(out))
}

fn mean_confidence(classes: &[u8], conf: &[f32]) -> f64 {
    let (sum, n) = classes
        .iter()
        .zip(conf)
        .filter(|(c, _)| **c > 0)
        .fold((0f64, 0u64), |(s, n), (_, p)| (s + *p as f64, n + 1));
    if n == 0 { 0.0 } else { sum / n as f64 }
}

async fn require_key(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if let Some(key) = &state.api_key {
        let given = req
            .headers()
            .get("x-api-key")
            .map(|v| v.as_bytes())
            .unwrap_or_default();
        if !bool::from(given.ct_eq(key.as_bytes())) {
            return Err(ApiError::Unauthorized);
        }
    }
    Ok(next.run(req).await)
}

/// `origins`: None = no CORS headers (server-to-server only), Some(["*"]) = any origin.
pub fn router(state: AppState, origins: Option<Vec<String>>) -> Router {
    let protected = Router::new()
        .route("/v1/predict/bbox", post(predict_bbox))
        .route("/v1/predict/image", post(predict_image))
        .route("/v1/predict/raster", post(predict_raster))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_key));
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/v1/model", get(model_info))
        .route("/v1/openapi.json", get(openapi))
        .merge(protected)
        .layer(RequestBodyLimitLayer::new(state.limits.max_body_bytes))
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    if let Some(origins) = origins {
        let allow = if origins.iter().any(|o| o == "*") {
            AllowOrigin::any()
        } else {
            AllowOrigin::list(origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()))
        };
        app = app.layer(
            CorsLayer::new()
                .allow_origin(allow)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }
    app
}
