//! Rust (tract) must reproduce the Python reference (onnxruntime + landcover/tiling.py) on the
//! windows of models/rust/parity.json, and the HTTP API must behave. Skipped without a bundle.

use std::path::PathBuf;
use std::sync::OnceLock;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use landcover_serve::engine::{Engine, Window};
use landcover_serve::{AppState, Limits, router};
use serde::Deserialize;
use tower::ServiceExt;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/rust")
}

fn have_bundle() -> bool {
    let ok = dir().join("bundle.json").exists() && dir().join("scene.bin").exists();
    if !ok {
        eprintln!("bundle absent : lancez `uv run python -m landcover.export`");
    }
    ok
}

fn engine() -> &'static Engine {
    static E: OnceLock<Engine> = OnceLock::new();
    E.get_or_init(|| Engine::load(&dir(), 4).expect("bundle"))
}

#[derive(Deserialize)]
struct Case {
    window: [usize; 4],
    classes: Vec<u8>,
    confidence: Vec<f32>,
}

#[test]
fn matches_python_reference() {
    if !have_bundle() {
        return;
    }
    let cases: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(dir().join("parity.json")).unwrap()).unwrap();
    let e = engine();
    for case in cases {
        let [r0, r1, c0, c1] = case.window;
        let p = e.predict(&e.scene, Window { r0, r1, c0, c1 }).unwrap();
        assert_eq!(p.classes.len(), case.classes.len());
        let same = p
            .classes
            .iter()
            .zip(&case.classes)
            .filter(|(a, b)| a == b)
            .count();
        let agreement = same as f64 / p.classes.len() as f64;
        let max_dp = p
            .confidence
            .iter()
            .zip(&case.confidence)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "{:?}: accord {agreement:.6}, écart max de probabilité {max_dp:.2e}",
            case.window
        );
        // Different float kernels can flip a pixel whose two best logits are within ~1e-6.
        assert!(agreement >= 0.999, "{:?}: accord {agreement}", case.window);
        assert!(
            max_dp < 1e-3,
            "{:?}: écart de probabilité {max_dp}",
            case.window
        );
    }
}

#[test]
fn window_independence() {
    // The core of a big window equals the same pixels predicted through a small window.
    if !have_bundle() {
        return;
    }
    let e = engine();
    let big = e
        .predict(
            &e.scene,
            Window {
                r0: 1000,
                r1: 1400,
                c0: 1000,
                c1: 1400,
            },
        )
        .unwrap();
    let small = e
        .predict(
            &e.scene,
            Window {
                r0: 1100,
                r1: 1150,
                c0: 1210,
                c1: 1290,
            },
        )
        .unwrap();
    let mut same = 0;
    for y in 0..50 {
        for x in 0..80 {
            same += (big.classes[(100 + y) * 400 + 210 + x] == small.classes[y * 80 + x]) as usize;
        }
    }
    assert!(same as f64 / 4000.0 > 0.97, "{same}/4000");
}

async fn call(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: &str,
    key: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    let mut req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(k) = key {
        req = req.header("x-api-key", k);
    }
    let resp = app
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

fn app(key: Option<&str>) -> axum::Router {
    let state = AppState::new(
        Engine::load(&dir(), 2).unwrap(),
        1,
        key.map(String::from),
        Limits::default(),
    );
    router(state, None)
}

#[tokio::test]
async fn http_api() {
    if !have_bundle() {
        return;
    }
    let a = app(None);
    let (s, v) = call(a.clone(), "GET", "/health", "", None).await;
    assert_eq!((s, v["status"].as_str()), (StatusCode::OK, Some("ok")));
    let (s, v) = call(a.clone(), "GET", "/v1/model", "", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(v["classes"].as_array().unwrap().len() >= 10);

    let (s, v) = call(
        a.clone(),
        "POST",
        "/v1/predict/bbox",
        r#"{"bbox": [101.3, -2.1, 101.35, -2.05], "include_classes": true}"#,
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let n = v["window"]["width"].as_u64().unwrap() * v["window"]["height"].as_u64().unwrap();
    assert_eq!(v["classes"].as_array().unwrap().len() as u64, n);
    let share: f64 = v["areas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["share"].as_f64().unwrap())
        .sum();
    assert!((share - 1.0).abs() < 1e-6);
    assert!(v["png_base64"].as_str().unwrap().starts_with("iVBORw0KGgo"));

    for (body, code) in [
        (
            r#"{"bbox": [10, 10, 11, 11]}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            r#"{"bbox": [101.35, -2.1, 101.3, -2.05]}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            r#"{"bbox": [101.0, -2.6, 101.8, -1.7]}"#,
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (r#"{"bbox": [1, 2]}"#, StatusCode::BAD_REQUEST),
        (
            r#"{"bbox": [101.3, -2.1, 101.35, -2.05], "extra": 1}"#,
            StatusCode::BAD_REQUEST,
        ),
        ("pas du json", StatusCode::BAD_REQUEST),
    ] {
        let (s, v) = call(a.clone(), "POST", "/v1/predict/bbox", body, None).await;
        assert_eq!(s, code, "{body} -> {v}");
        assert!(v["error"]["message"].is_string());
    }

    // A client image: 20 x 10 pixels of 8 channels, one missing pixel.
    let mut data: Vec<serde_json::Value> = Vec::new();
    for i in 0..200 {
        for v in [0.03, 0.05, 0.04, 0.3, 0.15, 0.07, -7.0, -12.0] {
            data.push(if i == 0 {
                serde_json::Value::Null
            } else {
                serde_json::json!(v)
            });
        }
    }
    let body = serde_json::json!({"width": 20, "height": 10, "data": data}).to_string();
    let (s, v) = call(a.clone(), "POST", "/v1/predict/image", &body, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let classes = v["classes"].as_array().unwrap();
    assert_eq!(classes.len(), 200);
    assert_eq!(classes[0], 0);
    let (s, _) = call(
        a,
        "POST",
        "/v1/predict/image",
        r#"{"width": 2, "height": 2, "data": [1.0]}"#,
        None,
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn api_key_is_enforced() {
    if !have_bundle() {
        return;
    }
    let a = app(Some("secret"));
    let body = r#"{"bbox": [101.3, -2.1, 101.31, -2.09]}"#;
    assert_eq!(
        call(a.clone(), "POST", "/v1/predict/bbox", body, None)
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        call(a.clone(), "POST", "/v1/predict/bbox", body, Some("wrong"))
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        call(a.clone(), "POST", "/v1/predict/bbox", body, Some("secret"))
            .await
            .0,
        StatusCode::OK
    );
    assert_eq!(call(a, "GET", "/health", "", None).await.0, StatusCode::OK);
}

async fn call_raw(app: axum::Router, uri: &str, body: Vec<u8>) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/octet-stream")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

#[tokio::test]
async fn raster_route() {
    if !have_bundle() {
        return;
    }
    let a = app(None);
    // A real 40 x 30 window of the scene, sent as raw float32: same classes as the scene route.
    let e = engine();
    let win = Window {
        r0: 1300,
        r1: 1330,
        c0: 1000,
        c1: 1040,
    };
    let mut body = Vec::new();
    let mut row = vec![0f32; 40 * 8];
    use landcover_serve::source::Source;
    for r in win.r0..win.r1 {
        e.scene.read_row(r, win.c0, win.c1, &mut row).unwrap();
        for v in &row {
            body.extend_from_slice(&v.to_le_bytes());
        }
    }
    let uri = "/v1/predict/raster?width=40&height=30&west=101.3&south=-2.1&east=101.31&north=-2.09&include_classes=true";
    let (s, v) = call_raw(a.clone(), uri, body.clone()).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["classes"].as_array().unwrap().len(), 1200);
    assert!(v["areas"][0]["hectares"].as_f64().unwrap() > 0.0);
    // Without scene context the border can differ, but most pixels match the scene prediction.
    let scene = e.predict(&e.scene, win).unwrap();
    let same = v["classes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&scene.classes)
        .filter(|(a, b)| a.as_u64() == Some(**b as u64))
        .count();
    assert!(same as f64 / 1200.0 > 0.6, "{same}/1200");

    let (s, _) = call_raw(
        a.clone(),
        "/v1/predict/raster?width=40&height=31",
        body.clone(),
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _) = call_raw(
        a.clone(),
        "/v1/predict/raster?width=40&height=30&west=1",
        body.clone(),
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _) = call_raw(
        a.clone(),
        "/v1/predict/raster?width=4000&height=4000",
        vec![0; 16],
    )
    .await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    let (s, _) = call_raw(a, "/v1/predict/raster?width=abc&height=1", vec![]).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}
