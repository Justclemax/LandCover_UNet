//! `landcover-serve`: loads models/rust/ and serves the API.
//!
//! Configuration (environment, or the project's `.env`):
//!   API_HOST (127.0.0.1)  API_PORT (8030)  LANDCOVER_BUNDLE_DIR (models/rust)
//!   API_KEY (none = open)  CORS_ORIGINS (none; "*" or a comma-separated list)
//!   MAX_CONCURRENT (2)  INFERENCE_THREADS (cores / MAX_CONCURRENT)  REQUEST_TIMEOUT_SECS (60)
//!   MAX_WINDOW_PIXELS (1200000)  MAX_RASTER_PIXELS (262144)  RUST_LOG (info)
//!
//! `landcover-serve healthcheck` exits 0 if /health answers (used by Docker).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use landcover_serve::engine::Engine;
use landcover_serve::{AppState, Limits, router};

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn num<T: std::str::FromStr>(name: &str, default: T) -> T {
    var(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn addr() -> String {
    format!(
        "{}:{}",
        var("API_HOST").unwrap_or_else(|| "127.0.0.1".into()),
        var("API_PORT").unwrap_or_else(|| "8030".into())
    )
}

/// Relative LANDCOVER_BUNDLE_DIR is resolved from the folder holding .env (the project root).
fn bundle_dir(env_file: Option<&Path>) -> PathBuf {
    let raw = PathBuf::from(var("LANDCOVER_BUNDLE_DIR").unwrap_or_else(|| "models/rust".into()));
    if raw.is_absolute() {
        return raw;
    }
    match env_file.and_then(Path::parent) {
        Some(root) => root.join(raw),
        None => raw,
    }
}

fn healthcheck() -> i32 {
    let host = var("API_HOST")
        .filter(|h| h != "0.0.0.0")
        .unwrap_or_else(|| "127.0.0.1".into());
    let target = format!(
        "{host}:{}",
        var("API_PORT").unwrap_or_else(|| "8030".into())
    );
    let ok = (|| -> std::io::Result<bool> {
        let mut s = TcpStream::connect(&target)?;
        s.set_read_timeout(Some(Duration::from_secs(3)))?;
        s.write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
        let mut buf = String::new();
        s.read_to_string(&mut buf)?;
        Ok(buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200"))
    })();
    if matches!(ok, Ok(true)) { 0 } else { 1 }
}

/// Ctrl-C locally, SIGTERM from Docker/Coolify: finish in-flight requests, then exit.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
    tracing::info!("arrêt demandé, fin des requêtes en cours");
}

#[tokio::main]
async fn main() {
    let env_file = dotenvy::dotenv()
        .ok()
        .or_else(|| dotenvy::from_filename("../.env").ok());
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        std::process::exit(healthcheck());
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .init();

    let max_concurrent: usize = num("MAX_CONCURRENT", 2);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let threads: usize = num("INFERENCE_THREADS", (cores / max_concurrent.max(1)).max(1));
    let dir = bundle_dir(env_file.as_deref());
    let t0 = Instant::now();
    let engine = Engine::load(&dir, threads).unwrap_or_else(|e| {
        eprintln!("Impossible de charger {} : {e}", dir.display());
        std::process::exit(1);
    });
    tracing::info!(
        dir = %dir.display(), classes = engine.bundle.classes.len(), threads, max_concurrent,
        load_ms = t0.elapsed().as_millis() as u64, "modèle chargé"
    );
    let limits = Limits {
        max_window_pixels: num("MAX_WINDOW_PIXELS", Limits::default().max_window_pixels),
        max_raster_pixels: num("MAX_RASTER_PIXELS", Limits::default().max_raster_pixels),
        request_timeout: Duration::from_secs(num("REQUEST_TIMEOUT_SECS", 60)),
        ..Limits::default()
    };
    let api_key = var("API_KEY");
    if api_key.is_none() {
        tracing::warn!("API_KEY non défini : les routes /v1/predict sont ouvertes");
    }
    let origins = var("CORS_ORIGINS").map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let app = router(
        AppState::new(engine, max_concurrent, api_key, limits),
        origins,
    );

    let addr = addr();
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Impossible d'écouter sur {addr} : {e}");
            std::process::exit(1);
        });
    tracing::info!("API sur http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("serveur");
}
