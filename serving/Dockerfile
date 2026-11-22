# Rust inference API. Build context: the repository root (see docker-compose.yml).
#
# 1. scene: rebuilds models/rust/scene.bin (240 MB, not in git) from the public rasters with
#    the core Python deps only (numpy + rasterio, no torch), and checks its SHA-256 against
#    the committed bundle.json, so the server can never run on a different scene.
# 2. build: compiles the server (dependency layer cached separately).
# 3. runtime: distroless, non-root, ~15 MB binary + model + scene.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS scene
WORKDIR /src
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
COPY data/lc_classes.json data/lc_classes.json
COPY models/rust/bundle.json models/rust/bundle.json
RUN uv sync --frozen --no-default-groups \
 && uv run --no-sync python -m landcover.export --scene-only \
 && rm -rf data/raw data/processed

FROM rust:1-slim-bookworm AS build
WORKDIR /build
COPY serving/Cargo.toml serving/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && touch src/lib.rs && cargo build --release --locked && rm -rf src
COPY serving/src ./src
COPY serving/openapi.json ./openapi.json
RUN touch src/main.rs src/lib.rs && cargo build --release --locked

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /build/target/release/landcover-serve /usr/local/bin/landcover-serve
COPY models/rust/bundle.json models/rust/model.onnx /models/
COPY --from=scene /src/models/rust/scene.bin /models/scene.bin
ENV LANDCOVER_BUNDLE_DIR=/models API_HOST=0.0.0.0 API_PORT=8030 MAX_CONCURRENT=2 RUST_LOG=info
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD ["/usr/local/bin/landcover-serve", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/landcover-serve"]
