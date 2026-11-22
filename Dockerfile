# Streamlit map app (no torch, no analysis: it reads reports/ and app/static/ and calls the API).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PORT=8910 LANDCOVER_ROOT=/app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-default-groups --group app --no-install-project
COPY .streamlit ./.streamlit
COPY app ./app
COPY reports/summary.json reports/areas.csv ./reports/
RUN useradd --create-home --uid 10001 lab && chown -R lab /app
USER lab
EXPOSE 8910
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"
CMD ["sh", "-c", "uv run --no-sync streamlit run app/Home.py --server.port=${PORT} --server.address=0.0.0.0"]
