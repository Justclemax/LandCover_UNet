"""Design system and helpers shared by the pages: theme tokens, banner, KPI cards, legend,
the Rust API client and the cached loaders. The app only reads reports/ and app/static/,
and asks the Rust API for new zones: no analysis runs in the app."""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get("LANDCOVER_ROOT", Path(__file__).resolve().parents[1]))
REPORTS = ROOT / "reports"
LAYERS_URL = "/app/static/layers"
API_URL = os.environ.get("LANDCOVER_API_URL", "http://127.0.0.1:8030").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")

TOKENS = {
    "light": {"fond": "#f6f6f9", "carte": "#ffffff", "carte2": "#f1f2f0", "bord": "#e3e4e0", "texte": "#15161a",
              "doux": "#5d5f66", "accent": "#1f7a4d", "accent_rgb": "31,122,77", "ocre": "#9a6400",
              "ocre_rgb": "184,120,10", "actif_bg": "#15161a", "actif_txt": "#ffffff", "header": "rgba(246,246,249,.85)",
              "tuiles": "World_Light_Gray_Base"},
    "dark": {"fond": "#0b0c0e", "carte": "#16171b", "carte2": "#1d1f24", "bord": "#26282e", "texte": "#ececf1",
             "doux": "#9a9ca8", "accent": "#6fd39a", "accent_rgb": "111,211,154", "ocre": "#f2d16b",
             "ocre_rgb": "242,209,107", "actif_bg": "#f4f4f6", "actif_txt": "#0b0c0e", "header": "rgba(11,12,14,.82)",
             "tuiles": "World_Dark_Gray_Base"},
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tokens() -> dict:
    return TOKENS[mode()]


def apply_style() -> None:
    t = tokens()
    st.html(f"""<style>
.stApp {{ background: radial-gradient(1100px 500px at 85% -140px, rgba({t['accent_rgb']},.14), transparent 60%), {t['fond']}; }}
.block-container {{ max-width: 1280px; padding-top: 4.4rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
[data-testid="stHeader"] {{ background: {t['header']}; backdrop-filter: blur(10px); }}
[data-testid="stToolbar"] > div:first-child {{ flex: 1 1 auto !important; }}
[data-testid="stTopNav"] {{ flex: 1 1 auto !important; justify-content: center; gap: 6px; }}
[data-testid="stTopNavLink"] {{ border-radius: 999px !important; padding: 4px 14px !important; background: {t['carte']}; border: 1px solid {t['bord']}; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: {t['actif_bg']} !important; }}
[data-testid="stTopNavLink"][aria-current="page"] p, [data-testid="stTopNavLink"][aria-current="page"] span {{ color: {t['actif_txt']} !important; font-weight: 700; }}
iframe[title="streamlit_folium.st_folium"] {{ border-radius: 22px; border: 1px solid {t['bord']}; }}
.lc-hero {{ border-radius: 24px; padding: 24px 30px; margin-bottom: 1rem; border: 1px solid {t['bord']};
           background: radial-gradient(90% 150% at 100% 0%, rgba({t['accent_rgb']},.26), transparent 58%),
                       radial-gradient(60% 120% at 0% 100%, rgba({t['ocre_rgb']},.08), transparent 60%), {t['carte']}; }}
.lc-hero .surtitre {{ color: {t['doux']}; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }}
.lc-hero h1 {{ color: {t['texte']}; font-size: 2.05rem; line-height: 1.12; margin: .4rem 0 .5rem; padding: 0; }}
.lc-hero p {{ color: {t['doux']}; font-size: 1rem; line-height: 1.55; max-width: 820px; margin: 0; }}
.kpi {{ background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 20px; padding: 16px 18px; min-height: 112px; }}
.kpi .etiquette {{ color: {t['doux']}; font-size: .82rem; font-weight: 600; }}
.kpi .valeur {{ font-size: 1.75rem; font-weight: 800; color: {t['texte']}; margin: .3rem 0 .2rem; line-height: 1.05; }}
.kpi .detail {{ color: {t['doux']}; font-size: .8rem; line-height: 1.4; }}
.legende {{ background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 22px; padding: 16px 18px; }}
.legende h4 {{ margin: 0 0 .6rem; padding: 0; font-size: 1rem; color: {t['texte']}; }}
.legende .ligne {{ display: flex; align-items: center; gap: 10px; padding: 5px 0; font-size: .86rem; color: {t['texte']};
                   border-bottom: 1px solid {t['bord']}; }}
.legende .ligne:last-child {{ border-bottom: none; }}
.legende .pastille {{ width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; box-shadow: inset 0 0 0 1px rgba(128,128,128,.35); }}
.legende .nom {{ flex: 1; line-height: 1.25; }}
.legende .val {{ color: {t['doux']}; font-variant-numeric: tabular-nums; white-space: nowrap; font-size: .8rem; }}
.note {{ color: {t['doux']}; font-size: .86rem; line-height: 1.5; margin: .4rem 0 .8rem; }}
.note b {{ color: {t['texte']}; }}
.encadre {{ border-radius: 16px; padding: 14px 18px; margin: .6rem 0 1rem; font-size: .93rem; line-height: 1.55; color: {t['texte']};
            border: 1px solid {t['bord']}; border-left: 4px solid {t['accent']};
            background: linear-gradient(90deg, rgba({t['accent_rgb']},.10), rgba({t['accent_rgb']},.02)); }}
.encadre.alerte {{ border-left-color: {t['ocre']}; background: linear-gradient(90deg, rgba({t['ocre_rgb']},.12), rgba({t['ocre_rgb']},.02)); }}
.pied {{ color: {t['doux']}; font-size: .78rem; margin-top: 2rem; padding-top: .9rem; border-top: 1px solid {t['bord']}; }}
@media (max-width: 700px) {{ .lc-hero h1 {{ font-size: 1.6rem; }} .lc-hero {{ padding: 20px; }} }}
</style>""")


def hero(kicker: str, title: str, text: str) -> None:
    st.html(f"<div class='lc-hero'><div class='surtitre'>{kicker}</div><h1>{title}</h1><p>{text}</p></div>")


def kpi(label: str, value: str, detail: str = "") -> None:
    st.html(f"<div class='kpi'><div class='etiquette'>{label}</div><div class='valeur'>{value}</div>"
            f"<div class='detail'>{detail}</div></div>")


def box(text: str, alert: bool = False) -> None:
    st.html(f"<div class='encadre{' alerte' if alert else ''}'>{text}</div>")


def legend(title: str, rows: list[tuple[str, str, str]]) -> None:
    """rows: (color, name, value)."""
    items = "".join(f"<div class='ligne'><span class='pastille' style='background:{c}'></span>"
                    f"<span class='nom'>{n}</span><span class='val'>{v}</span></div>" for c, n, v in rows)
    st.html(f"<div class='legende'><h4>{title}</h4>{items}</div>")


def footer() -> None:
    st.html("<div class='pied'>LandCover Lab · U-Net entraîné sur Landsat 8/9 et ALOS PALSAR-2 (2023), étiquettes de la "
            "carte d'occupation du sol du ministère indonésien de l'Environnement (KLHK). Données et idée d'origine : "
            "github.com/ramiqcom/deep-learning-lc-unet (MIT). Inférence : API Rust (tract). Fonds de carte : "
            "Esri (World Imagery, Canvas), © OpenStreetMap.</div>")


# ----------------------------------------------------------------------------- number formatting
def num(v: float, digits: int = 0) -> str:
    return f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")


def pct(v: float, digits: int = 0) -> str:
    return f"{num(v * 100, digits)} %"


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def summary() -> dict:
    return json.loads((REPORTS / "summary.json").read_text())


@st.cache_data(show_spinner=False)
def areas() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "areas.csv")


def bounds_latlon() -> list[list[float]]:
    w, s, e, n = summary()["scene"]["bounds"]
    return [[s, w], [n, e]]


# ----------------------------------------------------------------------------- map
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def base_map(basemap: str = "Plan", zoom: int = 10) -> folium.Map:
    """Folium map on the scene, with a theme-aware plan or Esri imagery as background."""
    (s, w), (n, e) = bounds_latlon()
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=zoom, tiles=None, control_scale=True,
                   prefer_canvas=True)
    if basemap == "Satellite (Esri)":
        folium.TileLayer(ESRI, attr="Esri World Imagery", name=basemap).add_to(m)
    else:
        canvas = f"https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{tokens()['tuiles']}/MapServer/tile/{{z}}/{{y}}/{{x}}"
        folium.TileLayer(canvas, attr="Esri, HERE, Garmin, © OpenStreetMap", name="Plan", max_native_zoom=16).add_to(m)
    folium.Rectangle(bounds_latlon(), color=tokens()["accent"], weight=1.5, dash_array="6 6", fill=False,
                     tooltip="Emprise de la scène Landsat").add_to(m)
    m.fit_bounds(bounds_latlon())
    return m


def overlay(m: folium.Map, name: str, url: str, bounds, opacity: float = 0.85) -> None:
    """Image overlay from a served URL or a data: URL (folium would read a "/..." path from disk)."""
    layer = folium.raster_layers.ImageOverlay(np.zeros((1, 1, 4), dtype=np.uint8), bounds=bounds, opacity=opacity,
                                              name=name, interactive=False, zindex=2, pixelated=True)
    layer.url = url
    layer.add_to(m)


# ----------------------------------------------------------------------------- Rust API client
def api(path: str, body: dict | None = None, timeout: float = 90) -> tuple[dict | None, str | None, float]:
    """(json, error message, round-trip ms). Never raises: the page shows the error."""
    data = json.dumps(body).encode() if body is not None else None
    return _call(path, data, "application/json", timeout)


def api_raw(path: str, body: bytes, timeout: float = 120) -> tuple[dict | None, str | None, float]:
    """POST raw bytes (float32 raster) to the API."""
    return _call(path, body, "application/octet-stream", timeout)


def _call(path: str, data: bytes | None, content_type: str, timeout: float) -> tuple[dict | None, str | None, float]:
    headers = {"content-type": content_type}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    req = urllib.request.Request(f"{API_URL}{path}", data=data, headers=headers, method="POST" if data else "GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (configured URL)
            return json.load(r), None, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e)["error"]["message"]
        except Exception:
            msg = f"HTTP {e.code}"
        return None, msg, (time.perf_counter() - t0) * 1000
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, f"API injoignable à {API_URL} ({getattr(e, 'reason', e)})", (time.perf_counter() - t0) * 1000


@st.cache_data(ttl=30, show_spinner=False)
def api_model() -> tuple[dict | None, str | None]:
    info, err, _ = api("/v1/model", timeout=5)
    return info, err


def png_data_url(b64: str) -> str:
    return "data:image/png;base64," + b64


def decode_png(b64: str) -> bytes:
    return base64.b64decode(b64)
