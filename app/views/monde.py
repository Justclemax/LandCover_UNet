"""Anywhere on Earth: fetch Landsat + PALSAR for a drawn box, classify it with the Rust API."""

import base64
import math
import time

import folium
import imagery
import numpy as np
from folium.plugins import Draw
from streamlit_folium import st_folium

import ui
from ui import st

PLACES = {
    "Kerinci (zone d'entraînement)": [101.33, -2.10, 101.43, -2.00],
    "Palangka Raya (Bornéo)": [113.85, -2.25, 113.97, -2.13],
    "Riau, palmeraies (Sumatra)": [101.40, 0.45, 101.52, 0.57],
    "Delta du Mékong (Vietnam)": [105.70, 10.00, 105.82, 10.12],
    "Rondônia (Amazonie)": [-62.95, -10.05, -62.83, -9.93],
    "Beauce (France)": [1.55, 48.20, 1.67, 48.32],
}
MAX_SIDE = 512  # pixels, about 15 km at 30 m

ui.hero("LandCover Lab · Partout dans le monde",
        "Classer n'importe quelle zone",
        "Dessinez un rectangle n'importe où sur la Terre. L'application télécharge les images Landsat de l'année "
        "choisie et le radar PALSAR (Microsoft Planetary Computer), fabrique une image sans nuages, puis l'API Rust "
        "exécute le U-Net.")
ui.box("<b>À lire avant d'interpréter :</b> le modèle a appris sur une seule région de Sumatra, avec la légende du "
       "ministère indonésien. Il est crédible dans les paysages tropicaux humides semblables (Indonésie, Malaisie, "
       "Asie du Sud-Est). Ailleurs, il plaque ses 14 classes indonésiennes sur un paysage qu'il n'a jamais vu : "
       "la carte n'a pas de valeur cartographique, elle montre seulement comment le modèle généralise.", alert=True)

info, err = ui.api_model()
if err:
    ui.box(f"<b>API Rust indisponible.</b> {err}.", alert=True)
    st.stop()
res_deg = info["scene"]["resolution_deg"]


def clamp(bbox: list[float]) -> tuple[list[float], bool]:
    """Shrink the box around its centre to at most MAX_SIDE x MAX_SIDE pixels."""
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    half = MAX_SIDE * res_deg / 2
    hw, hh = min((e - w) / 2, half), min((n - s) / 2, half)
    out = [round(cx - hw, 5), round(cy - hh, 5), round(cx + hw, 5), round(cy + hh, 5)]
    return out, (e - w) / 2 > half or (n - s) / 2 > half


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def composite(bbox: tuple[float, ...], year: int) -> imagery.Composite:
    return imagery.build(list(bbox), res_deg, year=year)


if "monde_bbox" not in st.session_state:
    st.session_state.monde_bbox = PLACES["Palangka Raya (Bornéo)"]

top = st.columns([2.2, 1])
place = top[0].selectbox("Lieux d'exemple", ["—"] + list(PLACES), index=0)
if place != "—" and PLACES[place] != st.session_state.monde_bbox and st.session_state.get("monde_place") != place:
    st.session_state.monde_bbox, st.session_state.monde_place = PLACES[place], place
    st.session_state.pop("monde_result", None)
year = top[1].select_slider("Année des images Landsat", list(range(2015, 2026)), value=2023)

left, right = st.columns([1, 1], gap="medium")
with left:
    st.markdown("**1 · Choisir la zone** (outil carré à gauche de la carte)")
    w, s, e, n = st.session_state.monde_bbox
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=9, tiles=None, control_scale=True)
    folium.TileLayer(ui.ESRI, attr="Esri World Imagery").add_to(m)
    (ks, kw), (kn, ke) = ui.bounds_latlon()
    folium.Rectangle([[ks, kw], [kn, ke]], color=ui.tokens()["accent"], weight=1.5, dash_array="6 6", fill=False,
                     tooltip="Zone d'entraînement (Kerinci)").add_to(m)
    folium.Rectangle([[s, w], [n, e]], color="#ffb020", weight=2, fill=True, fill_opacity=0.08).add_to(m)
    Draw(draw_options={"rectangle": {"shapeOptions": {"color": "#ffb020"}}, "polygon": False, "polyline": False,
                       "circle": False, "marker": False, "circlemarker": False},
         edit_options={"edit": False, "remove": False}).add_to(m)
    out = st_folium(m, height=500, use_container_width=True, returned_objects=["last_active_drawing"],
                    key=f"monde_{w}_{s}")
    drawn = (out or {}).get("last_active_drawing")
    if drawn and drawn.get("geometry", {}).get("type") == "Polygon":
        xs = [p[0] for p in drawn["geometry"]["coordinates"][0]]
        ys = [p[1] for p in drawn["geometry"]["coordinates"][0]]
        new, shrunk = clamp([min(xs), min(ys), max(xs), max(ys)])
        if new != st.session_state.monde_bbox:
            st.session_state.monde_bbox, st.session_state.monde_shrunk = new, shrunk
            st.session_state.pop("monde_result", None)
            st.rerun()
    w, s, e, n = st.session_state.monde_bbox
    km = lambda d, lat=0.0: d * 111.32 * math.cos(math.radians(lat))  # noqa: E731
    st.html(f"<div class='note'>Zone : {km(e - w, (s + n) / 2):.1f} × {km(n - s):.1f} km, centre "
            f"{(s + n) / 2:.3f}°, {(w + e) / 2:.3f}°. Maximum {MAX_SIDE} × {MAX_SIDE} pixels (≈ 15 km de côté)"
            f"{' : le rectangle dessiné a été réduit autour de son centre' if st.session_state.get('monde_shrunk') else ''}.</div>")
    if st.button("Télécharger les images et classer", type="primary", icon=":material/satellite_alt:",
                 width="stretch"):
        with st.status("Traitement de la zone…", expanded=True) as status:
            st.write(f"Recherche des images Landsat {year} et du radar PALSAR {imagery.PALSAR_YEAR}…")
            t0 = time.perf_counter()
            try:
                c = composite(tuple(st.session_state.monde_bbox), year)
            except Exception as exc:  # network, catalogue or raster errors: show, do not crash
                status.update(label="Téléchargement impossible", state="error")
                st.session_state.monde_result = {"error": f"Images indisponibles : {exc}"}
                st.rerun()
            t_fetch = (time.perf_counter() - t0) * 1000
            st.write(f"{c.scenes} scènes Landsat utilisées, {ui.pct(c.clear_share)} des pixels sans nuages.")
            st.write("Inférence par l'API Rust…")
            H, W = c.x.shape[1:]
            body = np.ascontiguousarray(c.x.transpose(1, 2, 0)).astype("<f4").tobytes()
            bw, bs, be, bn = c.bbox
            res, err, ms = ui.api_raw(f"/v1/predict/raster?width={W}&height={H}&west={bw}&south={bs}&east={be}"
                                      f"&north={bn}&opacity=0.9", body)
            status.update(label="Terminé" if res else "Erreur de l'API", state="complete" if res else "error")
        st.session_state.monde_result = {"composite": c, "api": res, "error": err, "ms": ms, "fetch_ms": t_fetch}

with right:
    st.markdown("**2 · Résultat**")
    r = st.session_state.get("monde_result")
    if not r:
        st.html("<div class='note'>Choisissez un lieu ou dessinez un rectangle, puis lancez le traitement "
                "(5 à 30 secondes selon le nombre d'images).</div>")
    elif r.get("error"):
        ui.box(f"<b>Erreur :</b> {r['error']}", alert=True)
    else:
        c, res = r["composite"], r["api"]
        w, s, e, n = c.bbox
        lat = (s + n) / 2
        if abs(lat) > 23.5:
            ui.box(f"<b>Hors des tropiques ({lat:.1f}°)</b> : paysage très différent de Sumatra, les classes "
                   "affichées ne sont pas fiables.", alert=True)
        if c.clear_share < 0.6:
            ui.box(f"Seulement {ui.pct(c.clear_share)} des pixels ont au moins une image sans nuages : les zones "
                   "transparentes n'ont pas pu être classées.", alert=True)
        if c.palsar_year is None:
            ui.box("Pas de radar PALSAR sur cette zone : le modèle travaille sans ces deux canaux.", alert=True)
        layer = st.segmented_control("Afficher", ["Classes (U-Net)", "Image composite"], default="Classes (U-Net)") \
            or "Classes (U-Net)"
        mm = folium.Map(location=[lat, (w + e) / 2], tiles=None, control_scale=True)
        folium.TileLayer(ui.ESRI, attr="Esri World Imagery").add_to(mm)
        if layer == "Image composite":
            url = "data:image/png;base64," + base64.b64encode(imagery.rgb_png(c.x)).decode()
            ui.overlay(mm, "Composite Landsat", url, [[s, w], [n, e]], 1.0)
        else:
            ui.overlay(mm, "U-Net", ui.png_data_url(res["png_base64"]), [[s, w], [n, e]], 0.9)
        mm.fit_bounds([[s, w], [n, e]])
        st_folium(mm, height=360, use_container_width=True, returned_objects=[], key=f"mres_{w}_{s}_{layer}")
        k = st.columns(3)
        with k[0]:
            ui.kpi("Images", f"{c.scenes} scènes", f"Landsat {year}, {c.observations:.0f} observations claires "
                   "par pixel en moyenne")
        with k[1]:
            ui.kpi("Temps", f"{ui.num(r['fetch_ms'] / 1000, 1)} s", f"téléchargement ; inférence Rust "
                   f"{ui.num(res['elapsed_ms'])} ms")
        with k[2]:
            ui.kpi("Confiance", ui.pct(res["mean_confidence"]), "probabilité moyenne de la classe choisie")
        ui.legend("Classes prédites", [(a["color"], a["label_fr"], f"{ui.num(a['hectares'])} ha · "
                                        f"{ui.pct(a['share'], 1)}") for a in res["areas"] if a["share"] >= 0.002])

ui.footer()
