"""Draw a rectangle (or pick a place): the Rust API classifies it and the result is shown on the map."""

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

import ui
from ui import st

PLACES = {
    "Lac Kerinci": [101.40, -2.20, 101.52, -2.08],
    "Sungai Penuh (ville)": [101.34, -2.11, 101.44, -2.02],
    "Forêt de Kerinci Seblat": [101.14, -1.96, 101.26, -1.84],
}

ui.hero("LandCover Lab · Inférence à la demande",
        "Analyser une zone",
        "Dessinez un rectangle sur la carte, ou choisissez un lieu. La zone est envoyée à l'API d'inférence écrite en "
        "Rust, qui exécute le U-Net tuile par tuile et renvoie la carte et les surfaces.")

info, err = ui.api_model()
if err:
    ui.box(f"<b>API Rust indisponible.</b> {err}. Lancez-la avec <code>cd serving && cargo run --release</code>, "
           "puis rechargez la page.", alert=True)
    st.stop()

if "bbox" not in st.session_state:
    st.session_state.bbox = PLACES["Lac Kerinci"]

cols = st.columns([1, 1, 1, 1.4])
for col, (name, bbox) in zip(cols, PLACES.items(), strict=False):
    if col.button(name, icon=":material/place:", width="stretch"):
        st.session_state.bbox = bbox
        st.session_state.pop("result", None)
max_km2 = info["limits"]["max_window_pixels"] * 0.0009
cols[3].html(f"<div class='note'>Zone maximale : environ {ui.num(max_km2)} km² "
             f"({ui.num(info['limits']['max_window_pixels'])} pixels).</div>")

left, right = st.columns([1, 1], gap="medium")
with left:
    st.markdown("**1 · Choisir la zone**")
    m = ui.base_map("Satellite (Esri)")
    w, s, e, n = st.session_state.bbox
    folium.Rectangle([[s, w], [n, e]], color="#ffb020", weight=2, fill=True, fill_opacity=0.08,
                     tooltip="Zone sélectionnée").add_to(m)
    Draw(draw_options={"rectangle": {"shapeOptions": {"color": "#ffb020"}}, "polygon": False, "polyline": False,
                       "circle": False, "marker": False, "circlemarker": False},
         edit_options={"edit": False, "remove": False}).add_to(m)
    out = st_folium(m, height=520, use_container_width=True, returned_objects=["last_active_drawing"], key="dessin")
    drawn = (out or {}).get("last_active_drawing")
    if drawn and drawn.get("geometry", {}).get("type") == "Polygon":
        xs = [p[0] for p in drawn["geometry"]["coordinates"][0]]
        ys = [p[1] for p in drawn["geometry"]["coordinates"][0]]
        new = [round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5)]
        if new != st.session_state.bbox:
            st.session_state.bbox = new
            st.session_state.pop("result", None)
            st.rerun()
    w, s, e, n = st.session_state.bbox
    st.html(f"<div class='note'>Zone : ouest {w:.3f}°, sud {s:.3f}°, est {e:.3f}°, nord {n:.3f}°.</div>")
    if st.button("Classer cette zone avec l'API Rust", type="primary", icon=":material/bolt:", width="stretch"):
        with st.spinner("Inférence en cours…"):
            res, err, ms = ui.api("/v1/predict/bbox", {"bbox": st.session_state.bbox, "opacity": 0.9})
        st.session_state.result = (res, err, ms)

with right:
    st.markdown("**2 · Résultat**")
    if "result" not in st.session_state:
        st.html("<div class='note'>Le résultat s'affichera ici : carte classée, surfaces par classe et temps de "
                "calcul.</div>")
    else:
        res, err, ms = st.session_state.result
        if err:
            ui.box(f"<b>Erreur de l'API :</b> {err}", alert=True)
        else:
            w, s, e, n = res["bounds"]
            r = folium.Map(location=[(s + n) / 2, (w + e) / 2], tiles=None, control_scale=True)
            folium.TileLayer(ui.ESRI, attr="Esri World Imagery").add_to(r)
            ui.overlay(r, "U-Net (API Rust)", ui.png_data_url(res["png_base64"]), [[s, w], [n, e]], 0.85)
            r.fit_bounds([[s, w], [n, e]])
            st_folium(r, height=380, use_container_width=True, returned_objects=[], key=f"res_{w}_{s}_{e}_{n}")
            px = res["window"]["width"] * res["window"]["height"]
            k = st.columns(3)
            with k[0]:
                ui.kpi("Surface", f"{ui.num(sum(a['hectares'] for a in res['areas']) / 100, 1)} km²",
                       f"{ui.num(px)} pixels")
            with k[1]:
                ui.kpi("Calcul Rust", f"{ui.num(res['elapsed_ms'])} ms", f"aller-retour : {ui.num(ms)} ms")
            with k[2]:
                ui.kpi("Confiance", ui.pct(res["mean_confidence"]), "probabilité moyenne")
            ui.legend("Occupation du sol de la zone", [
                (a["color"], a["label_fr"], f"{ui.num(a['hectares'])} ha · {ui.pct(a['share'], 1)}")
                for a in res["areas"] if a["share"] >= 0.002])

ui.footer()
