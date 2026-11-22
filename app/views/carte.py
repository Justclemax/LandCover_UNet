"""Main page: the U-Net land cover map of the whole scene, with its legend."""

from streamlit_folium import st_folium

import ui
from ui import st

S = ui.summary()
A = ui.areas()
LAYERS = {
    "U-Net": ("prediction.png", "occupation du sol prédite par le U-Net sur toute la scène."),
    "Référence": ("reference.png", "carte officielle du ministère indonésien (KLHK) utilisée pour l'apprentissage."),
    "Désaccords": ("disagreement.png", "en magenta, les pixels où le U-Net et la carte officielle diffèrent."),
    "Incertitude": ("confidence.png", "plus l'orange est soutenu, moins le modèle est sûr de lui."),
    "Satellite 2023": ("satellite.jpg", "composition en couleurs naturelles Landsat 2023."),
}

ui.hero("LandCover Lab · Kerinci, Sumatra",
        "L'occupation du sol vue par un U-Net",
        f"Chaque pixel de 30 m d'une scène de {ui.num(S['scene']['area_km2'])} km² autour du parc national de "
        "Kerinci Seblat, classé par un réseau de neurones à partir des images Landsat et radar de 2023. "
        "Le calcul se fait en amont ; la carte affiche le résultat.")

k = st.columns(4)
with k[0]:
    ui.kpi("Surface cartographiée", f"{ui.num(S['scene']['area_km2'])} km²", f"{ui.num(S['scene']['width'])} × "
           f"{ui.num(S['scene']['height'])} pixels de 30 m")
with k[1]:
    ui.kpi("Forêt", ui.pct(S["forest_share_pred"]), f"selon le U-Net, contre {ui.pct(S['forest_share_ref'])} "
           "sur la carte officielle")
with k[2]:
    ui.kpi("Fiabilité", ui.pct(S["test"]["test_overall_accuracy"]), "des pixels bien classés sur des zones jamais "
           "vues à l'entraînement")
with k[3]:
    ui.kpi("Confiance moyenne", ui.pct(S["mean_confidence"]), "probabilité moyenne de la classe choisie")

c1, c2, c3 = st.columns([2.4, 1.1, 1.1])
layer = c1.segmented_control("Couche", list(LAYERS), default="U-Net") or "U-Net"
opacity = c2.slider("Opacité", 0.2, 1.0, 0.8, 0.05)
basemap = c3.radio("Fond", ["Plan", "Satellite (Esri)"], horizontal=True)

file, note = LAYERS[layer]
left, right = st.columns([3, 1.15], gap="medium")
with left:
    m = ui.base_map(basemap)
    ui.overlay(m, layer, f"{ui.LAYERS_URL}/{file}", ui.bounds_latlon(), opacity)
    st_folium(m, height=640, use_container_width=True, returned_objects=[], key=f"carte_{layer}_{basemap}")
    st.html(f"<div class='note'>👁️ <b>{layer} :</b> {note} Zoomez pour voir le détail ; un pixel fait 30 m.</div>")
with right:
    if layer in ("U-Net", "Référence"):
        col = "pred" if layer == "U-Net" else "ref"
        d = A[A[f"{col}_ha"] > 0].sort_values(f"{col}_ha", ascending=False)
        ui.legend("Légende · surface", [(r.color, r.label_fr, f"{ui.num(r[f'{col}_ha'] / 100)} km² · "
                                         f"{ui.pct(r[f'{col}_share'], 1)}") for _, r in d.iterrows()])
    elif layer == "Désaccords":
        ui.legend("Légende", [("#d628a0", "U-Net ≠ carte officielle", ui.pct(1 - S["agreement_all_pixels"])),
                              ("transparent", "Accord", ui.pct(S["agreement_all_pixels"]))])
        ui.box("La carte officielle est dessinée à grande échelle, avec des polygones lissés : une partie des "
               "désaccords se trouve sur les limites entre classes, où la référence elle-même est approximative.")
    elif layer == "Incertitude":
        ui.legend("Légende", [("#ffb020", "Modèle hésitant (probabilité < 50 %)", ""),
                              ("rgba(255,176,32,.35)", "Modèle assez sûr", ""), ("transparent", "Modèle très sûr", "")])
    else:
        ui.legend("Image", [("#3d6b35", "Végétation", "vert"), ("#b9a48a", "Sol nu, bâti", "beige"),
                            ("#1b2a3a", "Eau", "sombre")])

ui.footer()
