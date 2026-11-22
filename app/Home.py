"""LandCover Lab: land cover of Kerinci (Sumatra) on a map. Entry point and navigation.

    uv run --env-file .env streamlit run app/Home.py
"""

import streamlit as st

st.set_page_config(page_title="LandCover Lab · Kerinci", page_icon="🛰️", layout="wide")

import ui  # noqa: E402  (after set_page_config)

ui.apply_style()
nav = st.navigation(
    [
        st.Page("views/carte.py", title="Carte", icon=":material/map:", default=True),
        st.Page("views/zone.py", title="Analyser une zone", icon=":material/crop_free:"),
        st.Page("views/api.py", title="Connecter une application", icon=":material/api:"),
    ],
    position="top",
)
nav.run()
