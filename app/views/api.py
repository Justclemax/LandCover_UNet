"""How another application connects to the Rust inference API."""

import json

import ui
from ui import st

ui.hero("LandCover Lab · API",
        "Connecter une application",
        "Le modèle est servi par une API HTTP écrite en Rust. Toute application (web, mobile, SIG, script) peut "
        "l'appeler en JSON : envoyer une zone en latitude/longitude, ou ses propres images, et recevoir la carte "
        "d'occupation du sol.")

info, err = ui.api_model()
k = st.columns(3)
with k[0]:
    ui.kpi("Statut", "En ligne" if info else "Hors ligne", ui.API_URL)
with k[1]:
    ui.kpi("Moteur", "tract (Rust)" if info else "–", "ONNX exécuté sans Python ni C++")
with k[2]:
    ui.kpi("Authentification", ("Clé requise" if info["limits"]["auth"] else "Ouverte") if info else "–",
           "en-tête <code>x-api-key</code>")
if err:
    ui.box(f"<b>API injoignable :</b> {err}", alert=True)

st.markdown("#### Routes")
st.dataframe(
    [
        {"Méthode": "GET", "Route": "/health", "Rôle": "état du serveur (sans clé)"},
        {"Méthode": "GET", "Route": "/v1/model", "Rôle": "classes, couleurs, emprise de la scène, limites, métriques"},
        {"Méthode": "GET", "Route": "/v1/openapi.json", "Rôle": "description OpenAPI 3.1 (générateurs de clients)"},
        {"Méthode": "POST", "Route": "/v1/predict/bbox", "Rôle": "classer une zone [ouest, sud, est, nord] → PNG + surfaces"},
        {"Méthode": "POST", "Route": "/v1/predict/image", "Rôle": "classer vos propres pixels (8 canaux, ≤ 256 × 256)"},
    ],
    hide_index=True, width="stretch")

t1, t2, t3 = st.tabs(["curl", "Python", "JavaScript"])
key = ' -H "x-api-key: $API_KEY"'
with t1:
    st.code(f"""curl -s {ui.API_URL}/v1/predict/bbox{key} \\
  -H "content-type: application/json" \\
  -d '{{"bbox": [101.40, -2.20, 101.52, -2.08]}}' | jq '.areas[:3], .elapsed_ms'""", language="bash")
with t2:
    st.code(f"""import base64, requests

r = requests.post("{ui.API_URL}/v1/predict/bbox",
                  json={{"bbox": [101.40, -2.20, 101.52, -2.08]}},
                  headers={{"x-api-key": API_KEY}}, timeout=60)
r.raise_for_status()
res = r.json()
open("zone.png", "wb").write(base64.b64decode(res["png_base64"]))   # calque RGBA
for a in res["areas"]:
    print(a["label_fr"], a["hectares"], "ha")""", language="python")
with t3:
    st.code(f"""const res = await fetch("{ui.API_URL}/v1/predict/bbox", {{
  method: "POST",
  headers: {{ "content-type": "application/json", "x-api-key": API_KEY }},
  body: JSON.stringify({{ bbox: [101.40, -2.20, 101.52, -2.08] }}),
}}).then(r => r.json());
// Leaflet : L.imageOverlay("data:image/png;base64," + res.png_base64,
//   [[res.bounds[1], res.bounds[0]], [res.bounds[3], res.bounds[2]]]).addTo(map);""", language="javascript")

st.markdown("#### Erreurs")
st.markdown(
    "Toutes les erreurs ont la même forme : `{\"error\": {\"code\": \"…\", \"message\": \"…\"}}`.\n\n"
    "| Code HTTP | `code` | Cause |\n|---|---|---|\n"
    "| 400 | `bad_request` | JSON invalide ou champ inconnu |\n"
    "| 401 | `unauthorized` | clé absente ou fausse |\n"
    "| 413 | `too_large` | zone ou image trop grande |\n"
    "| 422 | `invalid_input` | bbox hors de la scène, valeurs incohérentes |\n"
    "| 503 | `busy` | trop d'inférences simultanées (en-tête `Retry-After`) |\n"
    "| 504 | `timeout` | calcul trop long |")

if info:
    with st.expander("Réponse actuelle de /v1/model"):
        st.code(json.dumps({k: v for k, v in info.items() if k != "classes"}, indent=2, ensure_ascii=False),
                language="json")

ui.footer()
