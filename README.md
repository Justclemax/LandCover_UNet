# LandCover Lab · Kerinci

Cartographie de l'occupation du sol autour du parc national de Kerinci Seblat (Sumatra, Indonésie) par un **U-Net**, à partir des images **Landsat 8/9** et du radar **ALOS PALSAR-2** de 2023. Le modèle est servi par une **API d'inférence en Rust**, orchestré par un pipeline **Dagster** et présenté dans une **application Streamlit avec carte**.

Le projet reprend et étend [ramiqcom/deep-learning-lc-unet](https://github.com/ramiqcom/deep-learning-lc-unet) (MIT), avec les mêmes données publiques et la même architecture U-Net :

| | Projet d'origine | Ce projet |
|---|---|---|
| Entrées | 3 bandes Landsat | 6 bandes Landsat + radar HH/HV (8 canaux) |
| Découpage | patchs aléatoires qui se chevauchent entre entraînement et test | blocs spatiaux de 256 px, chacun dans un seul ensemble |
| Évaluation | précision sur des pixels voisins de l'entraînement | précision, mIoU, F1, kappa par classe sur des zones jamais vues, comparés à deux références |
| Inférence | `model.predict` dans le notebook | API Rust (tract), parité vérifiée avec Python |
| Restitution | figures matplotlib | carte interactive, analyse d'une zone à la demande |

## Architecture

```
rasters publics ──► ingestion ──► U-Net (PyTorch) ──► export ONNX ──► API Rust (axum + tract) ──► applications
 (Landsat, PALSAR,   alignement,    blocs spatiaux,      bundle +          /v1/predict/bbox            (Streamlit, SIG,
  carte KLHK)        recodage       sélection val        scène binaire     /v1/predict/image            scripts…)
                                          │
                                          └──► prédiction de la scène ──► couches de la carte (app/static/layers)
```

Chaque étape est une fonction appelée à la fois en ligne de commande et par un asset Dagster : le flux visuel et les commandes ne divergent jamais.

## Résultats

Score sur les blocs de test, comparé à la carte officielle KLHK :

| Modèle | Précision globale | mIoU | F1 macro | Kappa |
|---|---|---|---|---|
| Classe majoritaire | 0,502 | 0,039 | – | – |
| Forêt aléatoire par pixel (8 canaux) | 0,708 | 0,221 | 0,305 | 0,554 |
| **U-Net** (4,4 M paramètres) | **0,727** | **0,264** | **0,367** | **0,588** |

- Le U-Net bat la forêt aléatoire sur tous les indicateurs, grâce au contexte spatial (patchs de 256 pixels).
- La forêt primaire (IoU 0,87), les rizières (0,56) et l'agriculture mixte (0,52) sont bien reconnues. La part de forêt prédite (60,6 %) est proche de celle de la référence (59,5 %).
- Sol nu, forêt de plantation et marais ne sont pas retrouvés : ils couvrent moins de 0,5 % de la scène.
- Un premier essai (patchs de 128, perte plus fortement pondérée) sur-prédisait les classes rares : précision 0,665, mIoU 0,244.
- La carte KLHK est dessinée à grande échelle, avec des polygones lissés : il s'agit d'un **accord avec la référence**, pas d'une vérité terrain au pixel près.

## Inférence en Rust

- **100 % Rust**, sans `unsafe` : le graphe ONNX est exécuté par [tract](https://github.com/sonos/tract), sans Python ni bibliothèque C++.
- **Mémoire** : la scène (237 Mo) n'est jamais chargée en entier. Chaque requête lit seulement les lignes dont elle a besoin (lectures positionnées, sans curseur partagé). L'API occupe environ 160 Mo au repos, plus environ 75 Mo par fil d'inférence actif, et le total est plafonné par `MAX_CONCURRENT × INFERENCE_THREADS`.
- **Parité** : `cargo test` vérifie que Rust donne les mêmes classes que Python (onnxruntime) sur 4 fenêtres témoins : 100 % de pixels identiques, écart de probabilité inférieur à 10⁻⁶.
- **Performance** : la zone du lac Kerinci (≈ 180 km², 200 000 pixels) est classée en 0,5 s avec 4 fils. Le binaire fait 13 Mo.

### API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du serveur |
| GET | `/v1/model` | classes, couleurs, emprise, limites, métriques |
| GET | `/v1/openapi.json` | description OpenAPI 3.1 |
| POST | `/v1/predict/bbox` | classer une zone `[ouest, sud, est, nord]` : calque PNG, surfaces par classe |
| POST | `/v1/predict/image` | classer ses propres pixels en JSON (8 canaux, ≤ 256 × 256) |
| POST | `/v1/predict/raster` | la même chose en binaire float32 (≤ 512 × 512), surfaces en hectares si la zone est donnée |

```bash
curl -s localhost:8030/v1/predict/bbox -H 'content-type: application/json' \
  -d '{"bbox": [101.40, -2.20, 101.52, -2.08]}' | jq '.areas[:3], .elapsed_ms'
```

Le serveur applique les protections suivantes :
- une clé d'API optionnelle (`x-api-key`, comparaison en temps constant) ;
- un JSON strict, qui refuse les champs inconnus ;
- des limites de taille de zone, d'image et de corps de requête ;
- une file bornée, qui renvoie 503 avec `Retry-After` quand le serveur est saturé ;
- un délai maximal, au-delà duquel il renvoie 504 ;
- des erreurs JSON uniformes ;
- un arrêt propre sur SIGTERM.

## Partout dans le monde

La page « Partout dans le monde » classe un rectangle dessiné n'importe où sur la Terre, en trois étapes :

1. Elle télécharge depuis [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) (gratuit, sans compte) les images Landsat 8/9 de l'année choisie et la mosaïque radar ALOS PALSAR-2, disponible de 2015 à 2021 (la plus récente est utilisée).
2. Elle masque les nuages et calcule la médiane des observations claires, sur la même grille de 30 m que l'entraînement.
3. Elle envoie l'image à l'API Rust (`/v1/predict/raster`).

Pour valider la chaîne, une zone de Kerinci a été traitée des deux façons :
- **Optique** : le composite Planetary Computer reproduit la scène d'entraînement (exportée depuis Google Earth Engine). Les médianes diffèrent de moins de 5 % et la corrélation par pixel est de 0,87 à 0,95.
- **Radar** : la correspondance est moins bonne (2021 contre 2023), avec une corrélation HH de 0,24.
- **Classes** : 72 % sont identiques entre les deux sources. L'accord avec la carte KLHK est un peu meilleur avec Planetary Computer (0,63 contre 0,59).

**Limite :** le modèle n'a vu qu'une région de Sumatra. Il reste crédible dans les paysages tropicaux humides semblables. Ailleurs, il plaque ses 14 classes indonésiennes sur un paysage inconnu, et l'application l'indique.

## Démarrage

```bash
uv sync                                                    # Python 3.13
cp .env.example .env
uv run python -W ignore -m landcover.ingest                # télécharge 215 Mo et aligne les rasters
uv run python -W ignore -m landcover.train                 # ou --fast pour un essai de 2 min
uv run python -W ignore -m landcover.export                # ONNX + bundle pour Rust
uv run python -W ignore -m landcover.predict               # couches de la carte
cd serving && cargo test --release && cargo run --release  # API Rust
uv run --env-file .env streamlit run app/Home.py           # application
DAGSTER_HOME=$PWD/.dagster uv run dagster dev -m landcover.pipeline -p 3030   # flux visuel
```

| Variable | Rôle | Défaut |
|---|---|---|
| `STREAMLIT_SERVER_PORT` | port de l'application | 8501 |
| `API_HOST`, `API_PORT` | adresse de l'API Rust | 127.0.0.1, 8030 |
| `LANDCOVER_BUNDLE_DIR` | dossier du bundle | `models/rust` |
| `API_KEY` | clé exigée sur `/v1/predict/*` (vide = ouverte) | vide |
| `CORS_ORIGINS` | origines web autorisées (`*` ou liste) | aucune |
| `MAX_CONCURRENT`, `INFERENCE_THREADS` | inférences simultanées, fils par inférence | 2, cœurs / 2 |
| `LANDCOVER_API_URL` | API utilisée par l'application | `http://127.0.0.1:8030` |

## Déploiement (Coolify)

- Build pack Docker Compose sur `docker-compose.yml`, sans port publié.
- Le domaine se configure sur le service **app**, sous la forme `https://domaine:8910`.
- L'image de l'API reconstruit la scène binaire à partir des données publiques et vérifie son empreinte par canal (valeurs manquantes, moyenne, écart-type) par rapport au bundle versionné. Ce n'est pas un SHA-256, parce que les calculs vectorisés de numpy diffèrent au dernier bit d'un processeur à l'autre.
- Si l'API reçoit un domaine public pour d'autres applications, définissez `API_KEY` dans Coolify.

## Données et licences

- Images, carte de référence et idée d'origine : [ramiqcom/deep-learning-lc-unet](https://github.com/ramiqcom/deep-learning-lc-unet) (MIT), fichiers publics sur Google Cloud Storage. Landsat : USGS/NASA. PALSAR-2 : JAXA. Carte d'occupation du sol : KLHK (ministère indonésien de l'Environnement et des Forêts).
- Fonds de carte : Esri (World Imagery, Canvas), © contributeurs OpenStreetMap.
