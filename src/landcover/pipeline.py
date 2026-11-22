"""Dagster pipeline: the visual flow from data ingestion to the Rust API and the map app.

    uv run dagster dev -m landcover.pipeline -p 3030      # UI on http://127.0.0.1:3030

Asset graph (groups in brackets):

    [ingestion]  rasters_publics -> scene_alignee
    [modele]     scene_alignee -> unet_entraine
    [service]    unet_entraine -> bundle_rust          (check: parite_rust runs `cargo test`)
    [carte]      unet_entraine -> couches_carte        (check: couches_completes)
    [service]    bundle_rust -> api_en_ligne           (observes the running Rust API)

Each step calls the same functions as the command-line modules, so the pipeline and the
CLI never diverge.
"""

import json
import os
import subprocess
import urllib.request

import dagster as dg

from landcover import config as cfg


class TrainConfig(dg.Config):
    fast: bool = False
    epochs: int = 40
    patch: int = 256


@dg.asset(group_name="ingestion", kinds={"python", "geotiff"},
          description="Télécharge les rasters publics : Landsat 2023, PALSAR-2 2023, carte KLHK.")
def rasters_publics() -> dg.MaterializeResult:
    from landcover.ingest import download

    paths = download()
    return dg.MaterializeResult(metadata={
        name: dg.MetadataValue.text(f"{p.name} ({p.stat().st_size / 1e6:.0f} Mo)") for name, p in paths.items()
    } | {"source": dg.MetadataValue.url(cfg.BASE_URL)})


@dg.asset(deps=[rasters_publics], group_name="ingestion", kinds={"numpy"},
          description="Aligne le radar sur la grille Landsat (30 m) et recode les classes 1..K.")
def scene_alignee() -> dg.MaterializeResult:
    from landcover.ingest import build

    s = build()
    return dg.MaterializeResult(metadata={
        "largeur": s.meta["width"], "hauteur": s.meta["height"], "canaux": ", ".join(s.meta["channels"]),
        "classes": len(s.classes), "pixels_etiquetes": dg.MetadataValue.float(s.meta["labelled_share"]),
        "classes_ignorees": ", ".join(c["label_fr"] for c in s.meta["ignored_classes"]) or "aucune",
    })


@dg.asset(deps=[scene_alignee], group_name="modele", kinds={"pytorch"},
          description="Entraîne le U-Net sur les blocs d'entraînement, sélection sur la validation, score sur le test.")
def unet_entraine(config: TrainConfig) -> dg.MaterializeResult:
    from landcover.train import train

    m = train(fast=config.fast, epochs=config.epochs, patch=config.patch)
    s = m["summary"]
    return dg.MaterializeResult(metadata={
        "test_precision_globale": dg.MetadataValue.float(s["test_overall_accuracy"]),
        "test_miou": dg.MetadataValue.float(s["test_mean_iou"]),
        "test_kappa": dg.MetadataValue.float(s["test_kappa"]),
        "foret_aleatoire_miou": dg.MetadataValue.float(s["rf_test_mean_iou"]),
        "parametres": m["model"]["parameters"], "epoques": m["metadata"]["epochs_run"],
        "rapide": m["metadata"]["fast"],
    })


@dg.asset_check(asset=unet_entraine, description="Le U-Net doit battre la forêt aléatoire par pixel (mIoU test).")
def bat_la_reference() -> dg.AssetCheckResult:
    s = json.loads((cfg.MODELS_DIR / "metrics.json").read_text())["summary"]
    return dg.AssetCheckResult(passed=s["test_mean_iou"] > s["rf_test_mean_iou"],
                               metadata={"unet": s["test_mean_iou"], "foret_aleatoire": s["rf_test_mean_iou"]})


@dg.asset(deps=[unet_entraine], group_name="service", kinds={"onnx", "rust"},
          description="Exporte le U-Net en ONNX et la scène en binaire pour le serveur Rust (models/rust/).")
def bundle_rust() -> dg.MaterializeResult:
    from landcover.export import run

    b = run()
    return dg.MaterializeResult(metadata={"sha256_modele": b["model"]["sha256"][:16], "classes": len(b["classes"]),
                                          "dossier": dg.MetadataValue.path(str(cfg.RUST_DIR))})


@dg.asset_check(asset=bundle_rust, description="cargo test : Rust (tract) reproduit Python (onnxruntime) pixel pour pixel.")
def parite_rust() -> dg.AssetCheckResult:
    r = subprocess.run(["cargo", "test", "--release", "--", "--nocapture"], cwd=cfg.ROOT / "serving",
                       capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in r.stdout.splitlines() if "accord" in ln or ln.startswith("test result")]
    return dg.AssetCheckResult(passed=r.returncode == 0, metadata={"sortie": dg.MetadataValue.md("\n".join(
        f"- `{ln}`" for ln in lines) or r.stderr[-2000:])})


@dg.asset(deps=[unet_entraine], group_name="carte", kinds={"python", "png"},
          description="Prédit toute la scène et écrit les couches de la carte et les surfaces par classe.")
def couches_carte() -> dg.MaterializeResult:
    from landcover.predict import run

    s = run()
    return dg.MaterializeResult(metadata={
        "foret_predite": dg.MetadataValue.float(s["forest_share_pred"]),
        "foret_reference": dg.MetadataValue.float(s["forest_share_ref"]),
        "accord_global": dg.MetadataValue.float(s["agreement_all_pixels"]),
        "couches": ", ".join(s["layers"]),
    })


@dg.asset_check(asset=couches_carte, description="Toutes les couches attendues par l'application existent.")
def couches_completes() -> dg.AssetCheckResult:
    expected = {"satellite.jpg", "prediction.png", "reference.png", "disagreement.png", "confidence.png"}
    have = {p.name for p in (cfg.ROOT / "app" / "static" / "layers").glob("*")}
    return dg.AssetCheckResult(passed=expected <= have, metadata={"manquantes": ", ".join(sorted(expected - have)) or "aucune"})


@dg.asset(deps=[bundle_rust], group_name="service", kinds={"api"},
          description="Interroge l'API Rust en cours d'exécution (LANDCOVER_API_URL) et vérifie qu'elle sert le modèle exporté.")
def api_en_ligne() -> dg.MaterializeResult:
    url = os.environ.get("LANDCOVER_API_URL", "http://127.0.0.1:8030")
    with urllib.request.urlopen(f"{url}/v1/model", timeout=10) as r:  # noqa: S310
        info = json.load(r)
    bundle = json.loads((cfg.RUST_DIR / "bundle.json").read_text())
    if info["model"]["sha256"] != bundle["model"]["sha256"]:
        raise dg.Failure("L'API sert un autre modèle que le bundle exporté : redémarrez-la.")
    return dg.MaterializeResult(metadata={"url": dg.MetadataValue.url(url), "runtime": info["model"]["runtime"]})


pipeline_complet = dg.define_asset_job(
    "pipeline_complet", selection=dg.AssetSelection.assets(rasters_publics, scene_alignee, unet_entraine,
                                                           bundle_rust, couches_carte),
    description="Ingestion -> entraînement -> export Rust -> couches de la carte.")

defs = dg.Definitions(
    assets=[rasters_publics, scene_alignee, unet_entraine, bundle_rust, couches_carte, api_en_ligne],
    asset_checks=[bat_la_reference, parite_rust, couches_completes],
    jobs=[pipeline_complet],
)
