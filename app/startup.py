"""One-time model initialisation for cloud deployments (Streamlit Cloud / HF Spaces).

Called by the Streamlit app at startup when no trained models are found.
Runs a reduced pipeline so the app becomes interactive quickly:
  - 20 Newsgroups : downloaded automatically via sklearn (always available)
  - Wikipedia     : only trained if people_wiki.csv is present locally
"""

import logging
import pickle
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

_CONFIG_PATH   = _ROOT / "configs" / "config.yaml"
_MODELS_DIR    = _ROOT / "models"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_RAW_DIR       = _ROOT / "data" / "raw"
_REPORTS_DIR   = _ROOT / "outputs" / "reports"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _newsgroups_ready() -> bool:
    return (
        (_MODELS_DIR / "tfidf_newsgroups.pkl").exists()
        and (_REPORTS_DIR / "clustering_metrics_newsgroups.csv").exists()
    )


def _wikipedia_ready() -> bool:
    return (
        (_MODELS_DIR / "tfidf_wikipedia.pkl").exists()
        and (_REPORTS_DIR / "clustering_metrics_wikipedia.csv").exists()
    )


def _wikipedia_csv_present() -> bool:
    """Return True if the Kaggle people_wiki.csv is available."""
    try:
        cfg = _load_config()
        return (_RAW_DIR / cfg["datasets"]["wikipedia"]["filename"]).exists()
    except Exception:
        return False


def needs_setup() -> bool:
    """Return True if newsgroups models/metrics are missing.

    Newsgroups is always auto-trainable via sklearn.
    Wikipedia requires people_wiki.csv so it is optional.
    """
    return not _newsgroups_ready()


def run_setup(status_callback=None) -> None:
    """Train newsgroups always; train Wikipedia only if CSV is present."""
    def _status(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    config = _load_config()

    fast_config = dict(config)
    fast_config["clustering"] = {
        "kmeans":       {"k_range": [5, 10, 20], "n_init": 5, "max_iter": 200},
        "hierarchical": {"n_clusters": [5, 10], "linkage": ["ward"], "max_docs": 5000},
        "gmm":          {"n_components": [5, 10], "covariance_type": ["tied"], "max_iter": 50},
    }

    # Newsgroups — always train (sklearn downloads it automatically)
    if not _newsgroups_ready():
        try:
            _run_dataset("newsgroups", fast_config, _status)
        except Exception as exc:
            logger.exception("Newsgroups pipeline failed: %s", exc)
            _status(f"[newsgroups] ERROR: {exc}")

    # Wikipedia — only if CSV exists locally
    if not _wikipedia_ready():
        if _wikipedia_csv_present():
            try:
                _run_dataset("wikipedia", fast_config, _status)
            except Exception as exc:
                logger.exception("Wikipedia pipeline failed: %s", exc)
                _status(f"[wikipedia] ERROR: {exc}")
        else:
            _status("[wikipedia] people_wiki.csv not found — skipping Wikipedia training.")

    _status("Setup complete!")


def _run_dataset(dataset: str, config: dict, status) -> None:
    from src.data_loader import load_newsgroups, load_wikipedia_people
    from src.preprocessing import preprocess_corpus
    from src.features import build_tfidf, reduce_with_svd, save_feature_artifacts
    from src.clustering import run_all_experiments, save_cluster_results
    from src.evaluation import evaluate_all_experiments, save_metrics

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    status(f"[{dataset}] Loading data…")
    if dataset == "newsgroups":
        cfg = config["datasets"]["newsgroups"]
        texts, _, _ = load_newsgroups(subset=cfg["subset"], remove=tuple(cfg["remove"]))
    else:
        cfg      = config["datasets"]["wikipedia"]
        filepath = _RAW_DIR / cfg["filename"]
        texts, _ = load_wikipedia_people(
            filepath=str(filepath),
            text_col=cfg["text_col"],
            name_col=cfg["name_col"],
            min_text_length=cfg["min_text_length"],
            auto_download=False,
        )

    status(f"[{dataset}] Preprocessing {len(texts):,} documents…")
    processed, _ = preprocess_corpus(texts)

    proc_path = _PROCESSED_DIR / f"{dataset}_clean.pkl"
    with open(proc_path, "wb") as f:
        pickle.dump(processed, f)

    status(f"[{dataset}] Building TF-IDF + SVD features…")
    tfidf_cfg = config["tfidf"]
    X_tfidf, vectorizer = build_tfidf(
        processed,
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
    )
    X_reduced, svd_pipeline = reduce_with_svd(
        X_tfidf,
        n_components=config["svd"]["n_components"],
        random_state=config["seed"],
    )
    save_feature_artifacts(
        vectorizer, svd_pipeline, X_tfidf, X_reduced,
        dataset_name=dataset,
        models_dir=str(_MODELS_DIR),
        processed_dir=str(_PROCESSED_DIR),
    )

    status(f"[{dataset}] Running clustering experiments…")
    results = run_all_experiments(X_reduced, config, dataset=dataset, seed=config["seed"])
    save_cluster_results(results, dataset=dataset, models_dir=str(_MODELS_DIR))

    status(f"[{dataset}] Evaluating clusters…")
    metrics_df, _ = evaluate_all_experiments(
        results=results, X=X_reduced, vectorizer=vectorizer,
        X_tfidf=X_tfidf, corpus=processed, config=config,
    )
    save_metrics(metrics_df, dataset=dataset, reports_dir=str(_REPORTS_DIR))
    status(f"[{dataset}] Done — {len(results['kmeans'])} K-Means models ready.")
