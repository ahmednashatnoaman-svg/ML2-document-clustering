"""One-time model initialisation for Hugging Face Spaces.

Called by the Streamlit app at startup when no trained models are found.
Runs a reduced pipeline so the Space becomes interactive in ~60 seconds:
  - Wikipedia People  : full k sweep (538 docs — very fast)
  - 20 Newsgroups     : kmeans only, k in [5, 10, 20] (skips hierarchical/GMM for speed)
"""

import logging
import pickle
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def _load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def needs_setup() -> bool:
    """Return True if essential model files are missing."""
    return not Path("models/tfidf_wikipedia.pkl").exists()


def run_setup(status_callback=None) -> None:
    """Run a fast pipeline for both datasets.

    Args:
        status_callback: callable(message: str) — used to update Streamlit status text.
    """
    def _status(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    config = _load_config()

    # Use a reduced config for cloud startup speed
    fast_config = dict(config)
    fast_config["clustering"] = {
        "kmeans": {"k_range": [5, 10, 20], "n_init": 5, "max_iter": 200},
        "hierarchical": {"n_clusters": [5, 10], "linkage": ["ward"]},
        "gmm": {"n_components": [5, 10], "covariance_type": ["tied"], "max_iter": 50},
    }

    for dataset in ("wikipedia", "newsgroups"):
        _run_dataset(dataset, fast_config, _status)

    _status("Setup complete — dashboard ready!")


def _run_dataset(dataset: str, config: dict, status) -> None:
    from src.data_loader import load_newsgroups, load_wikipedia_people, save_raw
    from src.preprocessing import preprocess_corpus
    from src.features import build_tfidf, reduce_with_svd, save_feature_artifacts
    from src.clustering import run_all_experiments, save_cluster_results
    from src.evaluation import evaluate_all_experiments, save_metrics

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    # Load
    status(f"[{dataset}] Loading data...")
    if dataset == "newsgroups":
        cfg = config["datasets"]["newsgroups"]
        texts, labels, names = load_newsgroups(
            subset=cfg["subset"], remove=tuple(cfg["remove"])
        )
    else:
        cfg = config["datasets"]["wikipedia"]
        filepath = Path("data/raw") / cfg["filename"]
        texts, names = load_wikipedia_people(
            filepath=str(filepath),
            text_col=cfg["text_col"],
            name_col=cfg["name_col"],
            min_text_length=cfg["min_text_length"],
        )

    # Preprocess
    status(f"[{dataset}] Preprocessing {len(texts)} documents...")
    processed, _ = preprocess_corpus(texts)

    proc_path = Path("data/processed") / f"{dataset}_clean.pkl"
    with open(proc_path, "wb") as f:
        pickle.dump(processed, f)

    # Features
    status(f"[{dataset}] Building TF-IDF + SVD features...")
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
        X_tfidf, n_components=config["svd"]["n_components"], random_state=config["seed"]
    )
    save_feature_artifacts(
        vectorizer, svd_pipeline, X_tfidf, X_reduced,
        dataset_name=dataset,
        models_dir="models",
        processed_dir="data/processed",
    )

    # Cluster
    status(f"[{dataset}] Running clustering experiments...")
    results = run_all_experiments(X_reduced, config, dataset=dataset, seed=config["seed"])
    save_cluster_results(results, dataset=dataset, models_dir="models")

    # Evaluate
    status(f"[{dataset}] Evaluating clusters...")
    metrics_df, _ = evaluate_all_experiments(
        results=results, X=X_reduced, vectorizer=vectorizer,
        X_tfidf=X_tfidf, corpus=processed, config=config,
    )
    save_metrics(metrics_df, dataset=dataset, reports_dir="outputs/reports")
    status(f"[{dataset}] Done — {len(results['kmeans'])} K-Means models ready.")
