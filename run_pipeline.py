"""End-to-end pipeline runner.

Usage:
    python run_pipeline.py                      # both datasets, all steps
    python run_pipeline.py --dataset newsgroups # one dataset only
    python run_pipeline.py --steps preprocess features cluster evaluate visualize
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_load(config: dict, dataset: str) -> tuple[list[str], list | None]:
    """Load raw texts from the specified dataset."""
    from src.data_loader import load_newsgroups, load_wikipedia_people, save_raw

    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "newsgroups":
        cfg = config["datasets"]["newsgroups"]
        texts, labels, target_names = load_newsgroups(
            subset=cfg["subset"],
            remove=tuple(cfg["remove"]),
        )
        save_raw((texts, labels, target_names), raw_dir / "newsgroups_raw.pkl")
        logger.info("Newsgroups: %d docs, %d categories", len(texts), len(target_names))
        return texts, labels

    elif dataset == "wikipedia":
        cfg = config["datasets"]["wikipedia"]
        filepath = Path(config["paths"]["data_raw"]) / cfg["filename"]
        texts, names = load_wikipedia_people(
            filepath=str(filepath),
            text_col=cfg["text_col"],
            name_col=cfg["name_col"],
            min_text_length=cfg["min_text_length"],
        )
        save_raw((texts, names), raw_dir / "wikipedia_raw.pkl")
        logger.info("Wikipedia: %d docs loaded", len(texts))
        return texts, None

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def step_preprocess(texts: list[str], config: dict, dataset: str) -> list[str]:
    """Clean and lemmatize corpus; save processed version."""
    from src.preprocessing import preprocess_corpus

    processed, kept = preprocess_corpus(texts)

    proc_dir = Path(config["paths"]["data_processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / f"{dataset}_clean.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(processed, f)
    logger.info("Saved %d processed docs → %s", len(processed), out_path)
    return processed


def step_features(processed: list[str], config: dict, dataset: str):
    """Build TF-IDF + SVD and save all artifacts."""
    from src.features import (
        build_tfidf,
        reduce_with_svd,
        save_feature_artifacts,
    )

    tfidf_cfg = config["tfidf"]
    X_tfidf, vectorizer = build_tfidf(
        processed,
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
    )

    svd_cfg = config["svd"]
    X_reduced, svd_pipeline = reduce_with_svd(
        X_tfidf,
        n_components=svd_cfg["n_components"],
        random_state=config["seed"],
    )

    save_feature_artifacts(
        vectorizer=vectorizer,
        svd_pipeline=svd_pipeline,
        X_tfidf=X_tfidf,
        X_reduced=X_reduced,
        dataset_name=dataset,
        models_dir=config["paths"]["models"],
        processed_dir=config["paths"]["data_processed"],
    )
    return X_tfidf, vectorizer, X_reduced


def step_cluster(X_reduced: np.ndarray, config: dict, dataset: str) -> dict:
    """Run all clustering experiments and save models."""
    from src.clustering import run_all_experiments, save_cluster_results

    results = run_all_experiments(
        X_reduced,
        config,
        dataset=dataset,
        seed=config["seed"],
    )
    save_cluster_results(results, dataset=dataset, models_dir=config["paths"]["models"])
    return results


def step_evaluate(
    results: dict,
    X_reduced: np.ndarray,
    vectorizer,
    X_tfidf,
    processed: list[str],
    config: dict,
    dataset: str,
) -> tuple:
    """Compute all metrics and save CSV."""
    from src.evaluation import evaluate_all_experiments, save_metrics

    metrics_df, details = evaluate_all_experiments(
        results=results,
        X=X_reduced,
        vectorizer=vectorizer,
        X_tfidf=X_tfidf,
        corpus=processed,
        config=config,
    )
    save_metrics(metrics_df, dataset=dataset, reports_dir=config["paths"]["reports"])

    # Print summary table
    cols = ["algorithm", "k", "silhouette", "davies_bouldin", "calinski_harabasz"]
    cols = [c for c in cols if c in metrics_df.columns]
    logger.info("\n%s", metrics_df[cols].to_string(index=False))

    return metrics_df, details


def step_visualize(
    results: dict,
    X_reduced: np.ndarray,
    vectorizer,
    X_tfidf,
    metrics_df,
    details: dict,
    config: dict,
    dataset: str,
) -> None:
    """Generate all plots for the best model per algorithm."""
    from src.evaluation import best_result
    from src.visualization import (
        plot_cluster_sizes,
        plot_dendrogram,
        plot_elbow_curve,
        plot_metrics_by_k,
        plot_pca_clusters,
        plot_silhouette_comparison,
        plot_top_terms_heatmap,
        plot_tsne_clusters,
    )

    figs_dir = config["paths"]["figures"]
    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    # Elbow curve (K-Means only)
    kmeans_results = results.get("kmeans", [])
    if kmeans_results:
        inertias = [r.extra["inertia"] for r in kmeans_results]
        k_range = [r.k for r in kmeans_results]
        plot_elbow_curve(inertias, k_range, dataset, figures_dir=figs_dir)

    # Silhouette comparison
    plot_silhouette_comparison(metrics_df, dataset_name=dataset, figures_dir=figs_dir)

    # Metrics by k (per metric)
    for metric in ["silhouette", "davies_bouldin"]:
        if metric in metrics_df.columns:
            plot_metrics_by_k(metrics_df, dataset_name=dataset, metric=metric, figures_dir=figs_dir)

    # Per-algorithm: best k → PCA, t-SNE, cluster sizes, top-terms heatmap
    for algo in ("kmeans", "hierarchical", "gmm"):
        if algo not in metrics_df["algorithm"].unique():
            continue
        best = best_result(metrics_df, algorithm=algo, metric="silhouette")
        k = int(best["k"])

        # Load the best model and get labels
        model_files = list(Path(config["paths"]["models"]).glob(f"{algo}_{dataset}_k{k}*.pkl"))
        if not model_files:
            logger.warning("No saved model found for %s k=%d — skipping visualizations", algo, k)
            continue
        import joblib
        model = joblib.load(model_files[0])

        # AgglomerativeClustering stores labels from its fit sample; use that sample for visuals
        if hasattr(model, "predict"):
            labels_arr = model.predict(X_reduced)
            X_vis = X_reduced
            X_tfidf_vis = X_tfidf
        else:
            labels_arr = model.labels_
            n_labels = len(labels_arr)
            if n_labels < X_reduced.shape[0]:
                # Hierarchical was fitted on a sample — visualise on same sample size
                rng = np.random.default_rng(config["seed"])
                sample_idx = rng.choice(X_reduced.shape[0], n_labels, replace=False)
                X_vis = X_reduced[sample_idx]
                X_tfidf_vis = X_tfidf[sample_idx]
            else:
                X_vis = X_reduced
                X_tfidf_vis = X_tfidf

        plot_pca_clusters(X_vis, labels_arr, algo, k, dataset, figures_dir=figs_dir)
        plot_tsne_clusters(
            X_vis, labels_arr, algo, k, dataset,
            perplexity=config["visualization"]["tsne"]["perplexity"],
            n_iter=config["visualization"]["tsne"]["n_iter"],
            figures_dir=figs_dir,
        )
        plot_cluster_sizes(labels_arr, algo, k, dataset, figures_dir=figs_dir)

        # Top-terms heatmap
        from src.evaluation import get_top_terms
        top_terms = get_top_terms(labels_arr, vectorizer, X_tfidf_vis, n=10)
        plot_top_terms_heatmap(
            top_terms, labels_arr, vectorizer, X_tfidf_vis,
            algo, k, dataset, n_terms=10, figures_dir=figs_dir,
        )

    # Dendrogram (one linkage)
    plot_dendrogram(
        X_reduced, linkage_method="ward", dataset_name=dataset,
        sample_size=config["visualization"]["dendrogram"]["sample_size"],
        truncate_level=config["visualization"]["dendrogram"]["truncate_level"],
        figures_dir=figs_dir,
    )

    logger.info("All visualizations saved to %s/", figs_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, steps: list[str], config: dict) -> None:
    logger.info("=" * 60)
    logger.info("DATASET: %s  |  STEPS: %s", dataset.upper(), steps)
    logger.info("=" * 60)

    texts, labels = None, None
    processed = None
    X_tfidf, vectorizer, X_reduced = None, None, None
    results = None
    metrics_df, details = None, None

    if "load" in steps or "preprocess" in steps:
        texts, labels = step_load(config, dataset)

    if "preprocess" in steps:
        processed = step_preprocess(texts, config, dataset)
    else:
        proc_path = Path(config["paths"]["data_processed"]) / f"{dataset}_clean.pkl"
        if proc_path.exists():
            with open(proc_path, "rb") as f:
                processed = pickle.load(f)
            logger.info("Loaded preprocessed corpus (%d docs) from cache", len(processed))

    if "features" in steps:
        X_tfidf, vectorizer, X_reduced = step_features(processed, config, dataset)
    else:
        from src.features import load_feature_artifacts
        try:
            vectorizer, svd_pipeline, X_tfidf, X_reduced = load_feature_artifacts(
                dataset_name=dataset,
                models_dir=config["paths"]["models"],
                processed_dir=config["paths"]["data_processed"],
            )
        except FileNotFoundError:
            logger.warning("Feature artifacts not found — run with --steps features first")

    if "cluster" in steps and X_reduced is not None:
        results = step_cluster(X_reduced, config, dataset)

    if "evaluate" in steps and results is not None:
        metrics_df, details = step_evaluate(
            results, X_reduced, vectorizer, X_tfidf, processed, config, dataset
        )

    if "visualize" in steps and results is not None and metrics_df is not None:
        step_visualize(
            results, X_reduced, vectorizer, X_tfidf,
            metrics_df, details, config, dataset
        )

    logger.info("Pipeline complete for dataset: %s", dataset)


def main():
    parser = argparse.ArgumentParser(description="Document Clustering Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["newsgroups", "wikipedia", "both"],
        default="both",
        help="Which dataset to process",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["load", "preprocess", "features", "cluster", "evaluate", "visualize"],
        choices=["load", "preprocess", "features", "cluster", "evaluate", "visualize"],
        help="Pipeline steps to run",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    datasets = (
        ["newsgroups", "wikipedia"] if args.dataset == "both" else [args.dataset]
    )
    for ds in datasets:
        run_dataset(ds, steps=args.steps, config=config)

    logger.info("All done.")


if __name__ == "__main__":
    main()
