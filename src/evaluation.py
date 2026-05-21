"""Cluster evaluation: silhouette, Davies-Bouldin, Calinski-Harabasz, top terms, examples."""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.clustering import ClusterResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
    seed: int = 42,
) -> float:
    """Silhouette score, sampled for large corpora.

    Score in [-1, 1]. Higher → better-defined clusters.
    Sampling is used when n_docs > sample_size to keep runtime feasible.
    """
    n = X.shape[0]
    n_clusters = len(set(labels))
    if n_clusters < 2:
        logger.warning("Silhouette requires ≥ 2 clusters; got %d — returning NaN", n_clusters)
        return float("nan")

    actual_sample = min(sample_size, n)
    score = silhouette_score(
        X, labels, sample_size=actual_sample, random_state=seed
    )
    logger.debug("Silhouette=%.4f (n=%d, sample=%d, k=%d)", score, n, actual_sample, n_clusters)
    return float(score)


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin Index. Lower → better."""
    if len(set(labels)) < 2:
        return float("nan")
    return float(davies_bouldin_score(X, labels))


def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz Score (variance ratio). Higher → better."""
    if len(set(labels)) < 2:
        return float("nan")
    return float(calinski_harabasz_score(X, labels))


# ---------------------------------------------------------------------------
# Cluster naming
# ---------------------------------------------------------------------------

def get_cluster_names(
    top_terms: dict[int, list[str]],
    n_words: int = 3,
) -> dict[int, str]:
    """Generate a short human-readable name for each cluster from its top TF-IDF terms.

    Args:
        top_terms: output of get_top_terms — {cluster_id: [term, ...]}
        n_words: number of top terms to join into the name

    Returns:
        dict mapping cluster_id → name string, e.g. {0: "science physics quantum"}
    """
    return {
        cid: " / ".join(terms[:n_words])
        for cid, terms in top_terms.items()
    }


def cluster_display_label(cluster_id: int, cluster_names: dict[int, str]) -> str:
    """Format a cluster for display: 'Cluster 3 — science / physics / quantum'."""
    name = cluster_names.get(cluster_id, "")
    if name:
        return f"Cluster {cluster_id} — {name}"
    return f"Cluster {cluster_id}"


# ---------------------------------------------------------------------------
# Qualitative analysis
# ---------------------------------------------------------------------------

def get_top_terms(
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    X_tfidf,
    n: int = 15,
) -> dict[int, list[str]]:
    """Extract top TF-IDF terms per cluster by averaging document vectors.

    Args:
        labels: cluster assignment per document
        vectorizer: fitted TfidfVectorizer
        X_tfidf: sparse TF-IDF matrix (n_docs, vocab)
        n: number of top terms to return per cluster

    Returns:
        dict mapping cluster_id → list of top-n term strings
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_ids = sorted(set(labels))
    top_terms: dict[int, list[str]] = {}

    for cid in cluster_ids:
        mask = labels == cid
        cluster_matrix = X_tfidf[mask]
        # Mean TF-IDF across all documents in this cluster
        mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:n]
        top_terms[int(cid)] = feature_names[top_idx].tolist()

    return top_terms


def get_cluster_examples(
    labels: np.ndarray,
    corpus: list[str],
    n: int = 3,
    snippet_len: int = 200,
) -> dict[int, list[str]]:
    """Return up to n document snippets per cluster.

    Args:
        labels: cluster assignment per document
        corpus: preprocessed (or raw) documents aligned with labels
        n: number of examples per cluster
        snippet_len: max characters per snippet

    Returns:
        dict mapping cluster_id → list of doc snippets
    """
    cluster_ids = sorted(set(labels))
    examples: dict[int, list[str]] = {}

    for cid in cluster_ids:
        indices = np.where(labels == cid)[0]
        sampled = indices[:n]
        examples[int(cid)] = [corpus[i][:snippet_len] for i in sampled]

    return examples


def cluster_size_distribution(labels: np.ndarray) -> dict[int, int]:
    """Count documents per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


# ---------------------------------------------------------------------------
# Full experiment evaluation
# ---------------------------------------------------------------------------

def evaluate_result(
    result: ClusterResult,
    X: np.ndarray,
    vectorizer: TfidfVectorizer,
    X_tfidf,
    corpus: list[str],
    config: dict,
) -> dict:
    """Compute all metrics for a single ClusterResult.

    For hierarchical results fitted on a sample (extra['sample_idx'] present),
    metrics are computed on the same sample to keep X and labels aligned.
    """
    eval_cfg = config.get("evaluation", {})
    top_n = eval_cfg.get("top_n_terms", 15)
    sample_n = eval_cfg.get("sample_docs_per_cluster", 3)
    sil_sample = eval_cfg.get("silhouette_sample_size", 5000)

    # Use the sample subset when labels were fitted on a subsample
    sample_idx = result.extra.get("sample_idx", None)
    if sample_idx is not None:
        X_eval = X[sample_idx]
        X_tfidf_eval = X_tfidf[sample_idx]
        corpus_eval = [corpus[i] for i in sample_idx]
    else:
        X_eval = X
        X_tfidf_eval = X_tfidf
        corpus_eval = corpus

    sil = compute_silhouette(X_eval, result.labels, sample_size=sil_sample)
    db = compute_davies_bouldin(X_eval, result.labels)
    ch = compute_calinski_harabasz(X_eval, result.labels)
    top_terms = get_top_terms(result.labels, vectorizer, X_tfidf_eval, n=top_n)
    examples = get_cluster_examples(result.labels, corpus_eval, n=sample_n)
    sizes = cluster_size_distribution(result.labels)

    return {
        "algorithm": result.algorithm,
        "dataset": result.dataset,
        "k": result.k,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "runtime_s": result.runtime_s,
        "top_terms": top_terms,
        "examples": examples,
        "cluster_sizes": sizes,
        **result.extra,
    }


def evaluate_all_experiments(
    results: dict[str, list[ClusterResult]],
    X: np.ndarray,
    vectorizer: TfidfVectorizer,
    X_tfidf,
    corpus: list[str],
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate every experiment and return a metrics DataFrame + full detail dict.

    Args:
        results: output of clustering.run_all_experiments
        X: reduced feature matrix used for scoring
        vectorizer: fitted TfidfVectorizer
        X_tfidf: sparse TF-IDF matrix
        corpus: preprocessed documents
        config: full project config

    Returns:
        (metrics_df, detail_dict)
        metrics_df: one row per experiment with scalar metrics
        detail_dict: full evaluate_result output per experiment
    """
    rows = []
    details = {}

    for algo, result_list in results.items():
        for r in result_list:
            logger.info("Evaluating %s k=%d ...", algo, r.k)
            ev = evaluate_result(r, X, vectorizer, X_tfidf, corpus, config)
            key = f"{algo}_k{r.k}"
            if algo == "hierarchical":
                key += f"_{r.extra.get('linkage', 'ward')}"
            elif algo == "gmm":
                key += f"_{r.extra.get('covariance_type', 'full')}"
            details[key] = ev

            row = {
                "algorithm": algo,
                "dataset": r.dataset,
                "k": r.k,
                "silhouette": ev["silhouette"],
                "davies_bouldin": ev["davies_bouldin"],
                "calinski_harabasz": ev["calinski_harabasz"],
                "runtime_s": ev["runtime_s"],
            }
            # Carry over any algorithm-specific extras (inertia, bic, aic, linkage…)
            for extra_key in ("inertia", "bic", "aic", "linkage", "covariance_type"):
                if extra_key in ev:
                    row[extra_key] = ev[extra_key]
            rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(
        ["algorithm", "k"], ignore_index=True
    )
    logger.info("Evaluation complete: %d experiments", len(rows))
    return metrics_df, details


def save_metrics(
    metrics_df: pd.DataFrame,
    dataset: str,
    reports_dir: str = "outputs/reports",
) -> None:
    """Save the metrics DataFrame to CSV."""
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    path = f"{reports_dir}/clustering_metrics_{dataset}.csv"
    metrics_df.to_csv(path, index=False)
    logger.info("Metrics saved → %s", path)


def best_result(
    metrics_df: pd.DataFrame,
    algorithm: Optional[str] = None,
    metric: str = "silhouette",
    higher_is_better: bool = True,
) -> pd.Series:
    """Return the row with the best metric value, optionally filtered by algorithm."""
    df = metrics_df if algorithm is None else metrics_df[metrics_df["algorithm"] == algorithm]
    if higher_is_better:
        return df.loc[df[metric].idxmax()]
    return df.loc[df[metric].idxmin()]
