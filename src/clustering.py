"""Clustering algorithms: K-Means, Agglomerative Hierarchical, and GMM."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Holds everything produced by a single clustering run."""

    algorithm: str
    dataset: str
    k: int
    labels: np.ndarray
    model: Any
    runtime_s: float
    extra: dict = field(default_factory=dict)  # inertia, bic, aic, linkage, cov_type …


# ---------------------------------------------------------------------------
# Individual algorithm runners
# ---------------------------------------------------------------------------

def run_kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    seed: int = 42,
    dataset: str = "unknown",
) -> ClusterResult:
    """Fit K-Means and return a ClusterResult.

    Args:
        X: dense feature matrix (n_docs, n_features)
        k: number of clusters
        n_init: number of independent random initialisations
        max_iter: maximum EM iterations per run
        seed: random state for reproducibility
        dataset: label used for logging/saving

    Returns:
        ClusterResult with labels and fitted model
    """
    logger.info("[KMeans] dataset=%s k=%d", dataset, k)
    t0 = time.perf_counter()
    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[KMeans] k=%d  inertia=%.2f  time=%.1fs", k, model.inertia_, elapsed
    )
    return ClusterResult(
        algorithm="kmeans",
        dataset=dataset,
        k=k,
        labels=labels,
        model=model,
        runtime_s=elapsed,
        extra={"inertia": model.inertia_},
    )


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
    dataset: str = "unknown",
) -> ClusterResult:
    """Fit Agglomerative Hierarchical Clustering.

    Args:
        X: dense feature matrix
        n_clusters: target number of clusters
        linkage: 'ward', 'complete', or 'average'
        dataset: label for logging

    Returns:
        ClusterResult
    """
    logger.info("[Hierarchical] dataset=%s k=%d linkage=%s", dataset, n_clusters, linkage)
    t0 = time.perf_counter()
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0
    logger.info("[Hierarchical] k=%d linkage=%s  time=%.1fs", n_clusters, linkage, elapsed)
    return ClusterResult(
        algorithm="hierarchical",
        dataset=dataset,
        k=n_clusters,
        labels=labels,
        model=model,
        runtime_s=elapsed,
        extra={"linkage": linkage},
    )


def run_gmm(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = "full",
    max_iter: int = 100,
    seed: int = 42,
    dataset: str = "unknown",
) -> ClusterResult:
    """Fit a Gaussian Mixture Model.

    Args:
        X: dense feature matrix
        n_components: number of mixture components (clusters)
        covariance_type: 'full', 'tied', 'diag', or 'spherical'
        max_iter: EM iteration cap
        seed: random state
        dataset: label for logging

    Returns:
        ClusterResult with BIC and AIC in extra
    """
    logger.info(
        "[GMM] dataset=%s k=%d cov=%s", dataset, n_components, covariance_type
    )
    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X)
    labels = model.predict(X)
    elapsed = time.perf_counter() - t0

    bic = model.bic(X)
    aic = model.aic(X)
    logger.info(
        "[GMM] k=%d cov=%s  BIC=%.1f  AIC=%.1f  time=%.1fs",
        n_components, covariance_type, bic, aic, elapsed,
    )
    return ClusterResult(
        algorithm="gmm",
        dataset=dataset,
        k=n_components,
        labels=labels,
        model=model,
        runtime_s=elapsed,
        extra={"bic": bic, "aic": aic, "covariance_type": covariance_type},
    )


# ---------------------------------------------------------------------------
# Grid-search over all configured experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    X: np.ndarray,
    config: dict,
    dataset: str,
    seed: int = 42,
) -> dict[str, list[ClusterResult]]:
    """Run every algorithm × hyperparameter combination defined in config.

    Args:
        X: reduced feature matrix (n_docs, n_components)
        config: full project config dict (clustering sub-section used)
        dataset: dataset name ('newsgroups' or 'wikipedia')
        seed: random seed

    Returns:
        dict keyed by algorithm name, values are lists of ClusterResult
    """
    cfg = config["clustering"]
    results: dict[str, list[ClusterResult]] = {
        "kmeans": [],
        "hierarchical": [],
        "gmm": [],
    }

    # K-Means sweep
    for k in cfg["kmeans"]["k_range"]:
        r = run_kmeans(
            X, k=k,
            n_init=cfg["kmeans"]["n_init"],
            max_iter=cfg["kmeans"]["max_iter"],
            seed=seed, dataset=dataset,
        )
        results["kmeans"].append(r)

    # Hierarchical sweep — sample large corpora to avoid O(n²) memory crash
    max_hier = cfg["hierarchical"].get("max_docs", 8000)
    hier_sample_idx: Optional[np.ndarray] = None
    X_hier = X
    if X.shape[0] > max_hier:
        rng = np.random.default_rng(seed)
        hier_sample_idx = rng.choice(X.shape[0], max_hier, replace=False)
        X_hier = X[hier_sample_idx]
        logger.warning(
            "[Hierarchical] corpus too large (%d docs) — sampling %d for fitting",
            X.shape[0], max_hier,
        )
    for k in cfg["hierarchical"]["n_clusters"]:
        for linkage in cfg["hierarchical"]["linkage"]:
            r = run_hierarchical(X_hier, n_clusters=k, linkage=linkage, dataset=dataset)
            if hier_sample_idx is not None:
                r.extra["sample_idx"] = hier_sample_idx
            results["hierarchical"].append(r)

    # GMM sweep
    for k in cfg["gmm"]["n_components"]:
        for cov in cfg["gmm"]["covariance_type"]:
            r = run_gmm(
                X, n_components=k,
                covariance_type=cov,
                max_iter=cfg["gmm"]["max_iter"],
                seed=seed, dataset=dataset,
            )
            results["gmm"].append(r)

    total = sum(len(v) for v in results.values())
    logger.info("Completed %d clustering experiments for dataset: %s", total, dataset)
    return results


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_cluster_results(
    results: dict[str, list[ClusterResult]],
    dataset: str,
    models_dir: str = "models",
) -> None:
    """Save all fitted models to disk with consistent naming."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    for algo, result_list in results.items():
        for r in result_list:
            suffix = _result_suffix(r)
            path = f"{models_dir}/{algo}_{dataset}_{suffix}.pkl"
            joblib.dump(r.model, path)
    logger.info("Saved all cluster models for dataset: %s", dataset)


def load_cluster_model(
    algorithm: str,
    dataset: str,
    k: int,
    models_dir: str = "models",
    linkage: Optional[str] = None,
    covariance_type: Optional[str] = None,
) -> Any:
    """Load a specific saved clustering model."""
    if algorithm == "kmeans":
        suffix = f"k{k}"
    elif algorithm == "hierarchical":
        link = linkage or "ward"
        suffix = f"k{k}_{link}"
    else:  # gmm
        cov = covariance_type or "full"
        suffix = f"k{k}_{cov}"
    path = f"{models_dir}/{algorithm}_{dataset}_{suffix}.pkl"
    return joblib.load(path)


def _result_suffix(r: ClusterResult) -> str:
    if r.algorithm == "kmeans":
        return f"k{r.k}"
    if r.algorithm == "hierarchical":
        return f"k{r.k}_{r.extra.get('linkage', 'ward')}"
    return f"k{r.k}_{r.extra.get('covariance_type', 'full')}"
