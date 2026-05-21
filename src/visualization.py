"""Visualization: PCA/t-SNE scatter plots, elbow curves, dendrograms, heatmaps."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

# Global style
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
_PALETTE = "tab20"


def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# Elbow curve (K-Means inertia vs. k)
# ---------------------------------------------------------------------------

def plot_elbow_curve(
    inertias: list[float],
    k_range: list[int],
    dataset_name: str,
    figures_dir: str = "outputs/figures",
) -> str:
    """Line plot of K-Means inertia vs. number of clusters."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_range, inertias, "o-", linewidth=2, markersize=7, color="steelblue")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-cluster SSE)")
    ax.set_title(f"K-Means Elbow Curve — {dataset_name}")
    ax.set_xticks(k_range)
    ax.grid(alpha=0.3)
    path = f"{figures_dir}/elbow_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Silhouette comparison bar chart
# ---------------------------------------------------------------------------

def plot_silhouette_comparison(
    metrics_df: pd.DataFrame,
    dataset_name: str,
    figures_dir: str = "outputs/figures",
) -> str:
    """Grouped bar chart: silhouette score per (algorithm, k)."""
    df = metrics_df[metrics_df["dataset"] == dataset_name].copy()
    df["label"] = df["algorithm"] + " k=" + df["k"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"kmeans": "steelblue", "hierarchical": "darkorange", "gmm": "seagreen"}
    bar_colors = df["algorithm"].map(colors).tolist()

    bars = ax.bar(df["label"], df["silhouette"], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Silhouette Score Comparison — {dataset_name}")
    ax.tick_params(axis="x", rotation=60)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(color=c, label=a) for a, c in colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.grid(axis="y", alpha=0.3)
    path = f"{figures_dir}/silhouette_comparison_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# PCA scatter
# ---------------------------------------------------------------------------

def plot_pca_clusters(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    algo_name: str,
    k: int,
    dataset_name: str,
    figures_dir: str = "outputs/figures",
) -> str:
    """2-D PCA scatter coloured by cluster label."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_reduced)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap=_PALETTE,
        s=8, alpha=0.6, linewidths=0,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title(f"PCA — {algo_name.upper()} k={k} — {dataset_name}")
    ax.grid(alpha=0.2)

    path = f"{figures_dir}/pca_{algo_name}_k{k}_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# t-SNE scatter
# ---------------------------------------------------------------------------

def plot_tsne_clusters(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    algo_name: str,
    k: int,
    dataset_name: str,
    perplexity: int = 30,
    n_iter: int = 1000,   # kept for config compatibility; mapped to max_iter internally
    figures_dir: str = "outputs/figures",
    max_samples: int = 5000,
) -> str:
    """2-D t-SNE scatter coloured by cluster label.

    Subsamples to max_samples for speed on large corpora.
    """
    n = X_reduced.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        X_plot, y_plot = X_reduced[idx], labels[idx]
        title_suffix = f" (sample {max_samples:,})"
    else:
        X_plot, y_plot = X_reduced, labels
        title_suffix = ""

    logger.info("Running t-SNE on %d points...", X_plot.shape[0])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,   # sklearn >=1.5 renamed n_iter → max_iter
        random_state=42,
        init="pca",
    )
    coords = tsne.fit_transform(X_plot)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=y_plot, cmap=_PALETTE,
        s=8, alpha=0.6, linewidths=0,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE — {algo_name.upper()} k={k} — {dataset_name}{title_suffix}")
    ax.grid(alpha=0.2)

    path = f"{figures_dir}/tsne_{algo_name}_k{k}_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram(
    X_sample: np.ndarray,
    linkage_method: str,
    dataset_name: str,
    truncate_level: int = 5,
    figures_dir: str = "outputs/figures",
    sample_size: int = 500,
) -> str:
    """Hierarchical clustering dendrogram on a subsample.

    Args:
        X_sample: dense feature matrix (will be sub-sampled)
        linkage_method: 'ward', 'complete', or 'average'
        dataset_name: for file naming and title
        truncate_level: show only the last p merges
        sample_size: number of documents to sample for readability
    """
    n = X_sample.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, sample_size, replace=False)
        X_plot = X_sample[idx]
    else:
        X_plot = X_sample

    logger.info("Building linkage for dendrogram (%d points)...", X_plot.shape[0])
    Z = linkage(X_plot, method=linkage_method)

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z, ax=ax,
        truncate_mode="level",
        p=truncate_level,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_title(f"Dendrogram ({linkage_method} linkage) — {dataset_name}")
    ax.set_xlabel("Document Index / Cluster")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", alpha=0.2)

    path = f"{figures_dir}/dendrogram_{linkage_method}_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Cluster size distribution
# ---------------------------------------------------------------------------

def plot_cluster_sizes(
    labels: np.ndarray,
    algo_name: str,
    k: int,
    dataset_name: str,
    figures_dir: str = "outputs/figures",
) -> str:
    """Bar chart of document count per cluster, flagging thin clusters."""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    threshold = total * 0.01  # flag clusters < 1 % of corpus

    colors = ["tomato" if c < threshold else "steelblue" for c in counts]

    fig, ax = plt.subplots(figsize=(max(8, len(unique) // 2), 5))
    ax.bar(unique.astype(str), counts, color=colors, edgecolor="white")
    ax.axhline(threshold, linestyle="--", color="red", linewidth=1, label="1% threshold")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Document Count")
    ax.set_title(f"Cluster Size Distribution — {algo_name.upper()} k={k} — {dataset_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = f"{figures_dir}/cluster_sizes_{algo_name}_k{k}_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Top-terms heatmap
# ---------------------------------------------------------------------------

def plot_top_terms_heatmap(
    top_terms: dict[int, list[str]],
    labels: np.ndarray,
    vectorizer,
    X_tfidf,
    algo_name: str,
    k: int,
    dataset_name: str,
    n_terms: int = 10,
    figures_dir: str = "outputs/figures",
) -> str:
    """Heatmap: clusters (rows) × top terms (columns), colour = mean TF-IDF weight."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_ids = sorted(top_terms.keys())

    # Collect unique top terms across all clusters (up to n_terms each)
    all_terms: list[str] = []
    for cid in cluster_ids:
        for t in top_terms[cid][:n_terms]:
            if t not in all_terms:
                all_terms.append(t)

    # Build weight matrix: rows=clusters, cols=terms
    term_idx = {t: i for i, t in enumerate(all_terms)}
    matrix = np.zeros((len(cluster_ids), len(all_terms)))
    for row_i, cid in enumerate(cluster_ids):
        mask = labels == cid
        cluster_tfidf = X_tfidf[mask]
        mean_vec = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        for term in top_terms[cid][:n_terms]:
            if term in feature_names and term in term_idx:
                fidx = np.where(feature_names == term)[0]
                if len(fidx):
                    matrix[row_i, term_idx[term]] = mean_vec[fidx[0]]

    fig, ax = plt.subplots(figsize=(max(14, len(all_terms) // 2), max(6, len(cluster_ids) // 2)))
    sns.heatmap(
        matrix,
        xticklabels=all_terms,
        yticklabels=[f"Cluster {c}" for c in cluster_ids],
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        cbar_kws={"label": "Mean TF-IDF"},
    )
    ax.set_title(f"Top Terms Heatmap — {algo_name.upper()} k={k} — {dataset_name}")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    path = f"{figures_dir}/terms_heatmap_{algo_name}_k{k}_{dataset_name}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Metrics comparison line plot (silhouette vs. k per algorithm)
# ---------------------------------------------------------------------------

def plot_metrics_by_k(
    metrics_df: pd.DataFrame,
    dataset_name: str,
    metric: str = "silhouette",
    figures_dir: str = "outputs/figures",
) -> str:
    """Line plot: metric score vs. k, one line per algorithm."""
    df = metrics_df[metrics_df["dataset"] == dataset_name]
    fig, ax = plt.subplots(figsize=(10, 5))

    for algo, grp in df.groupby("algorithm"):
        grp_sorted = grp.sort_values("k")
        ax.plot(grp_sorted["k"], grp_sorted[metric], "o-", label=algo, linewidth=2, markersize=7)

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs. k — {dataset_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    path = f"{figures_dir}/{metric}_by_k_{dataset_name}.png"
    _save(fig, path)
    return path
