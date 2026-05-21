"""Streamlit interactive dashboard for Document Clustering Explorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.preprocessing import preprocess_text
from src.features import transform_new_docs

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Clustering Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #c8e0f5; margin: 0.3rem 0 0; font-size: 1rem; }

    .kpi-card {
        background: white;
        border: 1px solid #e0e7ef;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .kpi-card .label { color: #6b7280; font-size: 0.78rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-card .value { color: #1e3a5f; font-size: 1.9rem; font-weight: 700;
                       line-height: 1.2; margin: 0.2rem 0; }
    .kpi-card .delta { font-size: 0.78rem; }
    .delta-good { color: #16a34a; }
    .delta-info { color: #2563eb; }

    .section-header {
        color: #1e3a5f;
        font-size: 1.05rem;
        font-weight: 700;
        border-left: 4px solid #2d6a9f;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2em 0.6em;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-kmeans    { background: #dbeafe; color: #1d4ed8; }
    .badge-hier      { background: #fef3c7; color: #92400e; }
    .badge-gmm       { background: #dcfce7; color: #166534; }

    .stTabs [data-baseweb="tab-list"] button { font-size: 0.9rem; }
    div[data-testid="metric-container"] > label { font-size: 0.78rem; }

    .cluster-item {
        background-color: #f8fafc;
        border-left: 4px solid #2d6a9f;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
    }
    .cluster-id {
        font-weight: bold;
        color: #1e3a5f;
        background: #e2e8f0;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .cluster-name {
        color: #334155;
        font-weight: 500;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# One-time cloud setup
# ---------------------------------------------------------------------------
from app.startup import needs_setup, run_setup, _wikipedia_ready  # noqa: E402

if needs_setup():
    st.info("First run — training clustering models (20 Newsgroups). This takes 2-4 minutes…")
    progress_text = st.empty()
    run_setup(status_callback=lambda msg: progress_text.text(msg))
    progress_text.empty()
    if needs_setup():
        st.error(
            "Training did not complete successfully. "
            "Check that all dependencies are installed and there is enough disk space."
        )
        st.stop()
    st.success("Models ready! Loading dashboard…")
    st.rerun()

# ---------------------------------------------------------------------------
# Constants — absolute paths so the app works from any CWD (Streamlit Cloud)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH   = str(_ROOT / "configs" / "config.yaml")
MODELS_DIR    = str(_ROOT / "models")
PROCESSED_DIR = str(_ROOT / "data" / "processed")
REPORTS_DIR   = str(_ROOT / "outputs" / "reports")
FIGURES_DIR   = str(_ROOT / "outputs" / "figures")

ALGO_LABELS = {"kmeans": "K-Means", "hierarchical": "Hierarchical", "gmm": "GMM"}
ALGO_COLORS = {"kmeans": "#2563eb", "hierarchical": "#d97706", "gmm": "#16a34a"}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading config…")
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading artifacts…")
def load_artifacts(dataset: str):
    vec_path = Path(MODELS_DIR) / f"tfidf_{dataset}.pkl"
    svd_path = Path(MODELS_DIR) / f"svd_{dataset}.pkl"
    if not vec_path.exists():
        return None
    vectorizer   = joblib.load(vec_path)
    svd_pipeline = joblib.load(svd_path)
    # Look for X_reduced in models/ first (committed to git), then data/processed/
    X_reduced = None
    for base in [MODELS_DIR, PROCESSED_DIR]:
        xr = Path(base) / f"X_reduced_{dataset}.npy"
        if xr.exists():
            X_reduced = np.load(xr)
            break
    models = {
        pkl.stem: joblib.load(pkl)
        for pkl in Path(MODELS_DIR).glob(f"*_{dataset}_k*.pkl")
    }
    return {"vectorizer": vectorizer, "svd_pipeline": svd_pipeline,
            "X_reduced": X_reduced, "models": models}


def _x_reduced(dataset: str) -> np.ndarray | None:
    """Load X_reduced from models/ or data/processed/, whichever exists."""
    for base in [MODELS_DIR, PROCESSED_DIR]:
        p = Path(base) / f"X_reduced_{dataset}.npy"
        if p.exists():
            return np.load(p)
    return None


@st.cache_data(show_spinner=False)
def load_labels(dataset: str, algorithm: str, k: int, suffix: str = "") -> np.ndarray | None:
    key  = f"{algorithm}_{dataset}_k{k}{('_' + suffix) if suffix else ''}"
    # Prefer precomputed labels saved alongside the model (in git)
    for base in [MODELS_DIR, PROCESSED_DIR]:
        lp = Path(base) / f"{key}_labels.npy"
        if lp.exists():
            return np.load(lp)
    # Fall back: load model and derive labels
    path = Path(MODELS_DIR) / f"{key}.pkl"
    if not path.exists():
        return None
    model = joblib.load(path)
    if hasattr(model, "labels_"):
        return model.labels_
    # predict() needs X_reduced
    X = _x_reduced(dataset)
    if X is None:
        return None
    return model.predict(X)


@st.cache_data(show_spinner=False)
def load_metrics(dataset: str) -> pd.DataFrame | None:
    path = Path(REPORTS_DIR) / f"clustering_metrics_{dataset}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_corpus(dataset: str) -> list[str] | None:
    import pickle
    path = Path(PROCESSED_DIR) / f"{dataset}_clean.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_cluster_names(dataset: str, algorithm: str, k: int, suffix: str = "") -> dict[int, str]:
    """Compute cluster names — works with X_tfidf if available, falls back to model centroids."""
    from src.evaluation import get_cluster_names

    arts = load_artifacts(dataset)
    if arts is None:
        return {}

    vectorizer   = arts["vectorizer"]
    svd_pipeline = arts["svd_pipeline"]
    feature_names = vectorizer.get_feature_names_out()
    svd = svd_pipeline.named_steps["svd"]

    key = f"{algorithm}_{dataset}_k{k}{('_' + suffix) if suffix else ''}"
    model_path = Path(MODELS_DIR) / f"{key}.pkl"

    # --- Path 1: X_tfidf available → compute centroid means directly ---
    tfidf_path = Path(PROCESSED_DIR) / f"X_tfidf_{dataset}.npz"
    if tfidf_path.exists():
        from scipy.sparse import load_npz
        from src.evaluation import get_top_terms
        labels = load_labels(dataset, algorithm, k, suffix)
        if labels is not None:
            X_tfidf = load_npz(str(tfidf_path))
            X_red   = arts["X_reduced"]
            if X_red is not None:
                _, X_tfidf, _ = _align_to_labels(labels, X_red, X_tfidf_full=X_tfidf)
            top_terms = get_top_terms(labels, vectorizer, X_tfidf, n=5)
            return get_cluster_names(top_terms, n_words=3)

    # --- Path 2: no X_tfidf → use model centroids back-projected through SVD ---
    if not model_path.exists():
        return {}
    model = joblib.load(model_path)

    centroids_svd: np.ndarray | None = None
    if hasattr(model, "cluster_centers_"):      # KMeans
        centroids_svd = model.cluster_centers_
    elif hasattr(model, "means_"):               # GMM
        centroids_svd = model.means_
    else:                                        # Hierarchical — use labels + X_reduced
        labels = load_labels(dataset, algorithm, k, suffix)
        X_red  = arts["X_reduced"]
        if labels is not None and X_red is not None:
            X_aligned, _, _ = _align_to_labels(labels, X_red)
            centroids_svd = np.array([
                X_aligned[labels == i].mean(axis=0)
                for i in range(k)
                if (labels == i).any()
            ])

    if centroids_svd is None or len(centroids_svd) == 0:
        return {}

    centroids_tfidf = svd.inverse_transform(centroids_svd)
    top_terms = {
        i: [feature_names[j] for j in np.argsort(row)[-5:][::-1]]
        for i, row in enumerate(centroids_tfidf)
    }
    return get_cluster_names(top_terms, n_words=3)


def cluster_label(cluster_id: int, names: dict[int, str]) -> str:
    name = names.get(int(cluster_id), "")
    return f"Cluster {cluster_id} — {name}" if name else f"Cluster {cluster_id}"


def _hier_sample_idx(n_total: int) -> np.ndarray | None:
    """Reproduce the subsample indices used when fitting hierarchical on large corpora.

    Returns None when the corpus was small enough to fit without sampling.
    """
    cfg = load_config()
    max_docs = cfg["clustering"]["hierarchical"].get("max_docs", 8000)
    if n_total <= max_docs:
        return None
    rng = np.random.default_rng(cfg["seed"])
    return rng.choice(n_total, max_docs, replace=False)


def _align_to_labels(labels: np.ndarray, X_full: np.ndarray,
                     X_tfidf_full=None, corpus_full=None):
    """When hierarchical labels cover only a subsample, slice X/corpus to match."""
    n_full = X_full.shape[0]
    if len(labels) == n_full:
        return X_full, X_tfidf_full, corpus_full

    idx = _hier_sample_idx(n_full)
    if idx is None or len(idx) != len(labels):
        return X_full, X_tfidf_full, corpus_full

    X_sub      = X_full[idx]
    X_tfidf_sub = X_tfidf_full[idx] if X_tfidf_full is not None else None
    corpus_sub  = [corpus_full[i] for i in idx] if corpus_full is not None else None
    return X_sub, X_tfidf_sub, corpus_sub


# ---------------------------------------------------------------------------
# Helper: best row per algorithm
# ---------------------------------------------------------------------------

def _best_per_algo(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if "silhouette" in sub.columns and sub["silhouette"].notna().any():
            rows.append(sub.loc[sub["silhouette"].idxmax()])
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> dict:
    st.sidebar.title("Controls")

    cfg = load_config()

    wiki_available = _wikipedia_ready()
    dataset_options = ["newsgroups", "wikipedia"] if wiki_available else ["newsgroups"]
    dataset_labels  = {
        "newsgroups": "20 Newsgroups",
        "wikipedia":  "Wikipedia People",
    }
    dataset = st.sidebar.selectbox(
        "Dataset",
        dataset_options,
        format_func=lambda x: dataset_labels[x],
    )
    if not wiki_available:
        st.sidebar.info(
            "Wikipedia dataset requires `people_wiki.csv` (Kaggle). "
            "Run locally with the full dataset to enable it."
        )
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["kmeans", "hierarchical", "gmm"],
        format_func=lambda x: ALGO_LABELS[x],
    )

    if algorithm == "kmeans":
        k_options = cfg["clustering"]["kmeans"]["k_range"]
    elif algorithm == "hierarchical":
        k_options = cfg["clustering"]["hierarchical"]["n_clusters"]
    else:
        k_options = cfg["clustering"]["gmm"]["n_components"]

    k = st.sidebar.selectbox("Number of Clusters (k)", k_options,
                              index=min(1, len(k_options) - 1))

    suffix = ""
    if algorithm == "hierarchical":
        suffix = st.sidebar.selectbox("Linkage", cfg["clustering"]["hierarchical"]["linkage"])
    elif algorithm == "gmm":
        suffix = st.sidebar.selectbox("Covariance Type",
                                       cfg["clustering"]["gmm"]["covariance_type"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Datasets**\n\n"
        "- 20 Newsgroups (18,846 docs, 20 topics)\n"
        "- Wikipedia People (42,786 biographies)"
    )
    st.sidebar.markdown(
        "**Pipeline**\n\n"
        "TF-IDF (10k features, bigrams) → "
        "TruncatedSVD (100d, L2-norm) → "
        "K-Means / Hierarchical / GMM"
    )

    return {"dataset": dataset, "algorithm": algorithm, "k": k, "suffix": suffix}


# ---------------------------------------------------------------------------
# KPI bar
# ---------------------------------------------------------------------------

def render_kpi_bar(dataset: str, arts: dict) -> None:
    df = load_metrics(dataset)
    corpus = load_corpus(dataset)

    n_docs   = len(corpus) if corpus else (arts["X_reduced"].shape[0] if arts["X_reduced"] is not None else 0)
    n_exps   = len(df) if df is not None else 0
    best_sil = df["silhouette"].max() if df is not None and "silhouette" in df.columns else float("nan")
    best_row = df.loc[df["silhouette"].idxmax()] if df is not None and not df.empty else None
    best_lbl = (f"{ALGO_LABELS[best_row['algorithm']]} k={int(best_row['k'])}"
                if best_row is not None else "—")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="label">Documents</div>
            <div class="value">{n_docs:,}</div>
            <div class="delta delta-info">in corpus</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        vocab = len(arts["vectorizer"].vocabulary_) if arts else 0
        st.markdown(f"""<div class="kpi-card">
            <div class="label">Vocabulary</div>
            <div class="value">{vocab:,}</div>
            <div class="delta delta-info">TF-IDF features</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="label">Experiments</div>
            <div class="value">{n_exps}</div>
            <div class="delta delta-info">total runs</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="label">Best Silhouette</div>
            <div class="value">{best_sil:.4f}</div>
            <div class="delta delta-good">higher is better</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <div class="label">Best Model</div>
            <div class="value" style="font-size:1.1rem;padding-top:0.4rem">{best_lbl}</div>
            <div class="delta delta-good">by silhouette</div>
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def tab_overview(dataset: str, arts: dict) -> None:
    df = load_metrics(dataset)
    if df is None:
        st.warning("No metrics found. Run the pipeline first.")
        return

    st.markdown('<div class="section-header">Algorithm Comparison — Best k per Method</div>',
                unsafe_allow_html=True)

    best_df = _best_per_algo(df)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            best_df,
            x="algorithm",
            y="silhouette",
            color="algorithm",
            color_discrete_map=ALGO_COLORS,
            text=best_df["silhouette"].map("{:.4f}".format),
            labels={"algorithm": "Algorithm", "silhouette": "Silhouette Score"},
            title="Best Silhouette Score per Algorithm",
            height=340,
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", yaxis_gridcolor="#f0f0f0",
                          xaxis=dict(tickvals=best_df["algorithm"].tolist(),
                                     ticktext=[ALGO_LABELS[a] for a in best_df["algorithm"]]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            best_df,
            x="algorithm",
            y="davies_bouldin",
            color="algorithm",
            color_discrete_map=ALGO_COLORS,
            text=best_df["davies_bouldin"].map("{:.3f}".format),
            labels={"algorithm": "Algorithm", "davies_bouldin": "Davies-Bouldin Index"},
            title="Best Davies-Bouldin Index (lower = better)",
            height=340,
        )
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2.update_layout(showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", yaxis_gridcolor="#f0f0f0",
                           xaxis=dict(tickvals=best_df["algorithm"].tolist(),
                                      ticktext=[ALGO_LABELS[a] for a in best_df["algorithm"]]))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Silhouette Score vs. k (All Algorithms)</div>',
                unsafe_allow_html=True)

    fig3 = go.Figure()
    for algo, grp in df.groupby("algorithm"):
        g = grp.sort_values("k")
        fig3.add_trace(go.Scatter(
            x=g["k"], y=g["silhouette"],
            mode="lines+markers",
            name=ALGO_LABELS[algo],
            line=dict(color=ALGO_COLORS[algo], width=2),
            marker=dict(size=7),
        ))
    fig3.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        title=f"Silhouette Score vs. k — {dataset}",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_gridcolor="#f0f0f0",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["algorithm", "k", "silhouette", "davies_bouldin",
                                 "calinski_harabasz", "runtime_s"] if c in df.columns]
    styled = (df[display_cols]
              .rename(columns={"algorithm": "Algorithm", "k": "k",
                                "silhouette": "Silhouette ↑",
                                "davies_bouldin": "Davies-Bouldin ↓",
                                "calinski_harabasz": "Calinski-Harabasz ↑",
                                "runtime_s": "Runtime (s)"})
              .style
              .background_gradient(subset=["Silhouette ↑"], cmap="Blues")
              .background_gradient(subset=["Davies-Bouldin ↓"], cmap="Reds_r")
              .format({"Silhouette ↑": "{:.4f}", "Davies-Bouldin ↓": "{:.4f}",
                       "Calinski-Harabasz ↑": "{:.1f}", "Runtime (s)": "{:.1f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Cluster Map
# ---------------------------------------------------------------------------

def tab_cluster_plot(params: dict, arts: dict) -> None:
    labels = load_labels(params["dataset"], params["algorithm"], params["k"], params["suffix"])
    if labels is None:
        st.warning("No saved model found for this configuration.")
        return
    if arts["X_reduced"] is None:
        st.warning("Reduced feature matrix not found.")
        return

    names = load_cluster_names(params["dataset"], params["algorithm"],
                                params["k"], params["suffix"])

    from sklearn.decomposition import PCA
    corpus = load_corpus(params["dataset"])
    X_vis, _, corpus_vis = _align_to_labels(labels, arts["X_reduced"],
                                             corpus_full=corpus)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_vis)

    hover_text = ([c[:120] for c in corpus_vis]
                  if corpus_vis and len(corpus_vis) == len(labels)
                  else [f"doc {i}" for i in range(len(labels))])

    cluster_labels_named = [cluster_label(c, names) for c in labels]

    df_plot = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Cluster": cluster_labels_named, "Preview": hover_text,
    })

    fig = px.scatter(
        df_plot, x="PC1", y="PC2",
        color="Cluster",
        hover_data={"Preview": True, "PC1": False, "PC2": False},
        opacity=0.55,
        title=(f"PCA Projection — {ALGO_LABELS[params['algorithm']]} "
               f"k={params['k']} — {params['dataset']}"),
        height=600,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)
    var = pca.explained_variance_ratio_
    st.caption(f"PC1: {var[0]:.1%} variance explained  |  PC2: {var[1]:.1%} variance explained")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Cluster Names</div>', unsafe_allow_html=True)
        if names:
            for cid, n in sorted(names.items()):
                st.markdown(
                    f'<div class="cluster-item">'
                    f'<span class="cluster-id">{cid}</span>'
                    f'<span class="cluster-name">{n}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No cluster names available.")

    with col2:
        st.markdown('<div class="section-header">Cluster Sizes</div>', unsafe_allow_html=True)
        n_per_cluster = pd.Series(labels).value_counts().sort_index()
        size_df = pd.DataFrame({
            "Cluster ID": [str(c) for c in n_per_cluster.index],
            "Cluster Name": [cluster_label(c, names) for c in n_per_cluster.index],
            "Documents": n_per_cluster.values
        })

        fig_sz = px.bar(
            size_df,
            x="Cluster ID",
            y="Documents",
            color="Cluster Name",
            text="Documents",
            height=340,
            labels={"Cluster ID": "Cluster ID", "Documents": "Number of Documents"},
        )
        fig_sz.update_traces(textposition="outside")
        fig_sz.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_gridcolor="#f0f0f0",
            xaxis=dict(type='category')
        )
        st.plotly_chart(fig_sz, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Cluster Explorer
# ---------------------------------------------------------------------------

def tab_cluster_explorer(params: dict, arts: dict) -> None:
    labels = load_labels(params["dataset"], params["algorithm"], params["k"], params["suffix"])
    if labels is None:
        st.warning("No saved model found.")
        return

    names = load_cluster_names(params["dataset"], params["algorithm"],
                                params["k"], params["suffix"])

    cluster_ids = sorted(set(labels))
    selected_cluster = st.selectbox(
        "Select Cluster",
        cluster_ids,
        format_func=lambda cid: cluster_label(cid, names),
    )

    n_in = int((labels == selected_cluster).sum())
    pct  = n_in / len(labels) * 100
    if names and int(selected_cluster) in names:
        st.info(f"**{cluster_label(selected_cluster, names)}**  ·  "
                f"{n_in:,} documents ({pct:.1f}% of corpus)")

    from scipy.sparse import load_npz
    tfidf_path = Path(PROCESSED_DIR) / f"X_tfidf_{params['dataset']}.npz"

    corpus = load_corpus(params["dataset"])
    _, X_tfidf_aligned, corpus_aligned = _align_to_labels(
        labels, arts["X_reduced"],
        X_tfidf_full=load_npz(str(tfidf_path)) if tfidf_path.exists() else None,
        corpus_full=corpus,
    )

    col1, col2 = st.columns([2, 1])
    if tfidf_path.exists() and X_tfidf_aligned is not None:
        from src.evaluation import get_top_terms
        top_terms = get_top_terms(labels, arts["vectorizer"], X_tfidf_aligned, n=15)
        terms = top_terms.get(int(selected_cluster), [])

        with col1:
            if terms:
                st.markdown("**Top TF-IDF Terms**")
                term_df = pd.DataFrame({
                    "Term": terms[::-1],
                    "Rank": list(range(len(terms), 0, -1)),
                })
                fig = px.bar(
                    term_df, x="Rank", y="Term",
                    orientation="h",
                    color="Rank",
                    color_continuous_scale="Blues",
                    height=420,
                    labels={"Rank": "Importance rank"},
                )
                fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if terms:
                st.markdown("**Term List**")
                for i, t in enumerate(terms, 1):
                    st.write(f"{i}. `{t}`")

    if corpus_aligned and len(corpus_aligned) == len(labels):
        from src.evaluation import get_cluster_examples
        examples = get_cluster_examples(labels, corpus_aligned, n=5, snippet_len=350)
        docs = examples.get(int(selected_cluster), [])
        st.markdown("**Sample Documents**")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"Document {i}"):
                st.write(doc)


# ---------------------------------------------------------------------------
# Tab: Metrics
# ---------------------------------------------------------------------------

def tab_metrics(params: dict) -> None:
    df = load_metrics(params["dataset"])
    if df is None:
        st.warning("No metrics file found.")
        return

    metric_opts = [c for c in ["silhouette", "davies_bouldin", "calinski_harabasz"]
                   if c in df.columns]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**All Experiments**")
        cols_to_show = [c for c in ["algorithm", "k", "silhouette", "davies_bouldin",
                                     "calinski_harabasz", "runtime_s"] if c in df.columns]
        styled = (df[cols_to_show]
                  .style
                  .background_gradient(subset=["silhouette"], cmap="RdYlGn")
                  .format({"silhouette": "{:.4f}", "davies_bouldin": "{:.4f}",
                            "calinski_harabasz": "{:.1f}", "runtime_s": "{:.1f}"}))
        st.dataframe(styled, use_container_width=True, height=380, hide_index=True)

    with col2:
        metric = st.selectbox("Metric to visualise", metric_opts)
        higher = metric != "davies_bouldin"
        note   = "(higher = better)" if higher else "(lower = better)"

        fig = go.Figure()
        for algo, grp in df.groupby("algorithm"):
            g = grp.sort_values("k")
            fig.add_trace(go.Scatter(
                x=g["k"], y=g[metric],
                mode="lines+markers",
                name=ALGO_LABELS[algo],
                line=dict(color=ALGO_COLORS[algo], width=2),
                marker=dict(size=7),
            ))
        fig.update_layout(
            xaxis_title="k",
            yaxis_title=f"{metric.replace('_', ' ').title()} {note}",
            title=f"{metric.replace('_', ' ').title()} vs. k",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_gridcolor="#f0f0f0", height=380,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "silhouette" in df.columns:
        best = df.loc[df["silhouette"].idxmax()]
        st.success(
            f"**Best overall:** {ALGO_LABELS[best['algorithm']]} k={int(best['k'])}  "
            f"— Silhouette = {best['silhouette']:.4f}  "
            f"| Davies-Bouldin = {best['davies_bouldin']:.4f}"
        )

    if "inertia" in df.columns:
        kmeans_df = df[df["algorithm"] == "kmeans"].sort_values("k")
        if not kmeans_df.empty:
            st.markdown("**K-Means Elbow Curve**")
            fig_elbow = px.line(kmeans_df, x="k", y="inertia",
                                markers=True, height=300,
                                labels={"inertia": "Inertia (WSS)", "k": "k"},
                                title="K-Means Inertia vs. k")
            fig_elbow.update_traces(line=dict(color="#2563eb", width=2), marker=dict(size=7))
            fig_elbow.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                    yaxis_gridcolor="#f0f0f0")
            st.plotly_chart(fig_elbow, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Predict
# ---------------------------------------------------------------------------

def tab_predict(params: dict, arts: dict) -> None:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("**Enter document text**")
        user_text = st.text_area(
            "Text",
            height=200,
            label_visibility="collapsed",
            placeholder="Paste any text — article, description, paragraph — and discover which cluster it belongs to.",
        )

        algo    = params["algorithm"]
        k       = params["k"]
        suffix  = params["suffix"]
        key     = f"{algo}_{params['dataset']}_k{k}" + (f"_{suffix}" if suffix else "")

        predict_clicked = st.button("Predict Cluster", type="primary", use_container_width=True)

    with col_result:
        if predict_clicked:
            if not user_text.strip():
                st.warning("Please enter some text.")
                return

            cleaned = preprocess_text(user_text)
            if not cleaned.strip():
                st.error("Text became empty after preprocessing.")
                return

            X_new = transform_new_docs([cleaned], arts["vectorizer"], arts["svd_pipeline"])

            if key not in arts["models"]:
                st.error(f"Model `{key}` not found.")
                return

            model      = arts["models"][key]
            cluster_id = int(model.predict(X_new)[0])
            names      = load_cluster_names(params["dataset"], algo, k, suffix)
            label      = cluster_label(cluster_id, names)

            st.success(f"**Predicted: {label}**")

            if algo == "gmm":
                probs      = model.predict_proba(X_new)[0]
                confidence = float(probs[cluster_id])
                st.metric("GMM Confidence", f"{confidence:.2%}")

                prob_df = pd.DataFrame({
                    "Cluster": [cluster_label(i, names) for i in range(len(probs))],
                    "Probability": probs,
                }).sort_values("Probability", ascending=False)
                fig_probs = px.bar(
                    prob_df.head(10), x="Probability", y="Cluster",
                    orientation="h", height=320,
                    color="Probability", color_continuous_scale="Blues",
                    title="Top-10 Cluster Probabilities",
                )
                fig_probs.update_layout(showlegend=False, coloraxis_showscale=False,
                                         plot_bgcolor="white", paper_bgcolor="white",
                                         yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_probs, use_container_width=True)
            else:
                from scipy.sparse import load_npz
                tfidf_path = Path(PROCESSED_DIR) / f"X_tfidf_{params['dataset']}.npz"
                labels_arr = load_labels(params["dataset"], algo, k, suffix)
                if labels_arr is not None and tfidf_path.exists():
                    X_tfidf_full = load_npz(str(tfidf_path))
                    _, X_tfidf_aligned, _ = _align_to_labels(
                        labels_arr, arts["X_reduced"], X_tfidf_full=X_tfidf_full
                    )
                    from src.evaluation import get_top_terms
                    top_terms = get_top_terms(labels_arr, arts["vectorizer"],
                                              X_tfidf_aligned, n=12)
                    terms = top_terms.get(cluster_id, [])
                    if terms:
                        st.markdown("**Cluster Top Terms:**")
                        st.write("  ".join([f"`{t}`" for t in terms]))

            if names:
                with st.expander("All clusters in this model"):
                    for cid in sorted(names):
                        marker = "👉 " if cid == cluster_id else ""
                        st.write(f"{marker}**{cluster_label(cid, names)}**")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_display = {"newsgroups": "20 Newsgroups", "wikipedia": "Wikipedia People"}

    params = sidebar()
    arts   = load_artifacts(params["dataset"])

    st.markdown(f"""
    <div class="main-header">
        <h1>Document Clustering Explorer</h1>
        <p>Unsupervised clustering of <strong>{dataset_display[params['dataset']]}</strong>
           documents using K-Means, Hierarchical, and Gaussian Mixture Models
           with TF-IDF + LSA features.</p>
    </div>
    """, unsafe_allow_html=True)

    if arts is None:
        if params["dataset"] == "wikipedia":
            st.warning(
                "**Wikipedia dataset not available in this deployment.**\n\n"
                "The Wikipedia People dataset requires `people_wiki.csv` from Kaggle. "
                "Switch to **20 Newsgroups** in the sidebar to explore the available dataset, "
                "or run the app locally with the full Kaggle dataset."
            )
        else:
            st.error(
                f"No trained artifacts found for **{params['dataset']}**.\n\n"
                "Run the pipeline:\n```bash\npython run_pipeline.py\n```"
            )
        st.stop()

    render_kpi_bar(params["dataset"], arts)
    st.markdown("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Cluster Map",
        "Cluster Explorer",
        "Metrics",
        "Predict",
    ])

    with tab1:
        tab_overview(params["dataset"], arts)
    with tab2:
        tab_cluster_plot(params, arts)
    with tab3:
        tab_cluster_explorer(params, arts)
    with tab4:
        tab_metrics(params)
    with tab5:
        tab_predict(params, arts)


if __name__ == "__main__":
    main()
