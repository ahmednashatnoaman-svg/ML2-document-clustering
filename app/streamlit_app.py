"""Streamlit interactive dashboard for Document Clustering Explorer."""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from src.preprocessing import preprocess_text
from src.features import transform_new_docs

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Clustering Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_PATH = "configs/config.yaml"
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/figures"


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
    X_reduced_path = Path(PROCESSED_DIR) / f"X_reduced_{dataset}.npy"

    if not vec_path.exists():
        return None

    vectorizer = joblib.load(vec_path)
    svd_pipeline = joblib.load(svd_path)
    X_reduced = np.load(X_reduced_path) if X_reduced_path.exists() else None

    # Load all saved cluster models for this dataset
    models = {}
    for pkl in Path(MODELS_DIR).glob(f"*_{dataset}_k*.pkl"):
        models[pkl.stem] = joblib.load(pkl)

    return {"vectorizer": vectorizer, "svd_pipeline": svd_pipeline,
            "X_reduced": X_reduced, "models": models}


@st.cache_data(show_spinner="Loading labels…")
def load_labels(dataset: str, algorithm: str, k: int, suffix: str = "") -> np.ndarray | None:
    key = f"{algorithm}_{dataset}_k{k}{('_' + suffix) if suffix else ''}"
    path = Path(MODELS_DIR) / f"{key}.pkl"
    if not path.exists():
        return None
    model = joblib.load(path)
    # AgglomerativeClustering has no predict; labels stored in .labels_
    if hasattr(model, "predict"):
        X = np.load(Path(PROCESSED_DIR) / f"X_reduced_{dataset}.npy")
        return model.predict(X)
    return model.labels_


@st.cache_data(show_spinner="Loading metrics…")
def load_metrics(dataset: str) -> pd.DataFrame | None:
    path = Path(REPORTS_DIR) / f"clustering_metrics_{dataset}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner="Loading corpus…")
def load_corpus(dataset: str) -> list[str] | None:
    import pickle
    path = Path(PROCESSED_DIR) / f"{dataset}_clean.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

def sidebar() -> dict:
    st.sidebar.title("🔧 Controls")
    cfg = load_config()

    dataset = st.sidebar.selectbox(
        "Dataset",
        ["newsgroups", "wikipedia"],
        format_func=lambda x: "20 Newsgroups" if x == "newsgroups" else "Wikipedia People",
    )
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["kmeans", "hierarchical", "gmm"],
        format_func=lambda x: {"kmeans": "K-Means", "hierarchical": "Hierarchical", "gmm": "GMM"}[x],
    )

    # k options depend on algo
    if algorithm == "kmeans":
        k_options = cfg["clustering"]["kmeans"]["k_range"]
    elif algorithm == "hierarchical":
        k_options = cfg["clustering"]["hierarchical"]["n_clusters"]
    else:
        k_options = cfg["clustering"]["gmm"]["n_components"]

    k = st.sidebar.selectbox("Number of Clusters (k)", k_options, index=1)

    suffix = ""
    if algorithm == "hierarchical":
        suffix = st.sidebar.selectbox("Linkage", cfg["clustering"]["hierarchical"]["linkage"])
    elif algorithm == "gmm":
        suffix = st.sidebar.selectbox(
            "Covariance Type", cfg["clustering"]["gmm"]["covariance_type"]
        )

    return {"dataset": dataset, "algorithm": algorithm, "k": k, "suffix": suffix}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_cluster_plot(params: dict, arts: dict):
    st.subheader("Cluster Visualization (PCA 2-D)")

    labels = load_labels(params["dataset"], params["algorithm"], params["k"], params["suffix"])
    if labels is None:
        st.warning("No saved model found for this configuration. Run the pipeline first.")
        return
    if arts["X_reduced"] is None:
        st.warning("Reduced feature matrix not found.")
        return

    from sklearn.decomposition import PCA
    X = arts["X_reduced"]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    corpus = load_corpus(params["dataset"])
    hover_text = (
        [c[:120] for c in corpus] if corpus and len(corpus) == len(labels)
        else [f"doc {i}" for i in range(len(labels))]
    )

    df_plot = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": labels.astype(str),
        "Preview": hover_text,
    })

    fig = px.scatter(
        df_plot, x="PC1", y="PC2",
        color="Cluster",
        hover_data={"Preview": True, "PC1": False, "PC2": False},
        opacity=0.6,
        title=f"PCA — {params['algorithm'].upper()} k={params['k']} — {params['dataset']}",
        height=600,
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    var = pca.explained_variance_ratio_
    st.caption(f"PC1: {var[0]:.1%} variance  |  PC2: {var[1]:.1%} variance")


def tab_cluster_explorer(params: dict, arts: dict):
    st.subheader("Cluster Explorer — Top Terms & Sample Documents")

    labels = load_labels(params["dataset"], params["algorithm"], params["k"], params["suffix"])
    if labels is None:
        st.warning("No saved model found. Run the pipeline first.")
        return

    cluster_ids = sorted(set(labels))
    selected_cluster = st.selectbox("Select Cluster", cluster_ids)

    # Top terms
    from scipy.sparse import load_npz
    tfidf_path = Path(PROCESSED_DIR) / f"X_tfidf_{params['dataset']}.npz"
    if tfidf_path.exists():
        X_tfidf = load_npz(str(tfidf_path))
        from src.evaluation import get_top_terms
        top_terms = get_top_terms(labels, arts["vectorizer"], X_tfidf, n=15)
        terms = top_terms.get(int(selected_cluster), [])

        if terms:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Top Terms**")
                fig = px.bar(
                    x=terms[::-1], y=list(range(len(terms)))[::-1],
                    orientation="h",
                    labels={"x": "Term", "y": "Rank"},
                    height=400,
                )
                fig.update_layout(yaxis=dict(tickvals=list(range(len(terms))),
                                             ticktext=[str(i+1) for i in range(len(terms))]))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Term List**")
                for i, t in enumerate(terms, 1):
                    st.write(f"{i}. `{t}`")

    # Sample documents
    corpus = load_corpus(params["dataset"])
    if corpus and len(corpus) == len(labels):
        from src.evaluation import get_cluster_examples
        examples = get_cluster_examples(labels, corpus, n=5, snippet_len=300)
        docs = examples.get(int(selected_cluster), [])
        st.markdown("**Sample Documents**")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"Document {i}"):
                st.write(doc)

    # Cluster size info
    n_in_cluster = int((labels == selected_cluster).sum())
    st.metric("Documents in this cluster", f"{n_in_cluster:,}", delta=None)


def tab_metrics(params: dict):
    st.subheader("Evaluation Metrics")

    df = load_metrics(params["dataset"])
    if df is None:
        st.warning("No metrics file found. Run the pipeline first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### All Experiments")
        cols_to_show = [c for c in
                        ["algorithm", "k", "silhouette", "davies_bouldin",
                         "calinski_harabasz", "runtime_s"]
                        if c in df.columns]
        styled = df[cols_to_show].style.background_gradient(
            subset=["silhouette"], cmap="RdYlGn"
        ).format({"silhouette": "{:.4f}", "davies_bouldin": "{:.4f}",
                  "calinski_harabasz": "{:.1f}", "runtime_s": "{:.1f}s"})
        st.dataframe(styled, use_container_width=True, height=400)

    with col2:
        metric = st.selectbox(
            "Metric to visualise",
            ["silhouette", "davies_bouldin", "calinski_harabasz"],
        )
        fig = px.line(
            df.sort_values(["algorithm", "k"]),
            x="k", y=metric,
            color="algorithm",
            markers=True,
            title=f"{metric.replace('_', ' ').title()} vs. k",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Best model highlight
    if "silhouette" in df.columns:
        best = df.loc[df["silhouette"].idxmax()]
        st.success(
            f"**Best model:** {best['algorithm'].upper()} k={int(best['k'])}  "
            f"— Silhouette = {best['silhouette']:.4f}"
        )


def tab_predict(params: dict, arts: dict):
    st.subheader("Predict Cluster for New Document")

    user_text = st.text_area(
        "Paste or type your document here:",
        height=180,
        placeholder="Enter any text and discover which cluster it belongs to…",
    )

    if st.button("Predict Cluster", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
            return

        cleaned = preprocess_text(user_text)
        if not cleaned.strip():
            st.error("Text became empty after preprocessing (too many stop words / short).")
            return

        X_new = transform_new_docs([cleaned], arts["vectorizer"], arts["svd_pipeline"])

        # Build model key
        k = params["k"]
        algo = params["algorithm"]
        suffix = params["suffix"]
        key = f"{algo}_{params['dataset']}_k{k}" + (f"_{suffix}" if suffix else "")

        if key not in arts["models"]:
            st.error(f"Model `{key}` not found. Available: {sorted(arts['models'].keys())}")
            return

        model = arts["models"][key]
        cluster_id = int(model.predict(X_new)[0])

        st.success(f"**Predicted Cluster: {cluster_id}**")

        # Show confidence for GMM
        if algo == "gmm":
            probs = model.predict_proba(X_new)[0]
            confidence = float(probs[cluster_id])
            st.metric("Confidence (GMM probability)", f"{confidence:.2%}")

        # Show top terms for that cluster
        from scipy.sparse import load_npz
        tfidf_path = Path(PROCESSED_DIR) / f"X_tfidf_{params['dataset']}.npz"
        if tfidf_path.exists():
            labels = load_labels(params["dataset"], algo, k, suffix)
            if labels is not None:
                X_tfidf = load_npz(str(tfidf_path))
                from src.evaluation import get_top_terms
                top_terms = get_top_terms(labels, arts["vectorizer"], X_tfidf, n=15)
                terms = top_terms.get(cluster_id, [])
                if terms:
                    st.markdown("**Top Terms for this Cluster:**")
                    st.write("  ".join([f"`{t}`" for t in terms]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🔍 Document Clustering Explorer")
    st.markdown(
        "Explore unsupervised clustering of **20 Newsgroups** and **Wikipedia People** "
        "datasets using K-Means, Hierarchical Clustering, and Gaussian Mixture Models."
    )

    params = sidebar()
    arts = load_artifacts(params["dataset"])

    if arts is None:
        st.error(
            f"No trained artifacts found for **{params['dataset']}**.\n\n"
            "Run the pipeline first:\n```bash\npython run_pipeline.py\n```"
        )
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cluster Plot",
        "🔎 Cluster Explorer",
        "📈 Metrics",
        "🧠 Predict",
    ])

    with tab1:
        tab_cluster_plot(params, arts)
    with tab2:
        tab_cluster_explorer(params, arts)
    with tab3:
        tab_metrics(params)
    with tab4:
        tab_predict(params, arts)


if __name__ == "__main__":
    main()
