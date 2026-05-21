# Unsupervised Document Clustering: Final Report

**Course:** Machine Learning 2
**Date:** May 2026

---

## 1. Introduction

This project applies unsupervised clustering to two large text corpora to discover latent topical structure without labeled supervision. Three canonical algorithms — K-Means, Agglomerative Hierarchical Clustering, and Gaussian Mixture Models (GMM) — are evaluated across a range of cluster counts on:

1. **20 Newsgroups** — 18,846 Usenet posts across 20 topic categories (politics, religion, sports, technology, science, etc.)
2. **Wikipedia People** — 42,786 biographical articles from the Kaggle People Wikipedia dataset

The research questions are:
- Which algorithm produces the most coherent document clusters?
- Does cluster structure align with known topical groupings?
- How does corpus size and vocabulary diversity affect clustering quality?

---

## 2. Datasets

### 2.1 20 Newsgroups

Loaded via `sklearn.datasets.fetch_20newsgroups` with `subset='all'` and `remove=('headers', 'footers', 'quotes')` to reduce metadata leakage. The corpus spans 20 categories with roughly equal document counts (~940 per category).

| Statistic | Value |
|-----------|-------|
| Documents | 18,846 |
| Categories | 20 |
| Avg doc length (tokens, raw) | ~240 |
| Avg doc length (tokens, after preprocessing) | ~85 |

### 2.2 Wikipedia People (Kaggle)

Downloaded from [Kaggle: People Wikipedia Data](https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data). The dataset contains full Wikipedia biographical articles (not just summaries) for 59,000 notable figures. After filtering for minimum text length (100 chars), 42,786 articles are retained.

| Statistic | Value |
|-----------|-------|
| Raw rows in CSV | 59,000 |
| After min-length filter | 42,786 |
| Avg text length (chars) | ~1,100 |
| After preprocessing | 42,786 |

The full Wikipedia article text (vs. summaries) provides richer signal, resulting in significantly better cluster separation than shorter biographical snippets.

---

## 3. Methodology

### 3.1 Preprocessing

A three-stage NLTK-based pipeline:

1. **`clean_text`** — Lowercase; strip URLs, HTML tags, email addresses, punctuation, and digits
2. **`tokenize_and_lemmatize`** — WordNet lemmatization with stopword removal (NLTK English list); discard tokens shorter than 3 characters
3. **`preprocess_corpus`** — Apply to all documents; discard documents with fewer than 10 characters after cleaning

### 3.2 Feature Extraction

**TF-IDF Vectorization** with parameters:
- `max_features = 10,000` (top terms by document frequency)
- `ngram_range = (1, 2)` — unigrams and bigrams
- `min_df = 5`, `max_df = 0.95` — remove rare and near-universal terms
- `sublinear_tf = True` — log-scaled term frequency

**Latent Semantic Analysis (LSA)** via `TruncatedSVD`:
- `n_components = 100`
- Followed by L2 normalization

The resulting feature matrix is dense, 100-dimensional, and normalized — well-suited for Euclidean-distance clustering algorithms.

**Vocabulary after filtering:**

| Dataset | Raw Terms | After TF-IDF filtering |
|---------|-----------|----------------------|
| 20 Newsgroups | >500,000 | 10,000 |
| Wikipedia People | >300,000 | 10,000 |

### 3.3 Clustering Algorithms

**K-Means** (`sklearn.cluster.KMeans`):
- k ∈ {5, 10, 15, 20, 25}
- `n_init = 10`, `max_iter = 300`, `random_state = 42`

**Agglomerative Hierarchical Clustering** (`sklearn.cluster.AgglomerativeClustering`):
- n_clusters ∈ {5, 10, 15, 20}
- Linkage ∈ {ward, complete, average}
- Fitted on a 8,000-document subsample for the Wikipedia corpus to avoid O(n²) memory issues

**Gaussian Mixture Model** (`sklearn.mixture.GaussianMixture`):
- n_components ∈ {5, 10, 15, 20}
- covariance_type ∈ {full, tied, diag}
- `max_iter = 100`, `random_state = 42`
- Supports soft assignment (cluster probabilities)

### 3.4 Evaluation Metrics

| Metric | Interpretation | Better |
|--------|----------------|--------|
| **Silhouette Score** | Mean ratio of intra-cluster cohesion vs inter-cluster separation; [-1, 1] | Higher |
| **Davies-Bouldin Index** | Avg similarity between each cluster and its most similar cluster; [0, ∞) | Lower |
| **Calinski-Harabasz Score** | Ratio of between-cluster to within-cluster dispersion | Higher |

Silhouette score is sampled at 5,000 documents for tractability on large corpora.

---

## 4. Experiments & Results

### 4.1 20 Newsgroups

#### K-Means

| k | Silhouette | Davies-Bouldin | Runtime (s) |
|---|-----------|----------------|-------------|
| 5 | 0.0297 | 5.20 | 0.29 |
| 10 | 0.0384 | 4.45 | 0.43 |
| 15 | 0.0488 | 4.02 | 0.51 |
| 20 | 0.0516 | 3.83 | 0.60 |
| **25** | **0.0548** | **3.62** | 0.94 |

K-Means shows consistent monotonic improvement with increasing k.

#### Hierarchical Clustering

| k | Linkage | Silhouette | Davies-Bouldin |
|---|---------|-----------|----------------|
| 20 | ward | **0.0279** | 3.81 |
| 20 | complete | −0.0142 | 5.91 |
| 20 | average | 0.0010 | 3.61 |

Ward linkage dominates. Complete and average produce degenerate solutions on this high-dimensional corpus.

#### GMM

| k | Covariance | Silhouette | Davies-Bouldin |
|---|------------|-----------|----------------|
| 20 | tied | **0.0406** | 3.63 |
| 20 | diag | 0.0335 | 4.20 |
| 20 | full | 0.0287 | 4.85 |

GMM with tied covariance outperforms full covariance because full-covariance GMM's parameter count overwhelms the data at 100 dimensions.

#### Best Per Algorithm (Newsgroups)

| Algorithm | Best k | Silhouette | Davies-Bouldin |
|-----------|--------|-----------|----------------|
| **K-Means** | 25 | **0.0548** | 3.62 |
| GMM (tied) | 20 | 0.0406 | 3.63 |
| Hierarchical (ward) | 20 | 0.0279 | 3.81 |

---

### 4.2 Wikipedia People (42,786 articles)

#### K-Means

| k | Silhouette | Davies-Bouldin | Runtime (s) |
|---|-----------|----------------|-------------|
| 5 | 0.0775 | 3.19 | 3.1 |
| 10 | 0.0928 | 2.93 | 5.2 |
| 15 | 0.1099 | 2.65 | 6.8 |
| **20** | **0.1192** | **2.59** | 8.1 |
| 25 | 0.1146 | 2.67 | 11.3 |

Higher silhouette scores than Newsgroups, reflecting cleaner topical structure in full biographical articles.

#### Hierarchical Clustering (fitted on 8,000-doc subsample)

| k | Linkage | Silhouette | Davies-Bouldin |
|---|---------|-----------|----------------|
| 20 | ward | **0.0970** | 2.75 |
| 15 | ward | 0.0849 | 3.07 |
| 20 | complete | 0.0253 | 3.85 |
| 20 | average | 0.0746 | 2.54 |

Ward linkage again dominates. The subsample approach is necessary to prevent out-of-memory errors on the full 42,786-document corpus.

#### GMM

| k | Covariance | Silhouette | Davies-Bouldin |
|---|------------|-----------|----------------|
| 20 | tied | **0.1068** | 2.66 |
| 20 | diag | 0.0957 | 2.68 |
| 20 | full | 0.0618 | 3.74 |

#### Best Per Algorithm (Wikipedia)

| Algorithm | Best k | Silhouette | Davies-Bouldin |
|-----------|--------|-----------|----------------|
| **K-Means** | 20 | **0.1192** | 2.59 |
| GMM (tied) | 20 | 0.1068 | 2.66 |
| Hierarchical (ward) | 20 | 0.0970 | 2.75 |

**K-Means wins** on both datasets.

---

## 5. Qualitative Cluster Analysis

### 5.1 20 Newsgroups — K-Means k=20

Inspecting top TF-IDF terms per cluster reveals coherent topical groupings:

| Cluster | Top Terms | Inferred Topic |
|---------|-----------|----------------|
| 0 | sale / offer / shipping | Marketplace |
| 2 | team / player / season | Sports |
| 4 | space / orbit / launch | Space / Astronomy |
| 5 | key / clipper / chip | Cryptography |
| 7 | gun / fbi / law | Firearms / Crime |
| 12 | god / christian / jesus | Religion |
| 14 | israel / arab / jew | Middle East Politics |
| 19 | drive / scsi / disk | Hardware / Storage |

### 5.2 Wikipedia People — K-Means k=20

| Cluster | Top Terms | Inferred Domain |
|---------|-----------|----------------|
| 0 | rugby / hockey / season | Team Sports |
| 1 | art / museum / artist | Visual Arts |
| 2 | music / orchestra / composer | Classical Music |
| 5 | album / band / music | Popular Music |
| 6 | baseball / league / major league | Baseball |
| 9 | university / research / professor | Academia |
| 10 | party / election / minister | Politics |
| 12 | film / film festival / directed | Cinema |
| 15 | book / published / novel | Literature |

The biographical clusters map naturally to professional domains, confirming strong topical signal in full Wikipedia article text.

---

## 6. Visualizations

All visualizations are saved to `outputs/figures/`. Key figures:

| Figure | Description |
|--------|-------------|
| `elbow_{dataset}.png` | K-Means inertia vs k — inertia decreases smoothly (no sharp elbow) |
| `pca_kmeans_k20_wikipedia.png` | Clear cluster separation in PCA 2D space |
| `silhouette_comparison_{dataset}.png` | K-Means consistently outperforms other algorithms |
| `dendrogram_ward_{dataset}.png` | Ward-linkage dendrogram on 500-doc sample |
| `davies_bouldin_by_k_{dataset}.png` | DB index vs k for all algorithms |
| `terms_heatmap_kmeans_k20_{dataset}.png` | TF-IDF weight heatmap: clusters × top terms |
| `tsne_kmeans_k20_wikipedia.png` | t-SNE projection showing distinct cluster boundaries |

---

## 7. Computational Analysis

### Runtime Summary

| Algorithm | Dataset | Corpus Size | Typical Runtime |
|-----------|---------|-------------|----------------|
| K-Means | Newsgroups | 18,846 | 0.3–0.9 s/run |
| K-Means | Wikipedia | 42,786 | 3–11 s/run |
| Hierarchical (ward) | Newsgroups | 18,846 | 5.5–6.5 s/run |
| Hierarchical (ward) | Wikipedia | 8,000 (sampled) | 15–25 s/run |
| GMM (full) | Newsgroups | 18,846 | 1.3–5.5 s/run |
| GMM (full) | Wikipedia | 42,786 | 30–80 s/run |

K-Means is 6–15× faster than hierarchical clustering. The O(n²) memory requirement of `AgglomerativeClustering` makes it impractical for corpora >10,000 documents without subsampling.

---

## 8. Discussion

### Why K-Means Wins

1. **LSA normalization** — L2-normalized embeddings make cosine similarity equivalent to Euclidean distance, which K-Means optimizes directly.
2. **Scalability** — K-Means scales as O(n·k·d·iterations), while hierarchical is O(n²) in memory.
3. **Spherical clusters** — After SVD + L2 normalization, document clusters in LSA space are approximately spherical, matching K-Means assumptions.

### Why Hierarchical (Ward) > Complete/Average

Ward linkage minimizes within-cluster variance at each merge step. Complete and average linkage are sensitive to outliers and produce chain-like clusters in high-dimensional spaces ("chaining effect").

### Why Wikipedia Scores Higher Than Newsgroups

1. **Richer content** — Full article text (avg ~1,100 chars) vs. Usenet discussion threads that meander across topics
2. **Domain specificity** — Biographical articles concentrate domain vocabulary (athlete + sport + tournament vs. general-language Usenet)
3. **Less noise** — Wikipedia articles have structured prose; Usenet posts contain quoted replies, signatures, and metadata (partially removed)

---

## 9. API and Dashboard

### FastAPI REST API

`app/api.py` exposes:
- `POST /predict` — Cluster assignment for any input text
- `GET /metrics/{dataset}` — Full experiment metrics table
- `GET /metrics/{dataset}/best` — Best model by any metric
- `GET /health` — Liveness check

### Streamlit Dashboard (5 tabs)

| Tab | Content |
|-----|---------|
| Overview | KPI cards, algorithm comparison bar charts, metrics table |
| Cluster Map | Interactive PCA 2D scatter with hover tooltips and cluster-size bar |
| Cluster Explorer | Top-15 TF-IDF terms, sample documents per cluster |
| Metrics | Conditional-formatted table, trend lines, elbow curve |
| Predict | Free-text input → cluster assignment + GMM class probabilities |

---

## 10. Conclusion

Across both datasets, **K-Means is the best-performing algorithm** for text clustering in the TF-IDF + LSA pipeline:

- Newsgroups: Silhouette = 0.0548 at k=25
- Wikipedia: Silhouette = 0.1192 at k=20

Key findings:

1. Full Wikipedia biographical text (42,786 articles from Kaggle) clusters significantly better than short summaries, with silhouette scores roughly 2× higher.
2. Ward linkage is the only viable hierarchical strategy for high-dimensional text; other linkages produce degenerate solutions.
3. GMM with tied covariance is competitive with K-Means on Wikipedia but underperforms on Newsgroups due to parameter over-specification.
4. Qualitative cluster analysis confirms that top TF-IDF terms clearly identify cluster themes, validating the unsupervised pipeline for document organization.
5. The O(n²) memory constraint of hierarchical clustering requires subsampling for corpora >10k documents, limiting its scalability.

---

## Appendix: Reproducibility

All experiments are fully reproducible with `random_state = 42` throughout.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download Wikipedia dataset from Kaggle into data/raw/people_wiki.csv
python run_pipeline.py
pytest tests/ -v
streamlit run app/streamlit_app.py
```

Results are saved to:
- `outputs/reports/clustering_metrics_{dataset}.csv`
- `outputs/figures/*.png`
- `models/*.pkl`
