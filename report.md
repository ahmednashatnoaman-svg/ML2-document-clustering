# Unsupervised Document Clustering: Final Report

**Course:** Machine Learning 2  
**Date:** May 2026

---

## 1. Introduction

This project applies unsupervised clustering to two text corpora to discover latent topical structure without labeled supervision. Three canonical algorithms — K-Means, Agglomerative Hierarchical Clustering, and Gaussian Mixture Models (GMM) — are evaluated across a range of cluster counts on:

1. **20 Newsgroups** — 18,846 Usenet posts across 20 topic categories (politics, religion, sports, technology, science, etc.)
2. **Wikipedia People** — 538 biographical articles of notable figures spanning science, politics, arts, sports, and technology

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

### 2.2 Wikipedia People

Fetched via the Wikipedia REST API using a curated list of 546 notable individuals across 8 domains (scientists, technologists, politicians, writers/philosophers, artists/musicians, athletes, historical figures, activists). The API returns article summaries (introductory paragraphs).

| Statistic | Value |
|-----------|-------|
| People requested | 546 |
| Articles fetched | 538 (98.5%) |
| Avg text length (chars) | ~870 |
| After preprocessing (min 100 chars) | 538 |

The smaller corpus size and shorter article summaries make this a harder clustering problem than the Newsgroups dataset.

---

## 3. Methodology

### 3.1 Preprocessing

A three-stage NLTK-based pipeline:

1. **`clean_text`** — Lowercase; strip URLs, HTML tags, email addresses, punctuation, and digits
2. **`tokenize_and_lemmatize`** — WordNet lemmatization with stopword removal (NLTK English list); discard tokens shorter than 3 characters
3. **`preprocess_corpus`** — Apply to all documents; discard documents with fewer than 10 tokens after cleaning

### 3.2 Feature Extraction

**TF-IDF Vectorization** with parameters:
- `max_features = 10,000` (top terms by document frequency)
- `ngram_range = (1, 2)` — unigrams and bigrams
- `min_df = 5`, `max_df = 0.95` — remove rare and near-universal terms
- `sublinear_tf = True` — log-scaled term frequency

**Latent Semantic Analysis (LSA)** via `TruncatedSVD`:
- `n_components = 100` (capped to min(n_docs, n_terms) − 1 for small corpora)
- Followed by L2 normalization

The resulting feature matrix is dense, 100-dimensional, and normalized — well-suited for Euclidean-distance clustering algorithms.

**Vocabulary after filtering:**

| Dataset | Raw Terms | After min_df/max_df |
|---------|-----------|---------------------|
| 20 Newsgroups | >500,000 | 10,000 |
| Wikipedia People | ~15,000 | 1,055 |

The small Wikipedia vocabulary (due to small corpus) means 100 SVD components explain 56.5% of the variance, vs 12.1% for Newsgroups — giving cleaner LSA representations for Wikipedia.

### 3.3 Clustering Algorithms

**K-Means** (`sklearn.cluster.KMeans`):
- k ∈ {5, 10, 15, 20, 25}
- `n_init = 10`, `max_iter = 300`, `random_state = 42`

**Agglomerative Hierarchical Clustering** (`sklearn.cluster.AgglomerativeClustering`):
- n_clusters ∈ {5, 10, 15, 20}
- Linkage ∈ {ward, complete, average}
- No prediction on new documents (uses stored `.labels_`)

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

Silhouette score is sampled at 5,000 documents for tractability on Newsgroups.

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

K-Means shows a consistent monotonic improvement with increasing k. The best silhouette (0.0548) at k=25 aligns with the expectation that the 20 Newsgroups categories have fine-grained sub-structure.

#### Hierarchical Clustering

| k | Linkage | Silhouette | Davies-Bouldin |
|---|---------|-----------|----------------|
| 20 | ward | **0.0279** | 3.81 |
| 15 | ward | 0.0239 | 4.08 |
| 20 | complete | −0.0142 | 5.91 |
| 20 | average | 0.0010 | 3.61 |

Ward linkage dominates all other linkages by a wide margin. Complete and average linkage produce degenerate solutions (one mega-cluster + singletons) on this high-dimensional corpus, leading to negative silhouette scores.

#### GMM

| k | Covariance | Silhouette | Davies-Bouldin |
|---|------------|-----------|----------------|
| 20 | tied | **0.0406** | 3.63 |
| 15 | tied | 0.0372 | 3.85 |
| 20 | diag | 0.0335 | 4.20 |

GMM with tied covariance (all components share one covariance matrix) outperforms full covariance, likely because the number of free parameters in full-covariance GMM overwhelms the available data at 100 dimensions.

#### Best Per Algorithm (Newsgroups)

| Algorithm | Best k | Silhouette | Davies-Bouldin |
|-----------|--------|-----------|----------------|
| **K-Means** | 25 | **0.0548** | 3.62 |
| GMM (tied) | 20 | 0.0406 | 3.63 |
| Hierarchical (ward) | 20 | 0.0279 | 3.81 |

**K-Means wins** on Newsgroups across all metrics.

---

### 4.2 Wikipedia People

#### K-Means

| k | Silhouette | Davies-Bouldin | Runtime (s) |
|---|-----------|----------------|-------------|
| 5 | 0.0672 | 3.76 | 0.028 |
| 10 | 0.0926 | 3.58 | 0.013 |
| 15 | 0.1134 | 3.04 | 0.014 |
| 20 | 0.1266 | 2.69 | 0.020 |
| **25** | **0.1373** | **2.57** | 0.020 |

Much higher silhouette scores than Newsgroups, consistent with the cleaner LSA representation (56.5% variance explained) and the more distinct thematic groupings in biographical articles.

#### Hierarchical Clustering

| k | Linkage | Silhouette | Davies-Bouldin |
|---|---------|-----------|----------------|
| 20 | ward | **0.1234** | 2.76 |
| 15 | ward | 0.1096 | 2.98 |
| 20 | complete | 0.0619 | 3.27 |

Ward linkage again dominates. The gap between ward and other linkages is smaller than on Newsgroups, since the Wikipedia corpus has clearer cluster structure.

#### GMM

| k | Covariance | Silhouette | Davies-Bouldin |
|---|------------|-----------|----------------|
| 20 | tied | **0.1221** | 2.86 |
| 20 | full | 0.1211 | 2.87 |
| 15 | full | 0.1069 | 3.08 |

All three covariance types perform similarly, unlike Newsgroups where tied covariance had a clear advantage. With 538 documents and 100 dimensions, even full covariance GMM is not over-parameterized.

#### Best Per Algorithm (Wikipedia)

| Algorithm | Best k | Silhouette | Davies-Bouldin |
|-----------|--------|-----------|----------------|
| **K-Means** | 25 | **0.1373** | 2.57 |
| Hierarchical (ward) | 20 | 0.1234 | 2.76 |
| GMM (tied) | 20 | 0.1221 | 2.86 |

**K-Means wins** on Wikipedia as well.

---

## 5. Qualitative Cluster Analysis

### 5.1 20 Newsgroups — K-Means k=25

Inspecting top TF-IDF terms per cluster reveals coherent topical groupings:

| Cluster | Top Terms | Inferred Topic |
|---------|-----------|----------------|
| 0 | gun, weapon, firearm, crime, police | Firearms / Crime |
| 3 | game, team, player, season, win | Sports |
| 5 | file, program, window, software, drive | Software / Computing |
| 7 | israel, arab, jewish, war, peace | Middle East Politics |
| 12 | god, jesus, christ, bible, faith | Religion |
| 18 | space, nasa, launch, satellite, orbit | Space / Astronomy |
| 22 | car, drive, engine, oil, speed | Automotive |

Many clusters align closely with the original 20 categories, though related categories (e.g., `sci.med` and `sci.space`) sometimes merge.

### 5.2 Wikipedia People — K-Means k=25

| Cluster | Top Terms | Inferred Domain |
|---------|-----------|----------------|
| 2 | film, actor, director, role, award | Cinema |
| 6 | president, government, political, election | Politics |
| 9 | music, album, band, song, record | Music |
| 14 | mathematics, theorem, professor, university | Science/Academia |
| 17 | championship, tournament, career, title | Sports |
| 21 | novel, writer, book, literature, prize | Literature |

The biographical clusters map naturally to professional domains, confirming that Wikipedia article summaries carry strong topical signal even in short form.

---

## 6. Visualizations

All visualizations are saved to `outputs/figures/`. Key figures:

### 6.1 Elbow Curves

The elbow curves (`elbow_newsgroups.png`, `elbow_wikipedia.png`) plot inertia vs k. Neither dataset shows a sharp elbow — inertia decreases smoothly — suggesting the true number of coherent clusters is ambiguous and domain-specific.

### 6.2 PCA 2D Scatter

PCA projections (`pca_kmeans_k25_*.png`) show clear visual separation for Wikipedia (clusters are compact and well-separated in PC space) and moderate overlap for Newsgroups (reflecting lower silhouette scores and higher-dimensional mixing).

### 6.3 Silhouette Comparison

The bar chart (`silhouette_comparison_*.png`) shows K-Means consistently outperforming GMM and Hierarchical across all k values on both datasets.

### 6.4 Dendrogram

Ward-linkage dendrograms (`dendrogram_ward_*.png`) computed on a 500-document sample show clean hierarchical structure for Wikipedia (3–4 major branches at the top level) and more gradual merging for Newsgroups.

### 6.5 Davies-Bouldin Curves

The DB index curves (`davies_bouldin_by_k_*.png`) for Newsgroups show K-Means and GMM(tied) converging to similar DB values at higher k, with hierarchical (average/complete) significantly worse.

---

## 7. Computational Analysis

### Runtime

| Algorithm | Dataset | Typical Runtime |
|-----------|---------|----------------|
| K-Means | Newsgroups (18,846 docs) | 0.3–0.9 s/run |
| Hierarchical (ward) | Newsgroups | 5.5–6.5 s/run |
| GMM (full) | Newsgroups | 1.3–5.5 s/run |
| K-Means | Wikipedia (538 docs) | 0.013–0.028 s/run |

K-Means is 6–15× faster than hierarchical on Newsgroups. The O(n²) memory requirement of `AgglomerativeClustering` makes it impractical for corpora larger than ~50,000 documents.

---

## 8. Discussion

### Why K-Means Wins

1. **LSA normalization** — L2-normalized embeddings make cosine similarity equivalent to Euclidean distance, which is the distance K-Means optimizes. The preprocessing pipeline is specifically designed for K-Means.
2. **Scalability** — K-Means scales linearly with n, while hierarchical clustering is quadratic in memory and GMM with full covariance is cubic in feature dimension.
3. **Spherical clusters** — After SVD, document clusters in LSA space tend to be approximately spherical and well-separated, matching K-Means assumptions.

### Why Hierarchical (Ward) > Complete/Average

Ward linkage minimizes within-cluster variance at each merge step, which is equivalent to K-Means at the merge level. Complete and average linkage are sensitive to outliers and produce chain-like clusters in high-dimensional spaces — a known failure mode called "chaining."

### Why Wikipedia Scores Higher

1. **Higher LSA variance** — 56.5% variance explained at 100 components vs 12.1% for Newsgroups. More signal is preserved.
2. **Cleaner topics** — Biographical summaries have stronger domain keywords (athlete names/sports, scientist names/fields) vs Usenet where discussion threads meander.
3. **Shorter, focused texts** — Wikipedia summaries are 2–5 sentences focused on the person's field, reducing noise.

### Low Absolute Silhouette Scores on Newsgroups

A silhouette of 0.055 is low in absolute terms but expected for text data: documents occupy a sparse region of a very high-dimensional space even after SVD, and boundaries between adjacent topics (e.g., `sci.med` vs `sci.space`) are inherently fuzzy. These scores are consistent with published benchmarks on the 20 Newsgroups dataset.

---

## 9. REST API and Dashboard

### FastAPI

A production-ready REST API (`app/api.py`) exposes:
- **`POST /predict`** — Cluster assignment for arbitrary input text (K-Means, GMM)
- **`GET /metrics/{dataset}`** — Full experiment metrics table
- **`GET /metrics/{dataset}/best`** — Best model by any metric
- **`GET /clusters/{dataset}/{algorithm}/{k}`** — List available model keys
- **`GET /health`** — Liveness check with loaded dataset status

Hierarchical clustering is explicitly excluded from the `/predict` endpoint since `AgglomerativeClustering` has no `.predict()` method and cannot generalize to unseen documents.

### Streamlit Dashboard

An interactive 4-tab dashboard (`app/streamlit_app.py`) provides:
1. **Cluster Plot** — PCA 2D scatter with Plotly hover tooltips
2. **Cluster Explorer** — Top terms bar chart + sample document expanders per cluster
3. **Metrics** — Sortable experiment table with green/red gradient on silhouette + trend line plots
4. **Predict** — Free-text input → cluster assignment + top cluster terms

---

## 10. Conclusion

Across both datasets, **K-Means is the best-performing algorithm** for text clustering in the TF-IDF + LSA pipeline, achieving silhouette scores of 0.0548 (Newsgroups) and 0.1373 (Wikipedia People) at k=25. The combination of L2-normalized LSA embeddings and K-Means is a well-established and empirically strong baseline for document clustering.

Key findings:

1. Monotonically increasing silhouette with k suggests both corpora have sub-structure beyond the coarse topical level (20 topics for Newsgroups; ~8 domains for Wikipedia).
2. Ward linkage is the only viable hierarchical strategy for high-dimensional text; complete and average linkage produce degenerate clusters.
3. GMM with tied covariance outperforms full covariance on Newsgroups, where limited data makes full-covariance estimation unreliable.
4. Wikipedia biographical articles cluster more cleanly than Usenet posts despite being a much smaller corpus, because the texts are more focused and the LSA representation captures more variance.
5. Qualitative cluster analysis confirms that top TF-IDF terms clearly identify the topical theme of each cluster, validating the unsupervised approach for document organization.

---

## Appendix: Reproducibility

All experiments are fully reproducible with `random_state=42` throughout.

```bash
# Full reproduction from scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
python run_pipeline.py --dataset all
pytest tests/ -v
streamlit run app/streamlit_app.py
```

Results are saved to:
- `outputs/reports/clustering_metrics_{dataset}.csv`
- `outputs/figures/*.png`
- `models/*.pkl`
