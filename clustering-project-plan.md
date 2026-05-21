# Unsupervised Document Clustering — Full Implementation Plan

## Goal
Build a production-quality unsupervised document clustering system on two datasets (People Wikipedia + 20 Newsgroups) using K-Means, Hierarchical Clustering, and GMM, with TF-IDF features, full evaluation, visualizations, and a deployed interactive app.

---

## Project Structure (create this first)

```
ML2_final_project/
├── data/
│   ├── raw/                    # original downloaded datasets
│   └── processed/              # cleaned, vectorized outputs
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # text cleaning, tokenization
│   ├── features.py             # TF-IDF + optional embeddings
│   ├── clustering.py           # KMeans, Hierarchical, GMM
│   ├── evaluation.py           # silhouette, top terms, examples
│   └── visualization.py        # PCA, t-SNE, dendrograms
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_evaluation.ipynb
├── app/
│   ├── streamlit_app.py        # interactive dashboard
│   ├── api.py                  # FastAPI REST endpoints
│   └── requirements.txt
├── models/                     # saved model artifacts (joblib)
├── outputs/
│   ├── figures/                # all saved plots
│   └── reports/                # CSVs, JSON metrics
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_clustering.py
├── configs/
│   └── config.yaml             # all hyperparams, seeds, paths
├── requirements.txt
├── README.md
└── report.pdf  (final)
```

---

## Phase 1 — Environment & Project Scaffold

- [ ] **1.1** Create directory structure above
  → Verify: `tree ML2_final_project/` shows all folders

- [ ] **1.2** Create `requirements.txt` with pinned versions:
  ```
  scikit-learn==1.4.x
  numpy==1.26.x
  pandas==2.2.x
  scipy==1.13.x
  matplotlib==3.9.x
  seaborn==0.13.x
  plotly==5.22.x
  streamlit==1.35.x
  fastapi==0.111.x
  uvicorn==0.30.x
  joblib==1.4.x
  nltk==3.8.x
  spacy==3.7.x
  umap-learn==0.5.x
  pyyaml==6.0.x
  ```
  → Verify: `pip install -r requirements.txt` exits 0

- [ ] **1.3** Create `configs/config.yaml`:
  ```yaml
  seed: 42
  datasets:
    newsgroups:
      n_categories: 20
      subset: all
    wikipedia:
      source: people_wiki.csv
  tfidf:
    max_features: 10000
    ngram_range: [1, 2]
    min_df: 5
    max_df: 0.95
  clustering:
    kmeans:
      k_range: [5, 10, 15, 20, 25]
      n_init: 10
      max_iter: 300
    hierarchical:
      n_clusters: [5, 10, 15, 20]
      linkage: [ward, complete, average]
    gmm:
      n_components: [5, 10, 15, 20]
      covariance_type: [full, tied, diag]
  evaluation:
    top_n_terms: 15
    sample_docs_per_cluster: 3
  ```
  → Verify: `python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"` works

---

## Phase 2 — Data Collection & Loading

- [ ] **2.1** Load **20 Newsgroups** dataset:
  ```python
  # src/data_loader.py
  from sklearn.datasets import fetch_20newsgroups
  def load_newsgroups(subset='all', categories=None):
      data = fetch_20newsgroups(subset=subset, categories=categories,
                                 remove=('headers','footers','quotes'),
                                 random_state=42)
      return data.data, data.target, data.target_names
  ```
  → Verify: `len(texts) == 18846`, labels shape matches

- [ ] **2.2** Download **People Wikipedia** dataset:
  - Source: `https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data`
  - OR use `sklearn.datasets` or fetch from Hugging Face `datasets` library
  - Load into `pd.DataFrame` with columns: `[uri, name, text]`
  → Verify: DataFrame has ≥10,000 rows, `text` column non-null > 90%

- [ ] **2.3** Save raw data to `data/raw/` as compressed files
  ```python
  df.to_parquet('data/raw/people_wiki.parquet')
  ```
  → Verify: files exist, reload correctly

---

## Phase 3 — Preprocessing (`src/preprocessing.py`)

All functions accept a list of raw strings and return cleaned strings. Fixed seed=42 throughout.

- [ ] **3.1** Implement `clean_text(text: str) -> str`:
  - Lowercase
  - Remove URLs (`re.sub(r'http\S+', '', text)`)
  - Remove HTML tags
  - Remove non-ASCII characters
  - Remove punctuation / special chars (keep spaces)
  - Collapse whitespace
  → Verify: `clean_text("<b>Hello World!</b> http://x.com")` → `"hello world"`

- [ ] **3.2** Implement `tokenize_and_lemmatize(text: str) -> str`:
  - NLTK word tokenize
  - Remove stopwords (`nltk.corpus.stopwords.words('english')`)
  - Lemmatize with `WordNetLemmatizer`
  - Remove tokens < 3 chars
  → Verify: `"running dogs are faster"` → `"run dog faster"`

- [ ] **3.3** Implement `preprocess_corpus(texts: List[str]) -> List[str]`:
  - Apply clean → tokenize → lemmatize pipeline
  - Filter empty documents after cleaning
  - Log: total docs before/after filtering
  → Verify: No empty strings in output list

- [ ] **3.4** Save processed corpora:
  ```python
  # data/processed/newsgroups_clean.pkl
  # data/processed/wikipedia_clean.pkl
  ```
  → Verify: Reload and check first 5 docs look clean

---

## Phase 4 — Feature Extraction (`src/features.py`)

- [ ] **4.1** Implement `build_tfidf(corpus, config) -> (matrix, vectorizer)`:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  def build_tfidf(corpus, max_features=10000, ngram_range=(1,2),
                  min_df=5, max_df=0.95):
      vec = TfidfVectorizer(max_features=max_features,
                            ngram_range=ngram_range,
                            min_df=min_df, max_df=max_df)
      X = vec.fit_transform(corpus)
      return X, vec
  ```
  → Verify: `X.shape == (n_docs, ≤10000)`, sparse matrix

- [ ] **4.2** Implement `reduce_dimensionality(X, method='svd', n_components=100)`:
  - `method='svd'` → `TruncatedSVD` (for sparse TF-IDF, LSA)
  - `method='pca'` → `PCA` (after densifying)
  - Returns dense array for clustering
  → Verify: Output shape `(n_docs, n_components)`, explained variance logged

- [ ] **4.3** Save artifacts:
  ```python
  joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
  joblib.dump(svd, 'models/svd_reducer.pkl')
  np.save('data/processed/X_reduced.npy', X_reduced)
  ```
  → Verify: Files exist, can be loaded cleanly

- [ ] **4.4** (Optional / Bonus) Implement `build_sentence_embeddings(corpus)`:
  - Use `sentence-transformers` (`all-MiniLM-L6-v2`)
  - Returns `np.ndarray` of shape `(n_docs, 384)`
  → Verify: Shape correct, values in reasonable range

---

## Phase 5 — Clustering (`src/clustering.py`)

Each function returns `labels: np.ndarray` and the fitted model.

- [ ] **5.1** Implement `run_kmeans(X, k, seed=42) -> (labels, model)`:
  ```python
  from sklearn.cluster import KMeans
  def run_kmeans(X, k, seed=42):
      model = KMeans(n_clusters=k, n_init=10, max_iter=300,
                     random_state=seed)
      labels = model.fit_predict(X)
      return labels, model
  ```
  → Verify: `len(set(labels)) == k`, inertia logged

- [ ] **5.2** Implement `run_hierarchical(X, n_clusters, linkage='ward') -> (labels, model)`:
  ```python
  from sklearn.cluster import AgglomerativeClustering
  def run_hierarchical(X, n_clusters, linkage='ward'):
      model = AgglomerativeClustering(n_clusters=n_clusters,
                                      linkage=linkage)
      labels = model.fit_predict(X)
      return labels, model
  ```
  → Verify: `len(set(labels)) == n_clusters`

- [ ] **5.3** Implement `run_gmm(X, n_components, cov_type='full', seed=42) -> (labels, model)`:
  ```python
  from sklearn.mixture import GaussianMixture
  def run_gmm(X, n_components, covariance_type='full', seed=42):
      model = GaussianMixture(n_components=n_components,
                               covariance_type=covariance_type,
                               random_state=seed)
      model.fit(X)
      labels = model.predict(X)
      return labels, model
  ```
  → Verify: `len(set(labels)) == n_components`, BIC/AIC logged

- [ ] **5.4** Implement `run_all_experiments(X, config) -> results_dict`:
  - Loop over all k values for each algorithm
  - Store: `{algo: {k: {'labels': ..., 'model': ..., 'time': ...}}}`
  → Verify: dict has 3 keys (kmeans, hierarchical, gmm), each with all k values

- [ ] **5.5** Save all models:
  ```python
  joblib.dump(model, f'models/kmeans_k{k}.pkl')
  ```
  → Verify: All `.pkl` files in `models/`

---

## Phase 6 — Evaluation (`src/evaluation.py`)

- [ ] **6.1** Implement `silhouette_analysis(X, labels) -> float`:
  ```python
  from sklearn.metrics import silhouette_score
  # Use sample=5000 for large datasets (performance)
  score = silhouette_score(X, labels, sample_size=5000, random_state=42)
  ```
  → Verify: Score in `[-1, 1]`, higher is better

- [ ] **6.2** Implement `get_top_terms(labels, vectorizer, X_tfidf, n=15) -> dict`:
  - For each cluster: compute mean TF-IDF across all docs in cluster
  - Return top-n feature names by mean weight
  → Verify: Each cluster has exactly n terms, terms are real words

- [ ] **6.3** Implement `get_cluster_examples(labels, corpus, n=3) -> dict`:
  - For each cluster: return first n document snippets (first 200 chars)
  → Verify: Each cluster has sample texts

- [ ] **6.4** Implement `evaluate_all(results_dict, X, vectorizer, corpus) -> metrics_df`:
  - Compute silhouette for every (algo, k) combination
  - Return `pd.DataFrame` with columns: `[algorithm, k, silhouette_score, runtime_s]`
  → Verify: DataFrame has rows for all experiments, no NaN values

- [ ] **6.5** Save metrics:
  ```python
  metrics_df.to_csv('outputs/reports/clustering_metrics.csv', index=False)
  ```
  → Verify: CSV exists, readable

- [ ] **6.6** Compute Davies-Bouldin Index and Calinski-Harabasz Score for all experiments:
  ```python
  from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
  ```
  → Verify: Both scores added to `metrics_df`

---

## Phase 7 — Visualization (`src/visualization.py`)

All figures saved to `outputs/figures/` as high-res PNGs (dpi=150).

- [ ] **7.1** Implement `plot_elbow_curve(inertias, k_range, dataset_name)`:
  - Line plot of inertia vs. k for K-Means
  - Mark "elbow" point
  → Verify: `outputs/figures/elbow_newsgroups.png` exists

- [ ] **7.2** Implement `plot_silhouette_comparison(metrics_df, dataset_name)`:
  - Grouped bar chart: silhouette score per (algo, k)
  - Color-coded by algorithm
  → Verify: `outputs/figures/silhouette_comparison_newsgroups.png` exists

- [ ] **7.3** Implement `plot_pca_clusters(X_reduced, labels, algo_name, k, dataset_name)`:
  - Reduce to 2D with PCA
  - Scatter plot colored by cluster label
  - Legend showing cluster IDs
  → Verify: `outputs/figures/pca_kmeans_k10_newsgroups.png` exists

- [ ] **7.4** Implement `plot_tsne_clusters(X_reduced, labels, algo_name, k, dataset_name)`:
  - t-SNE with `perplexity=30, n_iter=1000, random_state=42`
  - Scatter plot colored by cluster label
  → Verify: `outputs/figures/tsne_kmeans_k10_newsgroups.png` exists

- [ ] **7.5** Implement `plot_dendrogram(X_sample, linkage_method, dataset_name)`:
  - Sample 500 docs for readability
  - `scipy.cluster.hierarchy.dendrogram` + `linkage`
  - Truncate at level 5 for clarity
  → Verify: `outputs/figures/dendrogram_ward_newsgroups.png` exists

- [ ] **7.6** Implement `plot_cluster_size_distribution(labels, algo_name, k, dataset_name)`:
  - Bar chart of document count per cluster
  - Flags severely imbalanced clusters (< 1% of total)
  → Verify: `outputs/figures/cluster_sizes_kmeans_k10_newsgroups.png` exists

- [ ] **7.7** Implement `plot_top_terms_heatmap(top_terms_dict, algo_name, k, dataset_name)`:
  - Heatmap: clusters × top_terms, color = TF-IDF weight
  → Verify: `outputs/figures/terms_heatmap_kmeans_k10_newsgroups.png` exists

---

## Phase 8 — Notebooks (EDA + Analysis)

- [ ] **8.1** `notebooks/01_eda.ipynb`:
  - Dataset statistics (n_docs, avg length, vocabulary size)
  - Word frequency distributions
  - Document length histograms
  - Sample documents from each dataset
  → Verify: All cells run without error, at least 6 plots rendered

- [ ] **8.2** `notebooks/02_preprocessing.ipynb`:
  - Before/after preprocessing comparison
  - Token count distribution
  - Most common tokens after cleaning
  → Verify: All cells run cleanly

- [ ] **8.3** `notebooks/03_clustering.ipynb`:
  - Run all experiments using `src/` modules
  - Display elbow curves, silhouette plots
  - Best k selection justification
  → Verify: All cells run, metrics printed

- [ ] **8.4** `notebooks/04_evaluation.ipynb`:
  - Full evaluation results table
  - Top terms per cluster (best model)
  - t-SNE and PCA plots
  - Cluster interpretation narrative
  → Verify: All cells run, plots display inline

---

## Phase 9 — Streamlit App (`app/streamlit_app.py`)

- [ ] **9.1** Build main app layout:
  ```python
  import streamlit as st
  st.title("Document Clustering Explorer")
  # Sidebar: dataset selector, algorithm, k value
  dataset = st.sidebar.selectbox("Dataset", ["20 Newsgroups", "Wikipedia People"])
  algorithm = st.sidebar.selectbox("Algorithm", ["K-Means", "Hierarchical", "GMM"])
  k = st.sidebar.slider("Number of Clusters", 5, 25, 10)
  ```
  → Verify: `streamlit run app/streamlit_app.py` loads without error

- [ ] **9.2** Add PCA/t-SNE cluster plot tab:
  - Load precomputed embeddings from `data/processed/`
  - Plotly interactive scatter (hover shows doc snippet)
  → Verify: Plot renders, hover works

- [ ] **9.3** Add cluster explorer tab:
  - Dropdown: select cluster ID
  - Show top terms (bar chart)
  - Show sample documents
  → Verify: All clusters navigable

- [ ] **9.4** Add metrics comparison tab:
  - Load `outputs/reports/clustering_metrics.csv`
  - Display `pd.DataFrame` with formatting
  - Bar chart: silhouette comparison across algorithms
  → Verify: Table and chart render correctly

- [ ] **9.5** Add "Predict Cluster" tab:
  - Text input area for user document
  - Load saved vectorizer + best model
  - Show predicted cluster + top terms of that cluster
  ```python
  X_new = vectorizer.transform([user_text])
  X_new_reduced = svd.transform(X_new)
  cluster_id = model.predict(X_new_reduced)
  ```
  → Verify: Input text returns a valid cluster ID + terms

---

## Phase 10 — FastAPI REST API (`app/api.py`)

- [ ] **10.1** Implement `GET /health` → `{"status": "ok"}`
  → Verify: `curl localhost:8000/health` returns 200

- [ ] **10.2** Implement `POST /predict`:
  ```python
  class PredictRequest(BaseModel):
      text: str
      dataset: str = "newsgroups"   # or "wikipedia"
      algorithm: str = "kmeans"
      k: int = 10

  @app.post("/predict")
  def predict(req: PredictRequest):
      # load correct model, vectorizer
      # return cluster_id, top_terms, confidence (GMM only)
  ```
  → Verify: `curl -X POST localhost:8000/predict -d '{"text":"..."}'` returns JSON

- [ ] **10.3** Implement `GET /metrics` → returns full metrics CSV as JSON
  → Verify: Response has silhouette scores for all experiments

- [ ] **10.4** Implement `GET /clusters/{dataset}/{algorithm}/{k}`:
  - Returns per-cluster top terms and document counts
  → Verify: Valid JSON with all cluster info

- [ ] **10.5** Add startup model loading (avoid per-request reload):
  ```python
  @app.on_event("startup")
  def load_models():
      app.state.models = load_all_models()
  ```
  → Verify: Models loaded once at startup, fast inference

---

## Phase 11 — Testing (`tests/`)

- [ ] **11.1** `tests/test_preprocessing.py`:
  - Test `clean_text` on edge cases (empty string, HTML, URLs)
  - Test `preprocess_corpus` filters empty docs
  → Verify: `pytest tests/test_preprocessing.py` passes

- [ ] **11.2** `tests/test_features.py`:
  - Test TF-IDF output shape
  - Test vectorizer serialization/deserialization
  → Verify: `pytest tests/test_features.py` passes

- [ ] **11.3** `tests/test_clustering.py`:
  - Test each algorithm produces correct label count
  - Test with tiny synthetic corpus (10 docs, 2 clusters)
  → Verify: `pytest tests/` passes (all green)

---

## Phase 12 — Deployment

- [ ] **12.1** Create `Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  EXPOSE 8501 8000
  CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
  ```
  → Verify: `docker build -t clustering-app .` succeeds

- [ ] **12.2** Deploy Streamlit app to **Hugging Face Spaces**:
  - Create Space at `huggingface.co/spaces`
  - Push repo with `README.md` containing `sdk: streamlit`
  - Include precomputed model artifacts (< 500MB limit)
  → Verify: Space URL loads the app publicly

- [ ] **12.3** (Optional) Deploy FastAPI to **Render** or **Railway**:
  - Add `Procfile`: `web: uvicorn app.api:app --host 0.0.0.0 --port $PORT`
  → Verify: `/health` endpoint returns 200 from public URL

---

## Phase 13 — Report (`report.md` / PDF)

Sections:
1. **Introduction** — Problem statement, datasets, objectives
2. **Methodology** — Preprocessing pipeline, TF-IDF, dimensionality reduction
3. **Experiments** — All algorithms, k values tested, hyperparameters
4. **Results** — Metrics table, silhouette scores comparison
5. **Cluster Analysis** — Top terms per cluster, example docs, interpretation
6. **Visualizations** — Key plots (PCA, t-SNE, dendrograms) with captions
7. **Discussion** — Best algorithm per dataset, failure modes, lessons
8. **Conclusion** — Summary, future work (sentence embeddings, HDBSCAN)

→ Verify: PDF ≥ 8 pages, all figures referenced, metrics cited

---

## Done When

- [ ] Both datasets preprocessed and saved
- [ ] All 3 algorithms run on both datasets for all k values in config
- [ ] Silhouette, Davies-Bouldin, Calinski-Harabasz scores computed and saved
- [ ] PCA, t-SNE, dendrogram plots generated for best model per dataset
- [ ] Streamlit app runs locally and is deployed (public URL)
- [ ] FastAPI `/predict` endpoint works
- [ ] All tests pass (`pytest tests/`)
- [ ] Final report written
- [ ] `README.md` includes: setup instructions, dataset download, how to run app

---

## Critical Path (Sequential)

```
Phase 1 (scaffold) → Phase 2 (data) → Phase 3 (preprocessing)
→ Phase 4 (features) → Phase 5 (clustering) → Phase 6 (evaluation)
→ Phase 7 (viz) → Phase 8 (notebooks) → Phase 9+10 (app+API)
→ Phase 11 (tests) → Phase 12 (deploy) → Phase 13 (report)
```

## Parallelizable

- Phase 9 (Streamlit) & Phase 10 (FastAPI) — independent
- Phase 7 (Visualization) & Phase 8 (Notebooks) — once clustering done
- Tests can be written incrementally alongside each phase

---

## Key Decisions & Defaults

| Decision | Choice | Reason |
|----------|--------|--------|
| Feature baseline | TF-IDF (max 10k, bigrams) | Fast, interpretable, standard |
| Dimensionality reduction | TruncatedSVD (LSA, 100 components) | Works on sparse matrix, fast |
| Best k selection | Highest silhouette score | Most robust unsupervised metric |
| Clustering evaluation | Silhouette + Davies-Bouldin + CH | Complementary perspectives |
| App framework | Streamlit | Fastest for ML dashboards |
| API framework | FastAPI | Type-safe, async, auto-docs |
| Deployment | Hugging Face Spaces | Free, ML-native, easy |
| Seed | 42 everywhere | Reproducibility |
