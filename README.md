---
title: Document Clustering Explorer
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Unsupervised Document Clustering

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io/)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/AhmedNashat1/document-clustering-explorer)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-52%20passed-brightgreen)](#testing)

An end-to-end machine learning project that applies **unsupervised clustering** to two large document corpora — **20 Newsgroups** (18,846 posts) and **Wikipedia People** (42,786 biographies) — using TF-IDF + LSA feature extraction and three clustering algorithms.

---

## Live Demo

**[Try the interactive dashboard on Hugging Face Spaces](https://huggingface.co/spaces/AhmedNashat1/document-clustering-explorer)**

The dashboard lets you explore clusters interactively, compare algorithm performance, inspect top terms per cluster, and predict which cluster any new text belongs to.

---

## Key Features

- **Three clustering algorithms**: K-Means, Agglomerative Hierarchical, Gaussian Mixture Models
- **Two large datasets**: 20 Newsgroups (sklearn) and Wikipedia People (Kaggle, 42k articles)
- **Full NLP pipeline**: text cleaning → lemmatization → TF-IDF (10k features, bigrams) → TruncatedSVD/LSA (100d) → L2 normalization
- **Automatic cluster naming**: each cluster labeled with its top 3 TF-IDF terms (e.g., "music / orchestra / composer")
- **Interactive Streamlit dashboard**: 5-tab KPI dashboard with PCA scatter, top-terms bar charts, metrics comparison, and live prediction
- **FastAPI REST endpoint**: `/predict` endpoint for integrating into other applications
- **52 automated tests** across preprocessing, features, and clustering
- **Reproducible results**: fixed seeds, config-driven hyperparameters

---

## Datasets

| Dataset | Source | Documents | Description |
|---|---|---|---|
| 20 Newsgroups | `sklearn.datasets` | 18,846 | Usenet posts across 20 topic categories |
| Wikipedia People | [Kaggle](https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data) | 42,786 | Biographical Wikipedia articles |

### Downloading the Wikipedia Dataset

The Newsgroups dataset downloads automatically from scikit-learn. For Wikipedia People:

1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` token in `~/.kaggle/`
3. Download the dataset:

```bash
kaggle datasets download sameersmahajan/people-wikipedia-data -p data/raw/ --unzip
```

The file `data/raw/people_wiki.csv` should contain columns: `URI`, `name`, `text`.

If the CSV is not found, the pipeline falls back to auto-fetching articles via the Wikipedia REST API.

---

## Project Structure

```
.
├── app/
│   ├── streamlit_app.py       # Interactive Streamlit dashboard (5 tabs)
│   ├── api.py                 # FastAPI REST endpoint
│   └── startup.py             # Auto-training for HF Spaces first boot
├── configs/
│   └── config.yaml            # All hyperparameters, paths, seeds
├── src/
│   ├── data_loader.py         # Load newsgroups + Wikipedia datasets
│   ├── preprocessing.py       # Clean → tokenize → lemmatize pipeline
│   ├── features.py            # TF-IDF + TruncatedSVD/LSA
│   ├── clustering.py          # K-Means, Hierarchical, GMM runners
│   ├── evaluation.py          # Silhouette, DB, CH metrics + top terms
│   └── visualization.py      # Matplotlib: PCA/t-SNE, elbow, heatmaps
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_clustering.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_evaluation.ipynb
├── run_pipeline.py            # End-to-end CLI pipeline runner
├── Dockerfile                 # Docker image for HF Spaces deployment
├── requirements.txt
└── packages.txt               # System deps for Docker build
```

---

## Architecture

```
Raw Text
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  PREPROCESSING  (src/preprocessing.py)               │
│  lowercase → strip URLs/HTML/email → remove punct    │
│  → NLTK tokenize → stopword removal → lemmatize      │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION  (src/features.py)               │
│  TF-IDF  max_features=10k  ngram=(1,2)               │
│  → TruncatedSVD (LSA)  n_components=100              │
│  → L2 Normalization                                  │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────┬─────────────────┬────────────────┐
│    K-MEANS       │  HIERARCHICAL   │      GMM       │
│  k ∈ [5…25]      │  k ∈ [5…20]     │  k ∈ [5…20]   │
│  n_init=10       │  ward/complete/ │  full/tied/    │
│                  │  average        │  diag          │
└──────────────────┴─────────────────┴────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  EVALUATION  (src/evaluation.py)                     │
│  Silhouette Score  |  Davies-Bouldin  |  CH Score    │
│  Top TF-IDF terms per cluster                        │
│  Automatic cluster naming (top-3 terms)              │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  VISUALIZATION + DASHBOARD  (src/visualization.py)   │
│  PCA scatter  |  t-SNE  |  Elbow  |  Dendrogram     │
│  Streamlit interactive dashboard  |  FastAPI REST    │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- 4 GB RAM (8 GB recommended for the full Wikipedia pipeline)

### 1. Clone and set up environment

```bash
git clone https://github.com/ahmednashatnoaman-svg/ML2-document-clustering.git
cd ML2-document-clustering
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Download Wikipedia dataset

```bash
kaggle datasets download sameersmahajan/people-wikipedia-data -p data/raw/ --unzip
```

### 3. Run the full pipeline

```bash
# Both datasets, all steps
python run_pipeline.py

# One dataset only
python run_pipeline.py --dataset newsgroups

# Skip slow steps if artifacts already exist
python run_pipeline.py --dataset wikipedia --steps cluster evaluate visualize
```

Pipeline steps: `load → preprocess → features → cluster → evaluate → visualize`

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

### 5. (Optional) Start the FastAPI REST endpoint

```bash
uvicorn app.api:app --reload --port 8000
```

Predict endpoint: `POST http://localhost:8000/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "She is a concert pianist who performed at Carnegie Hall.", "dataset": "wikipedia", "algorithm": "kmeans", "k": 20}'
```

---

## Configuration

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml):

```yaml
seed: 42

tfidf:
  max_features: 10000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.95
  sublinear_tf: true

svd:
  n_components: 100

clustering:
  kmeans:
    k_range: [5, 10, 15, 20, 25]
    n_init: 10
    max_iter: 300
  hierarchical:
    n_clusters: [5, 10, 15, 20]
    linkage: [ward, complete, average]
    max_docs: 8000        # subsample limit for O(n²) memory safety
  gmm:
    n_components: [5, 10, 15, 20]
    covariance_type: [full, tied, diag]
    max_iter: 100
```

---

## Results

### 20 Newsgroups

| Algorithm | Best k | Silhouette | Davies-Bouldin |
|---|---|---|---|
| K-Means | 25 | 0.0548 | 5.01 |
| Hierarchical (ward) | 20 | 0.0279 | 4.60 |
| GMM (tied) | 20 | 0.0406 | 4.84 |

### Wikipedia People

| Algorithm | Best k | Silhouette | Davies-Bouldin |
|---|---|---|---|
| K-Means | 20 | **0.1192** | 2.59 |
| Hierarchical (ward) | 20 | 0.0970 | 2.75 |
| GMM (tied) | 20 | 0.1068 | 2.66 |

Wikipedia clusters are significantly more coherent due to structured biographical content. Example K-Means k=20 cluster names:

```
Cluster 0  — rugby / hockey / season
Cluster 2  — music / orchestra / composer
Cluster 5  — album / band / music
Cluster 9  — university / research / professor
Cluster 10 — party / election / minister
Cluster 15 — book / published / novel
```

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| **Overview** | KPI cards, algorithm comparison bar charts, silhouette vs. k line chart, full metrics table |
| **Cluster Map** | Interactive PCA 2-D scatter colored by cluster, cluster-size bar chart |
| **Cluster Explorer** | Top-15 TF-IDF terms bar chart, sample documents per cluster |
| **Metrics** | Experiment comparison table with conditional formatting, elbow curve |
| **Predict** | Enter any text and get predicted cluster + top terms (GMM shows class probabilities) |

---

## Testing

```bash
pytest tests/ -v
```

52 tests covering preprocessing, TF-IDF/SVD features, and all three clustering algorithms.

```
tests/test_preprocessing.py   # clean_text, tokenize, preprocess_corpus
tests/test_features.py         # build_tfidf, reduce_with_svd, transform_new_docs
tests/test_clustering.py       # run_kmeans, run_hierarchical, run_gmm
```

---

## Docker

```bash
docker build -t doc-clustering .
docker run -p 8501:8501 doc-clustering
```

Open http://localhost:8501. The container auto-trains models on first startup.

---

## Deployment — Hugging Face Spaces

The space at `AhmedNashat1/document-clustering-explorer` uses the Docker SDK with `app_port: 8501`. On first boot, `app/startup.py` automatically trains a fast set of clustering models (reduced config, ~5-10 min) so the app is immediately usable without pre-built artifacts.

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis, word frequency, document length distributions |
| `02_preprocessing.ipynb` | Preprocessing pipeline walkthrough, token statistics |
| `03_clustering.ipynb` | Clustering experiments, elbow curve, dendrogram |
| `04_evaluation.ipynb` | Metric comparison across algorithms and datasets |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| ML / NLP | scikit-learn, NLTK |
| Feature extraction | TF-IDF + TruncatedSVD (LSA) |
| Clustering | K-Means, Agglomerative, GaussianMixture |
| Web app | Streamlit + Plotly |
| REST API | FastAPI + Uvicorn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Testing | pytest |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
