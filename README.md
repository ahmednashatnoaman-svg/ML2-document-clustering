---
title: Document Clustering Explorer
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.46.0
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# Unsupervised Document Clustering

Clustering **20 Newsgroups** and **Wikipedia People** documents with K-Means, Hierarchical Clustering, and GMM using TF-IDF + LSA features.

## Project Structure

```
ML2_final_project/
├── configs/config.yaml        # all hyperparams, paths, seeds
├── src/
│   ├── data_loader.py         # dataset loading
│   ├── preprocessing.py       # clean → tokenize → lemmatize
│   ├── features.py            # TF-IDF + TruncatedSVD (LSA)
│   ├── clustering.py          # KMeans, Hierarchical, GMM
│   ├── evaluation.py          # silhouette, DB, CH, top terms
│   └── visualization.py      # PCA, t-SNE, dendrogram, heatmap
├── app/
│   ├── streamlit_app.py       # interactive dashboard
│   └── api.py                 # FastAPI REST API
├── notebooks/                 # Jupyter notebooks (EDA → evaluation)
├── tests/                     # pytest test suite (52 tests)
├── run_pipeline.py            # end-to-end pipeline runner
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

## Run the Pipeline

```bash
# Full pipeline — both datasets
python run_pipeline.py

# One dataset only
python run_pipeline.py --dataset newsgroups

# Specific steps only (e.g. re-run evaluation + viz after clustering)
python run_pipeline.py --dataset newsgroups --steps evaluate visualize
```

The Wikipedia dataset is fetched automatically from the Wikipedia REST API (no key required). Alternatively, download from Kaggle and place at `data/raw/people_wiki.csv`:
```
https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data
```

## Run Tests

```bash
pytest tests/ -v
```

## Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Launch the API

```bash
uvicorn app.api:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

## Docker

```bash
docker build -t clustering-app .
docker run -p 8501:8501 clustering-app
```

## Key Results

| Dataset | Algorithm | k | Silhouette | Davies-Bouldin |
|---------|-----------|---|-----------|----------------|
| 20 Newsgroups | K-Means | 25 | 0.0548 | 3.62 |
| 20 Newsgroups | GMM (tied) | 20 | 0.0406 | 3.63 |
| 20 Newsgroups | Hierarchical | 20 | 0.0279 | 3.81 |
| Wikipedia People | K-Means | 25 | 0.1373 | 2.57 |
| Wikipedia People | Hierarchical | 20 | 0.1234 | 2.76 |
| Wikipedia People | GMM (full) | 20 | 0.1221 | 2.86 |

After running the pipeline, results are at:
- `outputs/reports/clustering_metrics_{dataset}.csv` — all metric scores
- `outputs/figures/` — all plots (PCA, t-SNE, dendrogram, heatmaps)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/datasets` | List loaded datasets |
| POST | `/predict` | Assign text to a cluster |
| GET | `/metrics/{dataset}` | Full metrics table |
| GET | `/metrics/{dataset}/best` | Best model row |
| GET | `/clusters/{dataset}/{algorithm}/{k}` | List cluster model keys |

## Configuration

Edit `configs/config.yaml` to change:
- `clustering.kmeans.k_range` — which k values to sweep
- `tfidf.max_features` — vocabulary size
- `svd.n_components` — LSA dimensions
- `seed` — global random seed (default: 42)
