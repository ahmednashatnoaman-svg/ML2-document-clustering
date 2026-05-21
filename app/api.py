"""FastAPI REST API for document clustering inference and metrics."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src/ is importable when running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import preprocess_text
from src.features import transform_new_docs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
REPORTS_DIR = os.getenv("REPORTS_DIR", "outputs/reports")

# ---------------------------------------------------------------------------
# Application state — loaded once at startup
# ---------------------------------------------------------------------------

class AppState:
    config: dict = {}
    artifacts: dict = {}   # {dataset: {vectorizer, svd_pipeline, models: {algo_k: model}}}
    metrics: dict[str, pd.DataFrame] = {}


state = AppState()


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_artifacts_for_dataset(dataset: str) -> dict:
    """Load vectorizer, SVD pipeline, and all available cluster models for a dataset."""
    vec_path = Path(MODELS_DIR) / f"tfidf_{dataset}.pkl"
    svd_path = Path(MODELS_DIR) / f"svd_{dataset}.pkl"

    if not vec_path.exists() or not svd_path.exists():
        logger.warning("Artifacts not found for dataset '%s'. Run the pipeline first.", dataset)
        return {}

    vectorizer = joblib.load(vec_path)
    svd_pipeline = joblib.load(svd_path)

    # Load every clustering model matching pattern: {algo}_{dataset}_k{k}*.pkl
    models = {}
    for pkl in Path(MODELS_DIR).glob(f"*_{dataset}_k*.pkl"):
        key = pkl.stem  # e.g. "kmeans_newsgroups_k10"
        try:
            models[key] = joblib.load(pkl)
        except Exception as e:
            logger.warning("Could not load %s: %s", pkl, e)

    return {"vectorizer": vectorizer, "svd_pipeline": svd_pipeline, "models": models}


def _load_metrics(dataset: str) -> Optional[pd.DataFrame]:
    path = Path(REPORTS_DIR) / f"clustering_metrics_{dataset}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts once at startup."""
    state.config = _load_config()
    for dataset in ("newsgroups", "wikipedia"):
        artifacts = _load_artifacts_for_dataset(dataset)
        if artifacts:
            state.artifacts[dataset] = artifacts
            logger.info("Loaded artifacts for: %s", dataset)
        df = _load_metrics(dataset)
        if df is not None:
            state.metrics[dataset] = df
            logger.info("Loaded metrics for: %s", dataset)
    logger.info("API startup complete. Datasets ready: %s", list(state.artifacts.keys()))
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Clustering API",
    description="Predict document clusters and explore clustering results.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw document text to cluster")
    dataset: str = Field("newsgroups", description="'newsgroups' or 'wikipedia'")
    algorithm: str = Field("kmeans", description="'kmeans', 'hierarchical', or 'gmm'")
    k: int = Field(10, ge=2, le=50, description="Number of clusters (must match a trained model)")
    linkage: Optional[str] = Field(None, description="Hierarchical linkage: ward/complete/average")
    covariance_type: Optional[str] = Field(None, description="GMM covariance: full/tied/diag")


class PredictResponse(BaseModel):
    cluster_id: int
    dataset: str
    algorithm: str
    k: int
    confidence: Optional[float] = None  # probability (GMM only)
    model_key: str


class ClusterInfoRequest(BaseModel):
    dataset: str
    algorithm: str
    k: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "datasets_loaded": list(state.artifacts.keys()),
        "metrics_loaded": list(state.metrics.keys()),
    }


@app.get("/config", tags=["System"])
def get_config():
    """Return the active project configuration."""
    if not state.config:
        raise HTTPException(500, "Config not loaded")
    return state.config


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Assign a document to a cluster using a pre-trained model.

    The model must have been trained and saved by run_pipeline.py.
    """
    if req.dataset not in state.artifacts:
        raise HTTPException(
            404,
            f"No artifacts for dataset '{req.dataset}'. Available: {list(state.artifacts.keys())}",
        )

    arts = state.artifacts[req.dataset]
    vectorizer = arts["vectorizer"]
    svd_pipeline = arts["svd_pipeline"]
    models = arts["models"]

    # Build the model key matching the saved filename pattern
    if req.algorithm == "kmeans":
        model_key = f"kmeans_{req.dataset}_k{req.k}"
    elif req.algorithm == "hierarchical":
        link = req.linkage or "ward"
        model_key = f"hierarchical_{req.dataset}_k{req.k}_{link}"
    elif req.algorithm == "gmm":
        cov = req.covariance_type or "full"
        model_key = f"gmm_{req.dataset}_k{req.k}_{cov}"
    else:
        raise HTTPException(400, f"Unknown algorithm: {req.algorithm}")

    if model_key not in models:
        raise HTTPException(
            404,
            f"Model '{model_key}' not found. Available: {sorted(models.keys())}",
        )

    # Preprocess + transform
    cleaned = preprocess_text(req.text)
    if not cleaned.strip():
        raise HTTPException(422, "Text became empty after preprocessing.")

    X_new = transform_new_docs([cleaned], vectorizer, svd_pipeline)
    model = models[model_key]

    if not hasattr(model, "predict"):
        raise HTTPException(
            400,
            f"Algorithm '{req.algorithm}' (AgglomerativeClustering) does not support "
            "prediction on new documents. Use 'kmeans' or 'gmm' for inference.",
        )
    cluster_id = int(model.predict(X_new)[0])
    confidence = None
    if req.algorithm == "gmm":
        probs = model.predict_proba(X_new)[0]
        confidence = float(probs[cluster_id])

    return PredictResponse(
        cluster_id=cluster_id,
        dataset=req.dataset,
        algorithm=req.algorithm,
        k=req.k,
        confidence=confidence,
        model_key=model_key,
    )


def _df_to_json_safe(df) -> list[dict]:
    """Convert DataFrame to JSON-safe records, replacing NaN/Inf with None."""
    return [
        {k: (None if (isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))) else v)
         for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


@app.get("/metrics/{dataset}", tags=["Analysis"])
def get_metrics(dataset: str):
    """Return the full metrics table for a dataset as JSON records."""
    if dataset not in state.metrics:
        raise HTTPException(
            404,
            f"No metrics for '{dataset}'. Run the pipeline first.",
        )
    return _df_to_json_safe(state.metrics[dataset])


@app.get("/metrics/{dataset}/best", tags=["Analysis"])
def get_best_model(
    dataset: str,
    algorithm: Optional[str] = None,
    metric: str = "silhouette",
):
    """Return the experiment row with the highest silhouette (or chosen metric)."""
    if dataset not in state.metrics:
        raise HTTPException(404, f"No metrics for '{dataset}'.")
    df = state.metrics[dataset]
    if algorithm:
        df = df[df["algorithm"] == algorithm]
    if df.empty:
        raise HTTPException(404, "No matching experiments found.")
    best_row = df.loc[df[metric].idxmax()]
    return _df_to_json_safe(best_row.to_frame().T)[0]


@app.get("/clusters/{dataset}/{algorithm}/{k}", tags=["Analysis"])
def get_cluster_info(dataset: str, algorithm: str, k: int):
    """List available cluster model keys for given (dataset, algorithm, k)."""
    if dataset not in state.artifacts:
        raise HTTPException(404, f"No artifacts for '{dataset}'.")

    models = state.artifacts[dataset]["models"]
    prefix = f"{algorithm}_{dataset}_k{k}"
    matching = {key: "loaded" for key in models if key.startswith(prefix)}

    if not matching:
        raise HTTPException(404, f"No models matching '{prefix}'. Available: {sorted(models.keys())}")

    return {"dataset": dataset, "algorithm": algorithm, "k": k, "models": matching}


@app.get("/datasets", tags=["System"])
def list_datasets():
    """List datasets with loaded artifacts."""
    return {
        "available": list(state.artifacts.keys()),
        "metrics_available": list(state.metrics.keys()),
    }
