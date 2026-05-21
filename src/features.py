"""Feature extraction: TF-IDF vectorization and dimensionality reduction (LSA/SVD)."""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)


def build_tfidf(
    corpus: list[str],
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> tuple:
    """Fit a TF-IDF vectorizer on corpus and transform it.

    Args:
        corpus: list of preprocessed document strings
        max_features: vocabulary size cap
        ngram_range: (min_n, max_n) n-gram window
        min_df: ignore terms appearing in fewer than this many docs
        max_df: ignore terms appearing in more than this fraction of docs
        sublinear_tf: apply log(1 + tf) scaling

    Returns:
        (X_tfidf, vectorizer) — sparse matrix + fitted vectorizer
    """
    logger.info(
        "Building TF-IDF: max_features=%d, ngram_range=%s, min_df=%d, max_df=%.2f",
        max_features, ngram_range, min_df, max_df,
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    X = vectorizer.fit_transform(corpus)
    logger.info(
        "TF-IDF matrix: %s  |  vocabulary size: %d",
        X.shape, len(vectorizer.vocabulary_),
    )
    return X, vectorizer


def reduce_with_svd(
    X,
    n_components: int = 100,
    random_state: int = 42,
) -> tuple:
    """Apply Truncated SVD (LSA) to a sparse TF-IDF matrix.

    Follows the LSA convention: SVD then L2-normalize rows.

    Args:
        X: sparse matrix from TF-IDF
        n_components: latent dimensions to keep
        random_state: reproducibility seed

    Returns:
        (X_reduced, svd_pipeline) — dense ndarray + fitted Pipeline
    """
    # SVD requires n_components < min(n_samples, n_features)
    max_components = min(X.shape[0], X.shape[1]) - 1
    if n_components >= max_components:
        logger.warning(
            "n_components=%d capped to %d (matrix shape %s)",
            n_components, max_components, X.shape,
        )
        n_components = max_components

    logger.info("Applying TruncatedSVD: n_components=%d", n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    normalizer = Normalizer(copy=False)
    pipe = Pipeline([("svd", svd), ("normalizer", normalizer)])
    X_reduced = pipe.fit_transform(X)

    explained = svd.explained_variance_ratio_.sum()
    logger.info(
        "SVD done: output shape=%s  |  explained variance=%.3f",
        X_reduced.shape, explained,
    )
    return X_reduced, pipe


def transform_new_docs(
    texts: list[str],
    vectorizer: TfidfVectorizer,
    svd_pipeline: Pipeline,
) -> np.ndarray:
    """Vectorize + reduce new (unseen) documents using fitted artifacts.

    Args:
        texts: preprocessed text strings
        vectorizer: fitted TfidfVectorizer
        svd_pipeline: fitted SVD + Normalizer pipeline

    Returns:
        Dense ndarray of shape (n_docs, n_components)
    """
    X = vectorizer.transform(texts)
    return svd_pipeline.transform(X)


def save_feature_artifacts(
    vectorizer: TfidfVectorizer,
    svd_pipeline: Pipeline,
    X_tfidf,
    X_reduced: np.ndarray,
    dataset_name: str,
    models_dir: str = "models",
    processed_dir: str = "data/processed",
) -> None:
    """Persist vectorizer, SVD pipeline, and matrices to disk.

    X_reduced is saved to both models/ (committed to git) and data/processed/
    so deployments without a training step can still load it.
    """
    from scipy.sparse import save_npz

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, f"{models_dir}/tfidf_{dataset_name}.pkl")
    joblib.dump(svd_pipeline, f"{models_dir}/svd_{dataset_name}.pkl")

    # Save X_reduced in models/ so it is included in git alongside the models
    np.save(f"{models_dir}/X_reduced_{dataset_name}.npy", X_reduced)
    np.save(f"{processed_dir}/X_reduced_{dataset_name}.npy", X_reduced)

    save_npz(f"{processed_dir}/X_tfidf_{dataset_name}.npz", X_tfidf)

    logger.info("Feature artifacts saved for dataset: %s", dataset_name)


def load_feature_artifacts(
    dataset_name: str,
    models_dir: str = "models",
    processed_dir: str = "data/processed",
) -> tuple:
    """Load persisted feature artifacts for a dataset.

    Returns:
        (vectorizer, svd_pipeline, X_tfidf, X_reduced)
    """
    from scipy.sparse import load_npz

    vectorizer = joblib.load(f"{models_dir}/tfidf_{dataset_name}.pkl")
    svd_pipeline = joblib.load(f"{models_dir}/svd_{dataset_name}.pkl")
    X_reduced = np.load(f"{processed_dir}/X_reduced_{dataset_name}.npy")
    X_tfidf = load_npz(f"{processed_dir}/X_tfidf_{dataset_name}.npz")

    logger.info("Feature artifacts loaded for dataset: %s", dataset_name)
    return vectorizer, svd_pipeline, X_tfidf, X_reduced
