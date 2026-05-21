"""Data loading for 20 Newsgroups and People Wikipedia datasets."""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_newsgroups(
    subset: str = "all",
    categories: Optional[list] = None,
    remove: tuple = ("headers", "footers", "quotes"),
    random_state: int = 42,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load the 20 Newsgroups dataset.

    Returns:
        texts: raw document strings
        labels: integer category labels
        target_names: human-readable category names
    """
    logger.info("Loading 20 Newsgroups (subset=%s)...", subset)
    data = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=remove,
        random_state=random_state,
    )
    texts = data.data
    labels = data.target
    target_names = data.target_names
    logger.info(
        "Loaded %d documents across %d categories", len(texts), len(target_names)
    )
    return texts, labels, target_names


def load_wikipedia_people(
    filepath: str,
    text_col: str = "text",
    name_col: str = "name",
    min_text_length: int = 100,
    auto_download: bool = True,
) -> tuple[list[str], list[str]]:
    """Load the People Wikipedia dataset from a CSV file.

    If the file is not found and auto_download=True, fetches articles
    automatically from the Wikipedia REST API (no key required).

    The CSV must have at minimum a text column and a name column.
    Rows with text shorter than min_text_length are dropped.

    Returns:
        texts: list of article texts
        names: list of person names
    """
    logger.info("Loading Wikipedia People dataset from %s...", filepath)
    if not Path(filepath).exists():
        if auto_download:
            logger.info("CSV not found at '%s' — auto-downloading from Wikipedia API...", filepath)
            from src.wikipedia_fetcher import download_people_wiki
            download_people_wiki(output_path=filepath)
        else:
            raise FileNotFoundError(
                f"Wikipedia dataset not found at '{filepath}'.\n"
                "Either set auto_download=True or download from:\n"
                "  https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data"
            )

    df = pd.read_csv(filepath)
    logger.info("Raw shape: %s", df.shape)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    # Drop rows with missing or too-short text
    before = len(df)
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.len() >= min_text_length].reset_index(drop=True)
    logger.info("Kept %d / %d docs (min_length=%d)", len(df), before, min_text_length)

    texts = df[text_col].tolist()
    names = df[name_col].tolist() if name_col in df.columns else [""] * len(df)
    return texts, names


def save_raw(obj, path: str) -> None:
    """Pickle-save a raw dataset list to disk."""
    import pickle

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved raw data → %s", path)


def load_raw(path: str):
    """Load a pickled raw dataset."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
