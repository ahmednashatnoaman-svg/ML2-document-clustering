"""Text preprocessing pipeline: clean → tokenize → lemmatize."""

import logging
import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# Download required NLTK resources on first import
_NLTK_DOWNLOADS = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]

def _ensure_nltk_resources() -> None:
    for resource in _NLTK_DOWNLOADS:
        try:
            if resource in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{resource}")
            elif resource in ("stopwords", "wordnet", "omw-1.4"):
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Compiled regexes for speed
_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_EMAIL = re.compile(r"\S+@\S+")
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")
_RE_DIGITS = re.compile(r"\b\d+\b")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[%s]" % re.escape(string.punctuation))


def clean_text(text: str) -> str:
    """Remove noise from raw text: URLs, HTML, email, punctuation, digits.

    Args:
        text: raw document string

    Returns:
        Lowercased, cleaned string
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_NON_ASCII.sub(" ", text)
    text = _RE_DIGITS.sub(" ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, min_token_len: int = 3) -> str:
    """Tokenize, remove stopwords, and lemmatize.

    Args:
        text: pre-cleaned lowercase string
        min_token_len: discard tokens shorter than this

    Returns:
        Space-joined lemmatized tokens
    """
    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) >= min_token_len and tok.isalpha()
    ]
    return " ".join(tokens)


def preprocess_text(text: str, min_token_len: int = 3) -> str:
    """Full single-document pipeline: clean → tokenize → lemmatize."""
    return tokenize_and_lemmatize(clean_text(text), min_token_len)


def preprocess_corpus(
    texts: List[str],
    min_token_len: int = 3,
    min_doc_len: int = 10,
) -> tuple[List[str], List[int]]:
    """Preprocess an entire corpus and filter empty/short documents.

    Args:
        texts: list of raw document strings
        min_token_len: minimum token character length
        min_doc_len: minimum number of characters in processed doc

    Returns:
        (processed_texts, kept_indices) — kept_indices maps output → input positions
    """
    logger.info("Preprocessing %d documents...", len(texts))
    processed, kept = [], []

    for i, text in enumerate(texts):
        result = preprocess_text(text, min_token_len)
        if len(result) >= min_doc_len:
            processed.append(result)
            kept.append(i)

    dropped = len(texts) - len(processed)
    logger.info(
        "Preprocessing complete: %d kept, %d dropped (too short after cleaning)",
        len(processed),
        dropped,
    )
    return processed, kept
