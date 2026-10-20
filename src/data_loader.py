"""Loaders for the FakeNewsNet minimal dataset.

Each CSV has columns: id, news_url, title, tweet_ids
where tweet_ids is a tab-separated list of tweet ids sharing the article.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    DATASET_FILES,
    DATASET_SOURCES,
    MALAYSIA_FILE,
    MALAYSIA_AUGMENTED_FILE,
    MALAYSIA_SOURCE,
)

log = logging.getLogger(__name__)

FILE_LABEL = {
    "politifact_fake": 1,
    "politifact_real": 0,
    "gossipcop_fake": 1,
    "gossipcop_real": 0,
}

FILE_SOURCE = {
    "politifact_fake": "politifact",
    "politifact_real": "politifact",
    "gossipcop_fake": "gossipcop",
    "gossipcop_real": "gossipcop",
}


def _read_csv(path: pd.DataFrame | str) -> pd.DataFrame:
    # FakeNewsNet titles occasionally contain non-utf8 bytes; fall back gracefully.
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.warning("Failed reading %s with default encodings: %s", path, exc)
        raise


def load_raw(sources: tuple[str, ...] = DATASET_SOURCES) -> pd.DataFrame:
    """Load the raw FakeNewsNet CSVs into a single labelled frame.

    Labels: 1 = fake, 0 = real. Source column keeps the ground-truth origin.
    """
    frames = []
    for key, path in DATASET_FILES.items():
        source = FILE_SOURCE[key]
        if source not in sources or not path.exists():
            continue
        df = _read_csv(path)
        df["label"] = FILE_LABEL[key]
        df["source"] = source
        frames.append(df)
    if not frames:
        from .config import DATA_DIR

        raise FileNotFoundError(
            f"No dataset files found under {DATA_DIR}. "
            "Expected the four FakeNewsNet CSVs in the data/ folder."
        )
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column types and fill missing fields for modelling."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    expected = {"id", "news_url", "title", "tweet_ids"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["news_url"] = df["news_url"].fillna("").astype(str).str.strip()
    df["tweet_ids"] = df["tweet_ids"].fillna("").astype(str)

    def split_tweets(val: str) -> list[str]:
        return [t for t in str(val).split("\t") if t.strip()]

    df["tweet_ids_list"] = df["tweet_ids"].apply(split_tweets)
    df["tweet_count"] = df["tweet_ids_list"].apply(len)
    df["label"] = df["label"].astype(int)
    return df


def load_prepared(sources: tuple[str, ...] = DATASET_SOURCES) -> pd.DataFrame:
    """One-stop loader returning a clean, ready-to-model frame.

    The Malaysia source is a single CSV (``title, news_url, tweet_count,
    label``) and is handled separately from the FakeNewsNet CSVs.
    """
    sources = tuple(sources)
    frames: list[pd.DataFrame] = []
    if MALAYSIA_SOURCE in sources:
        frames.append(load_malaysia())
    others = tuple(s for s in sources if s != MALAYSIA_SOURCE)
    if others:
        frames.append(prepare(load_raw(others)))
    if not frames:
        raise FileNotFoundError("No dataset sources were selected.")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def load_malaysia() -> pd.DataFrame:
    """Load the Malaysia political-misinformation dataset.

    Expected columns: ``title``, ``label`` (1 = misinformation, 0 = reliable),
    optional ``news_url`` and ``tweet_count``. Missing/zero share counts are
    filled with a deterministic simulated value (misinformation tends to spread
    more) — a documented limitation when real engagement data is unavailable.

    If an augmented file exists (``malaysia_news_augmented.csv``) it is used in
    preference to the raw seed file.
    """
    source_file = MALAYSIA_AUGMENTED_FILE if MALAYSIA_AUGMENTED_FILE.exists() else MALAYSIA_FILE
    if not source_file.exists():
        raise FileNotFoundError(
            f"Malaysia dataset not found: {source_file}. "
            "Create data/malaysia_news.csv with columns "
            "title,news_url,tweet_count,label."
        )

    df = _read_csv(source_file)
    df.columns = [c.strip() for c in df.columns]
    missing = {"title", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Malaysia dataset missing columns: {sorted(missing)}")

    if "group" not in df.columns:
        df["group"] = range(len(df))
    df["group"] = pd.to_numeric(df["group"], errors="coerce").fillna(0).astype(int)

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    if "news_url" not in df.columns:
        df["news_url"] = ""
    df["news_url"] = df["news_url"].fillna("").astype(str).str.strip()

    if "tweet_count" not in df.columns:
        df["tweet_count"] = 0
    df["tweet_count"] = (
        pd.to_numeric(df["tweet_count"], errors="coerce").fillna(0).astype(int)
    )

    df = _simulate_missing_shares(df)
    df["source"] = MALAYSIA_SOURCE
    df["id"] = [f"my-{i + 1:06d}" for i in range(len(df))]
    df["tweet_ids"] = ""
    df["tweet_ids_list"] = [[] for _ in range(len(df))]
    return df


def _simulate_missing_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Fill zero share counts with a deterministic simulated value.

    Misinformation (label 1) is given a larger share range than reliable news
    (label 0), reflecting the well-documented tendency for false content to
    spread further. Only rows that do not already carry a real share count are
    touched.
    """
    mask = df["tweet_count"] <= 0
    if not mask.any():
        return df
    rng = np.random.default_rng(42)
    fake = (df["label"] == 1) & mask
    real = (df["label"] == 0) & mask
    if fake.any():
        df.loc[fake, "tweet_count"] = rng.integers(300, 5000, size=int(fake.sum()))
    if real.any():
        df.loc[real, "tweet_count"] = rng.integers(5, 400, size=int(real.sum()))
    return df