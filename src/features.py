"""Feature engineering for FakeNewsNet.

Text features   -> TF-IDF over the article title (and any supplied body).
Social features -> tweet sharing volume (log-scaled).
Stylistic       -> title length / punctuation / all-caps statistics.

All non-text features are stored in a small numeric frame so they can be
plugged into any scikit-learn preprocessor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TEXT_COLUMN = "text"
NUMERIC_COLUMNS = [
    "log_tweet_count",
    "title_word_count",
    "title_char_count",
    "exclamation_count",
    "question_count",
    "all_caps_ratio",
    "has_number",
]

# Default English stopwords used by TfidfVectorizer; kept here for clarity.
STOPWORDS = None

MALAY_STOPWORDS = frozenset(
    """
    yang dan untuk ini itu di ke dari dengan adalah atau pada bagi akan tidak
    juga telah boleh tentang jika kerana supaya antara sebagai mereka kita kamu
    anda saya dia kami ialah iaitu tetapi serta namun dalam kepada mengenai
    seperti sementara manakala apabila selepas sebelum masih sudah belum hanya
    sahaja lagi pun lah kah nya ada apa bila mana siapa kenapa bagaimana oleh
    """.split()
)


def combined_stopwords() -> list[str]:
    """English + Malay stopwords for the TF-IDF vectoriser (bilingual content)."""
    return sorted(ENGLISH_STOP_WORDS | MALAY_STOPWORDS)


def clean_text(text: str) -> str:
    """Light normalisation; heavy lifting is left to TfidfVectorizer."""
    if not text:
        return ""
    return " ".join(str(text).replace("\u00a0", " ").split())


def join_text(title: str, body: str | None = None) -> str:
    title = clean_text(title)
    if body:
        body = clean_text(body)
        return f"{title} {body}" if title else body
    return title


def title_stats(series: pd.Series) -> pd.DataFrame:
    """Numeric stylistic features computed from article titles."""
    lower = series.astype(str).str.lower()
    words = series.astype(str).str.split()
    word_counts = words.apply(len)
    char_counts = series.astype(str).str.len()

    exclaim = series.astype(str).str.count(r"!")
    question = series.astype(str).str.count(r"\?")
    all_caps_chars = series.astype(str).str.count(r"[A-Z]")
    alpha_chars = series.astype(str).str.count(r"[A-Za-z]")
    all_caps_ratio = np.divide(
        all_caps_chars.values,
        alpha_chars.values,
        out=np.zeros(len(series), dtype=float),
        where=alpha_chars.values > 0,
    )
    has_number = lower.str.contains(r"\d", regex=True).astype(int)

    out = pd.DataFrame(
        {
            "title_word_count": word_counts,
            "title_char_count": char_counts,
            "exclamation_count": exclaim,
            "question_count": question,
            "all_caps_ratio": all_caps_ratio,
            "has_number": has_number,
        },
        index=series.index,
    )
    return out


def social_features(df: pd.DataFrame) -> pd.DataFrame:
    """Social engagement features derived from the tweet_ids column."""
    tweet_count = df["tweet_count"].astype(int)
    return pd.DataFrame(
        {
            "log_tweet_count": np.log1p(tweet_count),
        },
        index=df.index,
    )


def build_text_series(df: pd.DataFrame, body_column: str | None = None) -> pd.Series:
    """Return the cleaned text column (title + optional body) for TF-IDF."""
    if body_column and body_column in df.columns:
        return df.apply(
            lambda row: join_text(row["title"], row.get(body_column)), axis=1
        )
    return df["title"].map(clean_text)


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine social + stylistic features into a single numeric matrix."""
    return pd.concat(
        [social_features(df), title_stats(df["title"])],
        axis=1,
    )


def build_features(
    df: pd.DataFrame,
    use_social: bool = True,
    use_stylistic: bool = True,
    body_column: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full numeric feature frame.

    Returns (numeric_feature_df, list_of_columns_used). The text column is
    returned separately via `build_text_series`.
    """
    parts: list[pd.DataFrame] = []
    columns: list[str] = []
    if use_social:
        parts.append(social_features(df))
    if use_stylistic:
        parts.append(title_stats(df["title"]))

    if parts:
        numeric = pd.concat(parts, axis=1)
        columns = list(numeric.columns)
    else:
        numeric = pd.DataFrame(index=df.index)
        columns = []
    return numeric, columns


def available_numeric_columns() -> list[str]:
    return NUMERIC_COLUMNS


def available_social_columns() -> list[str]:
    return [c for c in NUMERIC_COLUMNS if c.startswith("log_tweet_count")]