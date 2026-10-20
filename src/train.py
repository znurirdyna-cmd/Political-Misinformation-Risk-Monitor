"""Training pipeline for the FakeNewsNet fake-news classifier.

Builds a scikit-learn Pipeline: a ColumnTransformer that combines TF-IDF
text features with scaled numeric (social + stylistic) features, followed by
a configurable classifier. Trained artifacts are persisted with joblib.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from . import features as ft

log = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_NAME = Literal[
    "Logistic Regression", "Linear SVM", "Random Forest", "Multinomial Naive Bayes"
]

MODEL_FACTORIES: dict[str, type[ClassifierMixin]] = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Linear SVM": lambda: LinearSVC(
        max_iter=10000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=250,
        max_depth=32,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "Multinomial Naive Bayes": lambda: MultinomialNB(alpha=0.5),
}

USE_MINMAX = {"Multinomial Naive Bayes"}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "Logistic Regression"
    use_text: bool = True
    use_social: bool = True
    use_stylistic: bool = True
    sources: tuple[str, ...] = ("politifact", "gossipcop")
    test_size: float = 0.25
    max_features: int = 20000
    body_column: str | None = None
    sample_frac: float = 1.0


def score_signal(clf: ClassifierMixin, X) -> np.ndarray:
    """Return a probability-like score for a fitted classifier."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        # Rough sigmoid scaling of the decision function for ROC / probability bar.
        dec = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-dec))
    raise AttributeError(f"{clf!r} exposes no probability or decision function.")


def make_pipeline(
    config: TrainConfig,
    numeric_cols: list[str],
) -> Pipeline:
    """Assemble the preprocessing + classifier pipeline."""
    transformers = []
    if config.use_text:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=config.max_features,
                    sublinear_tf=True,
                    stop_words=ft.combined_stopwords(),
                ),
                ft.TEXT_COLUMN,
            )
        )
    if numeric_cols:
        scaler = MinMaxScaler() if config.model_name in USE_MINMAX else StandardScaler()
        num_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", scaler)]
        )
        transformers.append(("num", num_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if not transformers:
        raise ValueError(
            "At least one feature group (text, social, or stylistic) must be enabled."
        )
    clf = MODEL_FACTORIES[config.model_name]()
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def evaluate(clf: ClassifierMixin, X, y: pd.Series) -> dict:
    """Compute evaluation metrics for a fitted classifier."""
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, preds, labels=[1], average="binary", zero_division=0
    )
    pos_ratio = float(y.mean())
    try:
        auc = float(roc_auc_score(y, score_signal(clf, X)))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "positive_ratio": round(pos_ratio, 4),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds, zero_division=0),
    }


def _prepare_xy(
    df: pd.DataFrame, config: TrainConfig
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, list[str], pd.Series | None]:
    df = df.copy()
    if config.sample_frac < 1.0:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=config.sample_frac, random_state=RANDOM_STATE)
        )
    text = ft.build_text_series(df, config.body_column)
    numeric, numeric_cols = ft.build_features(
        df, use_social=config.use_social, use_stylistic=config.use_stylistic
    )
    y = df["label"].reset_index(drop=True)
    groups = df["group"].reset_index(drop=True) if "group" in df.columns else None
    return (
        text.reset_index(drop=True),
        y,
        numeric.reset_index(drop=True),
        numeric_cols,
        groups,
    )


def _grouped_split(
    groups: pd.Series, y: pd.Series, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices keeping every row of a group in the same split.

    Groups are split independently per class (group-level stratification) so a
    class stays represented in both train and test. Prevents augmented
    near-duplicate rows from leaking across train/test.
    """
    groups_arr = groups.to_numpy()
    y_arr = y.to_numpy()
    rng = np.random.default_rng(random_state)
    test_mask = np.zeros(len(groups_arr), dtype=bool)
    for label in np.unique(y_arr):
        g_vals = np.unique(groups_arr[y_arr == label])
        rng.shuffle(g_vals)
        n_test = max(1, int(round(len(g_vals) * test_size)))
        test_groups = set(g_vals[:n_test].tolist())
        test_mask |= np.isin(groups_arr, list(test_groups))
    train_mask = ~test_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


def train(df: pd.DataFrame, config: TrainConfig) -> dict:
    """Run the full workflow: split, fit, evaluate, persist artifacts."""
    start = time.perf_counter()

    text, y, numeric, numeric_cols, groups = _prepare_xy(df, config)
    X = (
        pd.merge(text.to_frame(ft.TEXT_COLUMN), numeric, left_index=True, right_index=True)
        if len(numeric) and len(numeric_cols)
        else text.to_frame(ft.TEXT_COLUMN)
    )

    if groups is not None and groups.nunique() > 1:
        train_idx, test_idx = _grouped_split(groups, y, config.test_size, RANDOM_STATE)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, stratify=y, random_state=RANDOM_STATE
        )

    pipeline = make_pipeline(config, numeric_cols)
    pipeline.fit(X_train, y_train)

    test_metrics = evaluate(pipeline, X_test, y_test)
    train_metrics = evaluate(pipeline, X_train, y_train)

    # Propagation-risk normalisation bounds, fitted on the training split only
    # (log-scale min-max so a few viral outliers do not dominate the scale).
    if "log_tweet_count" in X_train.columns:
        log_tweet_counts = X_train["log_tweet_count"].dropna()
        log_tweet_min = float(log_tweet_counts.min())
        log_tweet_max = float(log_tweet_counts.max())
    else:
        log_tweet_min = log_tweet_max = None

    metadata = {
        "model_name": config.model_name,
        "label_map": {"0": "LIKELY RELIABLE", "1": "LIKELY MISINFORMATION"},
        "features": {
            "text": config.use_text,
            "social": config.use_social,
            "stylistic": config.use_stylistic,
            "numeric_columns": numeric_cols,
        },
        "sources": list(config.sources),
        "max_features": config.max_features,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fit_seconds": round(float(time.perf_counter() - start), 2),
        "risk_scoring": {
            "alpha": 0.6,
            "beta": 0.4,
            "log_tweet_min": log_tweet_min,
            "log_tweet_max": log_tweet_max,
            "risk_levels": [
                {"name": "Low", "min": 0, "max": 39},
                {"name": "Medium", "min": 40, "max": 69},
                {"name": "High", "min": 70, "max": 100},
            ],
        },
    }
    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "metrics": {"test": test_metrics, "train": train_metrics},
        "columns": list(X.columns),
        "y_test": y_test,
    }


def save_artifacts(result: dict, model_dir=None) -> None:
    from .config import MODEL_ARTIFACTS, MODEL_DIR

    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], MODEL_ARTIFACTS["pipeline"])
    with open(MODEL_ARTIFACTS["metadata"], "w") as fh:
        json.dump(result["metadata"], fh, indent=2)
    with open(MODEL_ARTIFACTS["metrics"], "w") as fh:
        json.dump(result["metrics"], fh, indent=2)


def load_trained(model_dir=None):
    """Load persisted pipeline + metadata. Returns (pipeline, metadata, metrics)."""
    from .config import MODEL_ARTIFACTS, MODEL_DIR

    model_dir = model_dir or MODEL_DIR
    pipeline = joblib.load(MODEL_ARTIFACTS["pipeline"])
    with open(MODEL_ARTIFACTS["metadata"]) as fh:
        metadata = json.load(fh)
    metrics = {}
    if MODEL_ARTIFACTS["metrics"].exists():
        with open(MODEL_ARTIFACTS["metrics"]) as fh:
            metrics = json.load(fh)
    return pipeline, metadata, metrics


def has_trained_model(model_dir=None):
    from .config import MODEL_ARTIFACTS, MODEL_DIR

    model_dir = model_dir or MODEL_DIR
    return MODEL_ARTIFACTS["pipeline"].exists() and MODEL_ARTIFACTS["metadata"].exists()


def train_default(df: pd.DataFrame) -> dict:
    """Quick default model used as a fallback when no artifacts exist."""
    config = TrainConfig()
    result = train(df, config)
    save_artifacts(result)
    return result