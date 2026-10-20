"""Prediction helpers for single news articles using the saved pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as ft
from . import train as training


class Predictor:
    """Wraps a persisted pipeline + metadata for convenient prediction."""

    def __init__(self, pipeline=None, metadata=None, metrics=None):
        self.pipeline = pipeline
        self.metadata = metadata or {}
        self.metrics = metrics or {}

    @classmethod
    def load(cls, fallback_df: pd.DataFrame | None = None) -> "Predictor":
        if training.has_trained_model():
            pipeline, metadata, metrics = training.load_trained()
            return cls(pipeline, metadata, metrics)
        if fallback_df is not None and len(fallback_df):
            result = training.train_default(fallback_df)
            return cls(result["pipeline"], result["metadata"], result["metrics"])
        raise FileNotFoundError(
            "No trained model found and no dataset available to train a fallback."
        )

    def predict(self, title: str, body: str | None = None, tweet_count: int = 0):
        """Predict a single article. Returns (label, probability, details)."""
        if self.pipeline is None:
            raise RuntimeError("Predictor has no fitted pipeline.")

        cols = self.metadata.get("features", {}).get("numeric_columns", [])
        row = pd.DataFrame({ft.TEXT_COLUMN: [ft.join_text(title, body)], "title": [title]})
        if "log_tweet_count" in self.metadata.get("features", {}).get("numeric_columns", []):
            row["tweet_count"] = [int(tweet_count)]

        numeric, _ = ft.build_features(
            row,
            use_social=self.metadata.get("features", {}).get("social", True),
            use_stylistic=self.metadata.get("features", {}).get("stylistic", True),
        )
        # keep only columns the model was trained on, in order
        if cols:
            numeric = numeric.reindex(columns=cols, fill_value=0.0)
        X = row[[ft.TEXT_COLUMN]].copy()
        if len(cols):
            X = pd.concat([X, numeric], axis=1)

        label = int(self.pipeline.predict(X)[0])
        p_fake = float(np.clip(training.score_signal(self.pipeline, X)[0], 0.0, 1.0))

        label_name = self.metadata.get("label_map", {})[str(label)]
        risk = self.compute_risk_score(p_fake, int(tweet_count))

        return label, label_name, float(p_fake), {
            "raw_score": float(p_fake),
            "features_used": self.metadata.get("features", {}),
            "tweet_count": int(tweet_count),
            "model": self.metadata.get("model_name", "unknown"),
            **risk,
        }

    def score_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a whole dataset of articles for dashboarding.

        Reuses the same feature pipeline and risk formula as single-article
        prediction. Expects a prepared frame with ``title`` and ``tweet_count``
        (see ``data_loader.load_prepared``). Returns a frame augmented with
        per-article risk values (nothing is invented — all values come from the
        fitted model and the actual tweet counts).
        """
        df = df.copy().reset_index(drop=True)
        meta_features = self.metadata.get("features", {})
        cols = list(meta_features.get("numeric_columns", []))
        use_social = bool(meta_features.get("social", True))
        use_stylistic = bool(meta_features.get("stylistic", True))

        text = ft.build_text_series(df)
        numeric, _ = ft.build_features(
            df, use_social=use_social, use_stylistic=use_stylistic
        )
        if cols:
            numeric = numeric.reindex(columns=cols, fill_value=0.0)
        X = text.to_frame(ft.TEXT_COLUMN).reset_index(drop=True)
        if cols:
            X = pd.concat([X, numeric.reset_index(drop=True)], axis=1)

        y_pred = self.pipeline.predict(X).astype(int)
        p_fake = np.clip(training.score_signal(self.pipeline, X), 0.0, 1.0)

        keep = [c for c in ("id", "title", "news_url", "source", "tweet_count") if c in df.columns]
        out = df[keep].copy()
        out["pred_label"] = y_pred
        out["p_fake"] = p_fake

        risk_rows = [
            self.compute_risk_score(float(p), int(t))
            for p, t in zip(p_fake, out["tweet_count"].astype(int))
        ]
        out["content_risk"] = [r["content_risk"] for r in risk_rows]
        out["propagation_risk"] = [r["propagation_risk"] for r in risk_rows]
        out["risk_score"] = [r["risk_score"] for r in risk_rows]
        out["risk_level"] = [r["risk_level"] for r in risk_rows]
        return out

    def compute_risk_score(
        self, p_fake: float, tweet_count: int
    ) -> dict[str, float | str]:
        """Composite cyber risk score (report §3.6).

        Rc (content risk)   = P(fake) * 100
        Rp (propagation)    = log-scaled min-max of tweet count, fitted on training
        Risk Score          = alpha * Rc + beta * Rp   (alpha=0.6, beta=0.4)
        Level               = Low (0-39) / Medium (40-69) / High (70-100)
        """
        rs = self.metadata.get("risk_scoring", {})

        alpha = float(rs.get("alpha", 0.6))
        beta = float(rs.get("beta", 0.4))
        content_risk = float(np.clip(p_fake, 0.0, 1.0)) * 100.0

        log_min = rs.get("log_tweet_min")
        log_max = rs.get("log_tweet_max")
        if log_min is not None and log_max is not None and log_max > log_min:
            log_value = float(np.log1p(max(int(tweet_count), 0)))
            propagation_risk = (log_value - log_min) / (log_max - log_min) * 100.0
        else:
            propagation_risk = 0.0
        propagation_risk = float(np.clip(propagation_risk, 0.0, 100.0))

        risk_score = alpha * content_risk + beta * propagation_risk
        risk_score = float(np.clip(risk_score, 0.0, 100.0))

        level = "Medium"
        for candidate in rs.get("risk_levels", []):
            if candidate.get("min", 0) <= risk_score <= candidate.get("max", 100):
                level = candidate["name"]
                break

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "content_risk": content_risk,
            "propagation_risk": propagation_risk,
            "alpha": alpha,
            "beta": beta,
            "risk_levels": rs.get("risk_levels", []),
        }

    def model_summary(self) -> str:
        feat = self.metadata.get("features", {})
        used = []
        if feat.get("text"):
            used.append("title text (TF-IDF)")
        if feat.get("social"):
            used.append("social (tweet volume)")
        if feat.get("stylistic"):
            used.append("stylistic (title stats)")
        return f"{self.metadata.get('model_name', '?')} — features: {', '.join(used) or 'none'}"