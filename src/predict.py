
"""Prediction helpers for single news articles using the saved pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as ft
from . import train as training


class Predictor:
    """Wraps a persisted pipeline and metadata for prediction."""

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
            return cls(
                result["pipeline"],
                result["metadata"],
                result["metrics"],
            )

        raise FileNotFoundError(
            "No trained model found and no dataset available."
        )

    def predict(
        self,
        title: str,
        body: str | None = None,
        tweet_count: int | None = None,
    ):
        """Predict one article and return its risk information."""

        if self.pipeline is None:
            raise RuntimeError("Predictor has no fitted pipeline.")

        meta_features = self.metadata.get("features", {})
        cols = list(meta_features.get("numeric_columns", []))

        # Handle missing engagement safely.
        # Zero is used only as a placeholder for the existing
        # model's input. Unknown engagement remains None
        # when calculating and displaying the risk score.

        if tweet_count is None:
            model_shares = 0
        else:
            try:
                model_shares = max(0, int(tweet_count))
            except (TypeError, ValueError, OverflowError):
                model_shares = 0
                tweet_count = None

        row = pd.DataFrame.from_records([
            {
                ft.TEXT_COLUMN: ft.join_text(title, body),
                "title": str(title),
                "tweet_count": model_shares,
            }
        ])

        numeric, _ = ft.build_features(
            row,
            use_social=meta_features.get("social", True),
            use_stylistic=meta_features.get("stylistic", True),
        )

        # Match the features used during model training.
        if cols:
            numeric = numeric.reindex(
                columns=cols,
                fill_value=0.0,
            )

        X = row[[ft.TEXT_COLUMN]].copy()

        if cols:
            X = pd.concat([X, numeric], axis=1)

        # Obtain model prediction.
        label = int(self.pipeline.predict(X)[0])

        p_fake = float(
            np.clip(
                training.score_signal(self.pipeline, X)[0],
                0.0,
                1.0,
            )
        )

        label_name = self.metadata.get(
            "label_map", {}
        ).get(
            str(label),
            "Likely misinformation" if label == 1
            else "Likely reliable",
        )

        # Keep missing engagement as None.
        risk = self.compute_risk_score(
            p_fake,
            tweet_count,
        )

        details = {
            "raw_score": p_fake,
            "features_used": meta_features,
            "tweet_count": tweet_count,
            "model": self.metadata.get(
                "model_name", "unknown"
            ),
            **risk,
        }

        return label, label_name, p_fake, details

    def score_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess the dataset using the saved pipeline."""

        df = df.copy().reset_index(drop=True)

        meta_features = self.metadata.get("features", {})
        cols = list(meta_features.get("numeric_columns", []))

        use_social = bool(
            meta_features.get("social", True)
        )

        use_stylistic = bool(
            meta_features.get("stylistic", True)
        )

        text = ft.build_text_series(df)

        numeric, _ = ft.build_features(
            df,
            use_social=use_social,
            use_stylistic=use_stylistic,
        )

        if cols:
            numeric = numeric.reindex(
                columns=cols,
                fill_value=0.0,
            )

        X = text.to_frame(ft.TEXT_COLUMN).reset_index(drop=True)

        if cols:
            X = pd.concat(
                [X, numeric.reset_index(drop=True)],
                axis=1,
            )

        y_pred = self.pipeline.predict(X).astype(int)

        p_fake = np.clip(
            training.score_signal(self.pipeline, X),
            0.0,
            1.0,
        )

        keep = [
            c for c in (
                "id",
                "title",
                "news_url",
                "source",
                "tweet_count",
            )
            if c in df.columns
        ]

        out = df[keep].copy()

        out["pred_label"] = y_pred
        out["p_fake"] = p_fake

        # Preserve missing engagement where applicable.
        if "tweet_count" in out.columns:
            share_counts = pd.to_numeric(
                out["tweet_count"],
                errors="coerce",
            )
        else:
            share_counts = pd.Series(
                np.nan,
                index=out.index,
            )
            out["tweet_count"] = share_counts

        risk_rows = []

        for p, t in zip(p_fake, share_counts):
            shares = None if pd.isna(t) else max(0, int(t))

            risk_rows.append(
                self.compute_risk_score(
                    float(p),
                    shares,
                )
            )

        out["content_risk"] = [
            r["content_risk"] for r in risk_rows
        ]

        out["propagation_risk"] = [
            r["propagation_risk"] for r in risk_rows
        ]

        out["risk_score"] = [
            r["risk_score"] for r in risk_rows
        ]

        out["risk_level"] = [
            r["risk_level"] for r in risk_rows
        ]

        out["engagement_known"] = [
            r["engagement_known"] for r in risk_rows
        ]

        return out

    def compute_risk_score(
        self,
        p_fake: float,
        tweet_count: int | None,
    ) -> dict:
        """
        Calculate misinformation risk.

        Content risk:
            Rc = P(False | Claim) * 100

        Spread risk:
            Rs = Normalised engagement score

        Combined risk:
            R = 0.6Rc + 0.4Rs

        The combined score is calculated only when
        engagement data is available.
        """

        rs = self.metadata.get("risk_scoring", {})

        alpha = float(rs.get("alpha", 0.6))
        beta = float(rs.get("beta", 0.4))

        content_risk = float(
            np.clip(p_fake, 0.0, 1.0)
        ) * 100.0

        risk_levels = rs.get("risk_levels", [])

        def get_risk_level(score: float) -> str:
            for candidate in risk_levels:
                if (
                    candidate.get("min", 0)
                    <= score
                    <= candidate.get("max", 100)
                ):
                    return candidate["name"]

            # Fallback to the dashboard's risk thresholds.
            if score >= 70:
                return "High"
            if score >= 40:
                return "Medium"
            return "Low"

        # Unknown shares:
        # Return content-only risk, not a 60/40 composite.
        if tweet_count is None:
            risk_score = content_risk
            level = get_risk_level(risk_score)

            return {
                "risk_score": risk_score,
                "risk_level": level,
                "content_risk": content_risk,
                "propagation_risk": None,
                "engagement_known": False,
                "alpha": alpha,
                "beta": beta,
                "risk_levels": risk_levels,
            }

        # Known shares:
        # Calculate the normalised spread risk.
        safe_shares = max(0, int(tweet_count))

        log_min = rs.get("log_tweet_min")
        log_max = rs.get("log_tweet_max")

        if (
            log_min is not None
            and log_max is not None
            and log_max > log_min
        ):
            log_value = float(
                np.log1p(safe_shares)
            )

            propagation_risk = (
                (log_value - log_min)
                / (log_max - log_min)
            ) * 100.0
        else:
            propagation_risk = 0.0

        propagation_risk = float(
            np.clip(
                propagation_risk,
                0.0,
                100.0,
            )
        )

        # Apply the existing FYP formula.
        risk_score = (
            alpha * content_risk
            + beta * propagation_risk
        )

        risk_score = float(
            np.clip(risk_score, 0.0, 100.0)
        )

        level = get_risk_level(risk_score)

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "content_risk": content_risk,
            "propagation_risk": propagation_risk,
            "engagement_known": True,
            "alpha": alpha,
            "beta": beta,
            "risk_levels": risk_levels,
        }

    def model_summary(self) -> str:
        """Describe the features used by the saved model."""

        feat = self.metadata.get("features", {})
        used = []

        if feat.get("text"):
            used.append("title text (TF-IDF)")

        if feat.get("social"):
            used.append("social (tweet volume)")

        if feat.get("stylistic"):
            used.append("stylistic (title stats)")

        return (
            f"{self.metadata.get('model_name', '?')} "
            f"— features: {', '.join(used) or 'none'}"
        )
