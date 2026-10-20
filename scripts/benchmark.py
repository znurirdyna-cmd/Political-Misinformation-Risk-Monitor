"""Benchmark: compare the Malaysia model against the USA FakeNewsNet baseline.

Trains several classifiers on both datasets in-memory (does NOT overwrite
``models/``) and prints a side-by-side comparison of test metrics. Reuses the
existing ``train()`` and ``load_prepared()`` — no ML logic is duplicated.

Usage (from the project root):
    python scripts\\benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_prepared
from src.train import TrainConfig, train

MODELS = [
    "Logistic Regression",
    "Linear SVM",
    "Random Forest",
    "Multinomial Naive Bayes",
]

DATASETS = [
    ("USA FakeNewsNet", ("politifact", "gossipcop")),
    ("Malaysia", ("malaysia",)),
]


def bench(model: str, sources: tuple[str, ...], label: str) -> dict:
    df = load_prepared(sources)
    fake = int((df["label"] == 1).sum())
    real = int((df["label"] == 0).sum())
    config = TrainConfig(
        model_name=model,
        sources=sources,
        test_size=0.25,
        max_features=20000,
    )
    result = train(df, config)
    t = result["metrics"]["test"]
    return {
        "Dataset": label,
        "Model": model,
        "Rows": len(df),
        "Fake%": fake / len(df) * 100,
        "Test rows": result["metadata"]["n_test"],
        "Accuracy": t["accuracy"],
        "Precision": t["precision"],
        "Recall": t["recall"],
        "F1": t["f1"],
        "ROC-AUC": t["roc_auc"] if t.get("roc_auc") is not None else float("nan"),
    }


def main() -> int:
    rows = []
    for label, sources in DATASETS:
        for model in MODELS:
            print(f"Training {model} on {label} ({len(load_prepared(sources)):,} rows)…")
            rows.append(bench(model, sources, label))

    table = pd.DataFrame(rows)
    order = ["Dataset", "Model", "Rows", "Fake%", "Test rows",
             "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    table = table[order].set_index(["Dataset", "Model"])

    pct_cols = ["Accuracy", "Precision", "Recall", "F1"]
    table[pct_cols] = table[pct_cols].mul(100).round(1).astype(str).add("%")
    table["Fake%"] = table["Fake%"].round(1).astype(str).add("%")
    table["ROC-AUC"] = table["ROC-AUC"].round(4)

    print("\n=== Benchmark results (test set) ===\n")
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
