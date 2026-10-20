"""Central configuration for paths and dataset metadata."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_FILES = {
    "politifact_fake": DATA_DIR / "politifact_fake.csv",
    "politifact_real": DATA_DIR / "politifact_real.csv",
    "gossipcop_fake": DATA_DIR / "gossipcop_fake.csv",
    "gossipcop_real": DATA_DIR / "gossipcop_real.csv",
}

MALAYSIA_FILE = DATA_DIR / "malaysia_news.csv"
MALAYSIA_AUGMENTED_FILE = DATA_DIR / "malaysia_news_augmented.csv"
MALAYSIA_SOURCE = "malaysia"

# This build serves Malaysia political misinformation by default. The original
# FakeNewsNet sources remain available for training via ``--sources``.
DATASET_SOURCES = ("malaysia",)

MODEL_ARTIFACTS = {
    "pipeline": MODEL_DIR / "pipeline.joblib",
    "metadata": MODEL_DIR / "metadata.json",
    "metrics": MODEL_DIR / "metrics.json",
}

LABEL_MAP = {0: "LIKELY RELIABLE", 1: "LIKELY MISINFORMATION"}