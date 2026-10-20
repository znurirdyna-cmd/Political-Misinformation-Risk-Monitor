"""Command-line entrypoint to train the FakeNewsNet classifier.

Usage (from the project root):
    python scripts\\train_model.py --model "Logistic Regression" --no-social
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_prepared  # noqa: E402
from src.train import TrainConfig, train, save_artifacts  # noqa: E402


def positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FakeNewsNet classifier")
    parser.add_argument(
        "--model",
        default="Logistic Regression",
        choices=[
            "Logistic Regression",
            "Linear SVM",
            "Random Forest",
            "Multinomial Naive Bayes",
        ],
    )
    parser.add_argument("--sources", nargs="*", default=["malaysia"])
    parser.add_argument("--no-text", action="store_true", help="disable text features")
    parser.add_argument("--no-social", action="store_true", help="disable social features")
    parser.add_argument("--no-stylistic", action="store_true", help="disable stylistic features")
    parser.add_argument("--test-size", type=positive_float, default=0.25)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--sample-frac", type=positive_float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"Loading dataset from sources: {args.sources} ...")
    df = load_prepared(tuple(args.sources))
    print(f"  rows={len(df):,}  fake={(df['label'] == 1).sum():,}  "
          f"real={(df['label'] == 0).sum():,}")

    config = TrainConfig(
        model_name=args.model,
        use_text=not args.no_text,
        use_social=not args.no_social,
        use_stylistic=not args.no_stylistic,
        sources=tuple(args.sources),
        test_size=args.test_size,
        max_features=args.max_features,
        sample_frac=args.sample_frac,
    )

    result = train(df, config)
    save_artifacts(result)

    print("\nTest metrics:")
    print(json.dumps(result["metrics"]["test"], indent=2, default=str))
    print("\nArtifacts saved to models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())