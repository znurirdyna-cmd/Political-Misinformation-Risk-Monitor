"""Data augmentation for the Malaysia political-misinformation dataset.

Expands the small real dataset using label-preserving text perturbations
(random word swap / deletion) and balances the two classes by generating more
variants of the under-represented class.

Usage (from the project root):
    python scripts\\augment.py            # defaults to target 300 rows
    python scripts\\augment.py --target 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MALAYSIA_FILE, MALAYSIA_AUGMENTED_FILE


def _tokens(text: str) -> list[str]:
    return [t for t in str(text).split() if t.strip()]


def _swap(tokens: list[str]) -> list[str]:
    if len(tokens) < 2:
        return tokens
    i = random.randrange(len(tokens))
    j = random.randrange(len(tokens))
    tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


def _delete(tokens: list[str]) -> list[str]:
    if len(tokens) <= 3:
        return tokens
    del tokens[random.randrange(len(tokens))]
    return tokens


def augment(text: str, ops: int = 2) -> str:
    """Return a lightly perturbed copy of ``text`` (meaning preserved)."""
    tokens = _tokens(text)
    if not tokens:
        return text
    for _ in range(ops):
        roll = random.random()
        if roll < 0.55:
            tokens = _swap(tokens)
        elif roll < 0.9:
            tokens = _delete(tokens)
    return " ".join(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment the Malaysia dataset")
    parser.add_argument("--input", default=str(MALAYSIA_FILE))
    parser.add_argument("--output", default=str(MALAYSIA_AUGMENTED_FILE))
    parser.add_argument(
        "--target", type=int, default=300,
        help="target total rows (balanced across the two classes)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.input, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df = df.reset_index(drop=True)
    df["group"] = df.index  # each original claim is its own group

    counts = df["label"].value_counts().to_dict()
    n_fake = int(counts.get(1, 0))
    n_real = int(counts.get(0, 0))
    print(f"Loaded {len(df)} rows: fake={n_fake}, real={n_real}")

    target_per_class = max(args.target // 2, n_fake, n_real)
    needed = {1: target_per_class - n_fake, 0: target_per_class - n_real}

    frames = [df]
    for label, extra in needed.items():
        if extra <= 0:
            continue
        base = df[df["label"] == label]
        variants: list[dict] = []
        i = 0
        while len(variants) < extra:
            row = base.iloc[i % len(base)]
            new_title = augment(str(row["title"]))
            if new_title != str(row["title"]) and new_title.strip():
                variants.append({
                    "title": new_title,
                    "news_url": row["news_url"],
                    "tweet_count": 0,
                    "label": label,
                    "group": int(row["group"]),
                })
            i += 1
        frames.append(pd.DataFrame(variants))

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    final = out["label"].value_counts().to_dict()
    print(
        f"Wrote {len(out)} rows to {args.output}: "
        f"fake={int(final.get(1, 0))}, real={int(final.get(0, 0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
