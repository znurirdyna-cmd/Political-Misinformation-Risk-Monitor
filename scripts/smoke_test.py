"""Smoke test for the training + prediction pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import Predictor

p = Predictor.load()
samples = [
    ("BREAKING: Court Orders Obama To Pay $400 Million In Restitution", 150),
    ("Netherlands Beat Italy 2-0 In U20 Friendly", 2),
    ("Romney makes pitch, hoping to close deal", 5),
    ("Oscar Pistorius Attempts To Commit Suicide", 22),
]
for (title, tweets) in samples:
    label, name, proba, details = p.predict(title, None, tweets)
    print(
        f"{title[:52]:<54} tweets={tweets:<5} -> {name:<5} ({proba:.1%})  "
        f"risk={details['risk_level']:<6} ({details['risk_score']:.1f}, "
        f"Rc={details['content_risk']:.1f}, Rp={details['propagation_risk']:.1f})"
    )
print("predictor OK")