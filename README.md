# Political Misinformation Risk Monitoring System (End-User)

A simplified, non-technical front end over the existing fake-news detection
backend. Designed for general users (not IT / ML experts) to monitor political
misinformation risk, understand how strongly news is spreading, and identify
content that requires attention.

The **backend is unchanged** — this build only restructures the user interface
and hides technical model-development details.

## Pages

| Page | Purpose |
| ---- | ------- |
| 🏠 Overview | Current risk situation: High / Medium / Low counts, risk distribution, content requiring attention |
| 🔍 Check News | Paste a headline (+ optional content + social media shares) to get an overall risk assessment |
| 📊 Misinformation Monitor | Risk distribution, spread patterns, and a searchable/filterable list of news by risk |

## Project structure

```
.
├── app.py                  # Streamlit end-user app (Overview / Check News / Monitor)
├── scripts/
│   ├── train_model.py      # CLI training entrypoint (developer only)
│   └── test_*.py           # Headless smoke tests
├── src/
│   ├── config.py           # Paths + dataset metadata
│   ├── data_loader.py      # Reads the 4 FakeNewsNet CSVs into a labelled frame
│   ├── features.py         # Text + social + stylistic feature engineering
│   ├── train.py            # sklearn pipeline (TF-IDF + numeric), training, eval
│   └── predict.py          # Predictor wrapper (single + bulk risk scoring)
├── data/                   # FakeNewsNet minimal CSVs
└── models/                 # Trained artifacts (pipeline.joblib, metadata, metrics)
```

## Quick start

```bash
run.bat
```

(Equivalent: `C:\Users\znuri\anaconda3\python.exe -m streamlit run app.py`)

Open http://localhost:8501 — the app opens on the **Overview** page. On first
load the system scores the monitored news once (a short spinner appears).

## Risk scoring (unchanged backend)

For every article the system computes:

```
Overall Risk = 0.6 × Content Risk + 0.4 × Spread Risk
```

- **Content Risk** = classifier probability the article is misinformation × 100.
- **Spread Risk** = log-scaled, min-max normalized social-media share count × 100.
- Levels: **Low** 0–39, **Medium** 40–69, **High** 70–100.

In the interface, "Propagation Risk" is presented to users as **Spread Risk** and
the classifier verdict is presented as an **AI Assessment** ("Likely
misinformation" / "Likely reliable"), never as an absolute fact-check.

## Developer notes (not shown in the UI)

- Model training is retained in `src/train.py` / `scripts/train_model.py` and is
  intentionally **not** exposed in the user navigation.
- Technical details (model name, TF-IDF, train/test split, feature groups,
  timestamps, dataset IDs) are hidden from end users.

## Test runner

```bash
python scripts\smoke_test.py      # pipeline + prediction sanity checks
python scripts\test_app.py        # all 3 user pages load without exceptions
python scripts\test_classify.py   # analyse-news flow in-app
```

> Academic prototype only — assessments are probabilistic and not
> production-grade moderation decisions.
