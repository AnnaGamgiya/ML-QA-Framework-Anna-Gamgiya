---
license: cc-by-nc-sa-4.0
library_name: sklearn
tags:
  - tabular-classification
  - scikit-learn
  - calibrated-classifier
  - e-commerce
  - logistics
datasets:
  - samuelalex37/olist-delivery-risk-data
metrics:
  - roc_auc
  - recall
  - brier_score
---

# Olist Delivery Risk — Gradient Boosting

Predicts whether a Brazilian e-commerce order will be **delivered later than the
date promised to the customer at checkout**.

Registered automatically by the project's MLOps pipeline.
Live application: [Olist Delivery Risk Radar](https://huggingface.co/spaces/samuelalex37/olist-delivery-risk)

## Performance

Measured on 19,294 held-out orders the model never saw
during training.

| Metric | Value |
|---|---|
| ROC-AUC | **0.8101** |
| PR-AUC | 0.3354 |
| Recall | 0.5323 |
| Precision | 0.3045 |
| F1 | 0.3874 |
| Brier score | 0.0633 |
| Decision threshold | **0.155** |

### Models compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | 0.777 | 0.219 | 0.686 | 0.332 | 0.810 | 0.336 |
| Random Forest | 0.845 | 0.273 | 0.551 | 0.365 | 0.799 | 0.330 |
| Logistic Regression | 0.657 | 0.144 | 0.655 | 0.236 | 0.710 | 0.183 |
| *"always on time"* | *0.919* | — | *0.000* | — | *0.500* | *0.081* |

That last row is the point. Only 8.11% of orders are late, so a model that
ignores every input and always answers "on time" scores **91.9% accuracy** while
catching **zero** late deliveries. Accuracy is not the metric to judge this on.

## The probabilities are calibrated

Class weighting made the raw model overstate risk by roughly **4.4×** — orders it
scored "30–40% likely late" were in reality late only 3.4% of the time.

Isotonic calibration corrected this. Mean predicted probability moved from 36.0%
to 8.11% against a true rate of
8.11%, and the Brier score improved by
about 60% while ROC-AUC was unchanged.

**A reading of 20% from this model genuinely means about one order in five.**

## The threshold comes from business cost

Not the default 0.5, and not maximum F1 — both assume a missed late delivery
costs the same as a false alarm. It doesn't: 54.1% of late deliveries attract a
1–2 star review, against 9.2% of on-time ones.

The shipped threshold of **0.155** minimises
`5 × misses + false alarms`, reducing expected cost by
about 29% versus taking no action.

## Usage

```python
import joblib, json, pandas as pd
from huggingface_hub import hf_hub_download

repo = "samuelalex37/olist-delivery-risk-model"
model = joblib.load(hf_hub_download(repo, "model.joblib"))
meta = json.load(open(hf_hub_download(repo, "model_metadata.json")))

columns = meta["numeric_features"] + meta["categorical_features"]
probability = model.predict_proba(pd.DataFrame([order])[columns])[0, 1]
flagged = probability >= meta["decision_threshold"]
```

`model.joblib` contains the **entire pipeline** — imputation, scaling, one-hot
encoding, the classifier and the calibrator — so no preprocessing code is needed
and the transformations cannot drift out of step with the model.

## Honest limitations

- **Precision is 0.30.** Roughly
  7 in every 10 flagged orders would have
  arrived on time. Acceptable only because the intervention is cheap.
- Trained on 2016–2018 courier conditions; would need periodic retraining.
- Seller-level history was excluded, since computing it safely requires
  time-aware aggregation to avoid reintroducing leakage.
- ROC-AUC 0.8101 means good ranking, not certainty. This is
  decision support, not an oracle.

## Provenance

- Trained: 2026-07-28T09:46:21
- Training rows: 77,176
- Calibration: isotonic (CalibratedClassifierCV, cv=5)
- scikit-learn: 1.9.0
- Registered by: `mlops/train.py`
