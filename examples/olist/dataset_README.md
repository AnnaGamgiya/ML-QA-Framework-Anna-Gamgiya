---
license: cc-by-nc-sa-4.0
task_categories:
  - tabular-classification
tags:
  - e-commerce
  - logistics
  - delivery-prediction
  - brazil
size_categories:
  - 10K<n<100K
configs:
  - config_name: analytical
    default: true
    data_files:
      - split: full
        path: olist_analytical.parquet
  - config_name: train
    data_files:
      - split: train
        path: Xtrain.parquet
  - config_name: test
    data_files:
      - split: test
        path: Xtest.parquet
---

# Olist Delivery Risk — analytical dataset

Registered automatically by the MLOps pipeline of the
[Olist Delivery Risk Radar](https://huggingface.co/spaces/samuelalex37/olist-delivery-risk)
project.

## What this contains

**96,470 delivered orders** from the Brazilian marketplace Olist, assembled from
the nine raw Kaggle tables into a single modelling table.

> **Note on row counts.** This repository holds the full table *and* the
> train/test partitions derived from it. Use the configuration selector above to
> view one at a time — viewing them together would count the same orders twice.

| Configuration | Rows | Contents |
|---|---|---|
| `analytical` | 96,470 | The complete modelling table, 53 columns |
| `train` | 77,176 | Training features only, 34 columns |
| `test` | 19,294 | Held-out features only, 34 columns |

The targets are stored separately as `ytrain.csv` and `ytest.csv`.

## The prediction task

Predict whether an order will be **delivered after the date promised to the
customer at checkout**. 8.11% of orders are late.

## How it was built

Nine relational tables were joined on five keys. Item-level rows were aggregated
to one row per order. Most significantly, **`distance_km` was engineered** — the
great-circle distance between seller and buyer, computed with the haversine
formula from 1,000,163 GPS readings after filtering coordinates that fell outside
Brazil's borders.

## Leakage

Columns recorded *after* checkout are excluded, because the prediction is needed
at checkout. The pipeline asserts this before every training run and stops if a
banned column reappears:

`order_delivered_customer_date`, `order_delivered_carrier_date`,
`order_approved_at`, `order_estimated_delivery_date`, `order_status`,
`actual_delivery_days`, `days_late`, `review_score`, `purchase_year`

## The split is frozen deliberately

The train/test partition is computed once and published, rather than recomputed
per training run. If every run split independently, two models could not be
compared fairly — a difference in score might mean nothing more than one model
receiving an easier test set.

## Source

Derived from the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
licensed CC BY-NC-SA 4.0. This derived dataset carries the same licence.
