# Student Behavior and Learning Barrier Analysis System

Analyzing participant behavior, identifying learning barriers.
Predicting participant dropout from behaviour in testing system,
using a model trained on the OULAD dataset.

---

## How it works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   MySQL DB   │────▶│  extract_    │────▶│  CatBoost    │
│  (8 tables)  │     │  features.py │     │  model.cbm   │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │
                     ┌──────▼───────┐     ┌──────▼───────┐
                     │  impute_srs  │     │  predictions │
                     │  (43 feats)  │     │  + barriers  │
                     └──────┬───────┘     └──────┬───────┘
                            │                     │
                     ┌──────▼───────┐     ┌──────▼───────┐
                     │  nginx :443  │────▶│  FastAPI     │
                     │  reverse px  │     │  :8003       │
                     └──────────────┘     └──────────────┘
```

## Project structure

```
srs/
├── config.py                # paths and settings from the environment (.env)
├── main.py                  # FastAPI: routes, rendering, model page, project tree
├── auth.py                  # Authentication (SHA-256 + salt), sessions
├── db.py                    # MySQL connection + DB_CONFIG (reads .env)
├── task_generator.py        # Task generation (6 blocks × 3 difficulty levels)
├── extract_features.py      # Feature extraction from DB + impute_srs()
├── production_predict.py    # Production pipeline: extract → impute → predict
├── run_inference_all_ss.py  # Recalculate every boundary: snapshots + predictions
├── train_catboost.py        # CatBoost training on OULAD
├── train_xgb.py             # Alternative: XGBoost (hidden on the page)
├── save_pipeline.py         # Save pipeline parameters
├── setup_tables.py          # Create DB schema
├── create_tables.py         # DDL for tables
├── demo_generator.py        # Demo data generator
├── install.sh               # Install dependencies and service
├── apply_nginx.sh           # nginx configuration
├── srs-api.service          # systemd unit
├── templates/
│   ├── model.html           # /srs/model — dataset, training, predictions, tree
│   ├── test.html            # /srs/test — testing UI (6 sessions)
│   ├── register.html        # /srs/register — demographics form
│   ├── enter.html           # /srs/enter — login form
│   └── schema_diagram.svg   # OULAD table relations diagram
├── model/
│   ├── catboost/
│   │   ├── model.cbm        # Trained model
│   │   ├── metrics.json     # Accuracy, ROC AUC, PR AUC
│   │   ├── params.json      # Hyperparameters
│   │   ├── confusion.json   # Confusion matrix
│   │   ├── feature_importance.json
│   │   └── pipeline/
│   │       ├── build_features.py      # OULAD feature engineering (reference)
│   │       ├── impute.py              # OULAD imputation (reference)
│   │       ├── imputation_params.json # Imputation values
│   │       ├── barrier_stats.json     # p50/p75 thresholds for barriers
│   │       ├── barrier_feature_refs.json
│   │       └── feature_spec.json
│   └── xgb/                 # XGBoost model (hidden on the page)
└── data/
    ├── *.csv                        # Raw OULAD (studentInfo, studentVle, ...)
    ├── oulad.zip                    # OULAD archive (studentVle 453 MB inside)
    ├── merged_features.csv          # Merged OULAD dataset
    ├── train.csv / test.csv         # 80/20 stratified split
    ├── train_imputed.csv            # After imputation
    ├── test_imputed.csv
    ├── srs_features.csv             # Features from DB (all users)
    ├── srs_features_imputed_first_ss.csv   # Snapshots per session
    ├── ... sixth_ss.csv             # (first..sixth, all participants)
    └── predictions/                 # all per-boundary predictions
        ├── prod_dataset_first_ss.csv      # Features at the boundary
        ├── prod_dataset_second_ss.csv     # ... third/fourth/fifth
        ├── prod_predictions_first_ss.json # Predictions + barriers
        └── prod_predictions_second_ss.json # ... third/fourth/fifth
```

> **Note on the published archive.** This copy intentionally excludes three
> infrastructure modules: `auth.py` (password hashing and sessions), `config.py`
> (paths and settings from the environment) and `db.py` (MySQL connection).
> `main.py` imports all three, so the application will not start without them —
> add your own equivalents or copy them from a private deployment before running.
> `data/` and `.env` are not included either (see Quick start below).

## Database

**Connection:** host, port, user, password and schema are set in `.env`
(`SRS_DB_HOST`, `SRS_DB_PORT`, `SRS_DB_USER`, `SRS_DB_NAME`, `SRS_DB_PASS`).
No hard-coded addresses or passwords in code.
**Schema:** SRS

### Tables

| Table | Purpose | Rows |
|-------|---------|:----:|
| `auth_users` | Accounts (+ `is_admin` flag) | 9 |
| `auth_sessions` | Login sessions (token + expires) | 9 |
| `student_info` | Demographics: gender, age, education | 6 |
| `task_templates` | Task templates | 18 |
| `template_difficulty` | Difficulty variants | 54 |
| `test_sessions` | Test sessions (6 × 9 tasks) | 25 |
| `task_instances` | Issued tasks: answers, time, correctness | 234 |
| `user_events` | Clickstream: clicks, input, submits | 434 |

## Model

### 1. Training (OULAD)

Trained on the Open University dataset (32,593 students, 7 courses, 10.7M VLE events).

```bash
cd data && python3 build_features.py   # merged_features.csv + train/test
cd data && python3 impute.py           # train_imputed / test_imputed
python3 save_pipeline.py         # feature_spec, imputation_params, barrier_stats
python3 train_catboost.py        # CatBoost -> model.cbm
```

**Test metrics (6,519 students):**

| Metric | Value |
|--------|:-----:|
| Accuracy | 87.96% |
| Precision | 75.51% |
| Recall | 90.79% |
| ROC AUC | 95.08% |
| PR AUC | 88.89% |

**Hyperparameters:** iterations=2000, lr=0.05, depth=6, auto_class_weights=Balanced, early_stopping=50

### 2. Features

| Group | Count | Examples |
|-------|:-----:|----------|
| Demographics | 6 | gender, highest_education, age_band, disability, num_of_prev_attempts, studied_credits |
| Performance | 9 | avg_score, score_trend, performance_change, num_low_scores, assessment_success_rate |
| Engagement | 11 | total_clicks, activity_slope, activity_drop_ratio, early/late_engagement |
| Consistency | 4 | avg_gap_days, max_gap_days, activity_variance, inactive_days_ratio |
| Course Load | 5 | num_assessments, assessment_density, deadline_density |
| Temporal | 1 | days_registered |
| Engineered | 2 | learning_efficiency, resource_diversity |
| Barrier Scores | 4 | low_engagement_score, knowledge_gap_score, consistency_score, cognitive_overload_score |
| Flags | 3 | is_no_activity, is_no_assessment, missing_barrier_flag |

### 3. Where features come from

`extract_features.py` runs four queries and aggregates them:

| Source | Contribution |
|--------|--------------|
| `student_info` | demographics (gender, education, age band, disability) |
| `test_sessions` | temporal features, number of completed sessions |
| `task_instances` + `task_templates` | performance, timing, task types per block |
| `user_events` | engagement: clicks, active days, gaps, dynamics |

`max_session=N` filters `test_sessions.session_num`, `task_instances.session_id`
and `user_events.session_id`, so only data up to session N is used.

## Production: per-boundary predictions

Rule: a participant who completed `k` sessions gets `k-1` predictions —
one per transition (after session 1, 2, ... k-1).

Full recalculation in one command:

```bash
python3 run_inference_all_ss.py
```

What it does:
1. Refreshes six snapshots `data/srs_features_imputed_{first..sixth}_ss.csv` for all participants.
2. Reads completed sessions per user from the DB.
3. For each boundary 1...5 calls `production_predict.run(max_session=i, user_ids=...)`.
4. Writes results to `data/predictions/`:
   `prod_dataset_<name>.csv` and `prod_predictions_<name>.json`.

Single boundary manually:

```python
from production_predict import run
run(max_session=2, user_ids=[1, 2, 3],
    dataset_name="prod_dataset_second_ss",
    output_predictions="prod_predictions_second_ss.json")
```

**`prod_predictions_*.json` contents:** array per participant with
`user_id`, `probability` (0...1), `prediction` (0/1, threshold 0.5) and `barriers`:

| Barrier field | Meaning |
|---------------|---------|
| `label` | name (Low Engagement, Knowledge Gap, Consistency, Cognitive Overload) |
| `name` | feature name in the dataset |
| `value` | participant value |
| `p50` / `p75` | OULAD reference thresholds |
| `level` | none / possible / formed |

## Prediction results (example run)

Aggregates from the current run. Participant identifiers and their individual
probabilities are not published — those are personal assessments.

| Boundary | Participants | At risk (p ≥ 0.5) | Mean probability |
|----------|:------------:|:------------------:|:----------------:|
| after session 1 | 4 | 1 | 0.370 |
| after session 2 | 4 | 3 | 0.541 |
| after session 3 | 4 | 3 | 0.520 |
| after session 4 | 3 | 1 | 0.467 |
| after session 5 | 3 | 3 | 0.549 |

The most frequent barriers are **Consistency** and **Cognitive Overload** at every boundary;
**Knowledge Gap** appears less often (1–2 participants per boundary).

Record format (values are illustrative):

```json
[
  {
    "user_id": "user-1",
    "probability": 0.42,
    "prediction": 0,
    "barriers": [
      {
        "label": "Consistency",
        "name": "consistency_score",
        "value": 0.031,
        "p50": 0.0219,
        "p75": 0.0404,
        "level": "possible",
        "color": "#d29922"
      }
    ]
  }
]
```

## Routes

| URL | Access | Description |
|-----|--------|-------------|
| `/srs/enter` | Everyone | Login form |
| `/srs/register` | Authenticated | Demographics form |
| `/srs/test` | Authenticated + profile | Testing (6 sessions × 9 tasks) |
| `/srs/model` | is_admin | Dataset, training, predictions, tree |
| `/srs/logout` | Everyone | Logout |

## Accounts

```python
from auth import hash_password
from db import execute
pw_hash = hash_password("password")
execute("INSERT INTO auth_users (username, password_hash, section) VALUES (%s,%s,%s)",
        ("login", pw_hash, "srs"))
```


## OULAD Dataset

**Source:** Open University, UCI ML Repository (CC BY 4.0)
**Composition:** 32,593 students, 7 modules, 22 presentations, 10.7M VLE clicks
**Location:** `data/` (`oulad.zip` archive + extracted CSV)
**Target:** `Withdrawn=1`, `Pass/Fail/Distinction=0` (dropout rate 31.2%)

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, FastAPI, uvicorn |
| Model | CatBoostClassifier (AUC 0.9508) |
| Database | MySQL 8 (Timeweb, managed), 8 tables |
| Frontend | Jinja2, dark theme (GitHub style) |
| Web server | nginx reverse proxy, Let's Encrypt SSL |
| Runtime | systemd service `srs-api` |
