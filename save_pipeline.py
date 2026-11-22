import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import DATA_DIR as RSRC
from config import CATBOOST_PIPELINE_DIR as OUT


train_raw = pd.read_csv(RSRC / "train.csv")
train_imp = pd.read_csv(RSRC / "train_imputed.csv")

# Список и порядок признаков
id_cols = ["code_module", "code_presentation", "id_student"]
leak_cols = ["days_until_dropout", "dropout_week", "course_completion_ratio"]
cat_features_list = ["gender", "highest_education", "age_band", "disability"]

feature_cols = [c for c in train_imp.columns
                if c not in id_cols + leak_cols + ["target", "final_result"]]

cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features_list]

print(f"Признаков: {len(feature_cols)}")
print(f"  Категориальных: {len(cat_indices)} (индексы {cat_indices})")
print(f"  Исключено: {id_cols + leak_cols}")

feature_spec = {
    "feature_names": feature_cols,
    "n_features": len(feature_cols),
    "cat_features": cat_features_list,
    "cat_indices": cat_indices,
    "excluded": {
        "identifiers": id_cols,
        "leaks": leak_cols,
    },
}

# Параметры заполнения пропусков
imputation_params = {}

# Категориальные демо → 'Unknown'
imputation_params["cat_demo_fill"] = {c: "Unknown" for c in cat_features_list}

# Числовые демо: медиана из train (до заполнения)
for col in ["num_of_prev_attempts", "studied_credits"]:
    imputation_params[f"median_{col}"] = float(train_raw[col].median())

# VLE-признаки: → 0
vle_cols = [
    "total_clicks", "avg_clicks_per_day", "active_days", "num_resources_visited",
    "weekly_clicks_mean", "weekly_clicks_std", "activity_slope", "activity_drop_ratio",
    "engagement_change", "early_engagement", "late_engagement",
]
imputation_params["vle_fill"] = {c: 0 for c in vle_cols}

# Consistency: avg_gap_days, max_gap_days → max_gap из train, остальные → 0
max_gap_train = float(train_raw["max_gap_days"].max())
imputation_params["max_gap_days_train"] = max_gap_train
imputation_params["consistency_fill"] = {
    "avg_gap_days": max_gap_train,
    "max_gap_days": max_gap_train,
    "activity_variance": 0,
    "inactive_days_ratio": 0,
}

# Performance: → 0
perf_cols = [
    "avg_score", "min_score", "max_score", "score_std", "score_trend",
    "performance_change", "num_assessments_completed", "num_low_scores",
    "assessment_success_rate",
]
imputation_params["performance_fill"] = {c: 0 for c in perf_cols}

# Temporal: → 0, затем days_until_dropout=-1 для не-дропаутов
imputation_params["temporal_fill"] = {c: 0 for c in leak_cols + ["days_registered"]}

# Course Load: медианы по (code_module, code_presentation) из train
course_cols = ["num_assessments", "avg_assessment_weight", "max_assessment_weight",
               "assessment_density", "deadline_density"]
course_medians = {}
for col in course_cols:
    med_by_course = train_raw.groupby(["code_module", "code_presentation"])[col].median()
    course_medians[col] = {f"{mod}|{pres}": float(v)
                           for (mod, pres), v in med_by_course.items()}
course_medians["_global"] = {col: float(train_raw[col].median()) for col in course_cols}
imputation_params["course_load_medians"] = course_medians

# Barrier scores: → 0
barrier_cols = ["low_engagement_score", "knowledge_gap_score", "consistency_score",
                "cognitive_overload_score"]
imputation_params["barrier_fill"] = {c: 0 for c in barrier_cols}

# Остальные: → 0
imputation_params["other_fill"] = {c: 0 for c in ["learning_efficiency", "resource_diversity"]}

# Статистики для нормализации барьерных признаков
# (minmax scaling использует min/max из train)
barrier_stats = {}
for col in barrier_cols:
    barrier_stats[col] = {
        "min": float(train_imp[col].min()),
        "max": float(train_imp[col].max()),
        "mean": float(train_imp[col].mean()),
        "std": float(train_imp[col].std()),
        # Перцентили дропаутов для порогов
        "dropout_p50": float(train_imp.loc[train_imp["target"] == 1, col].quantile(0.50)),
        "dropout_p75": float(train_imp.loc[train_imp["target"] == 1, col].quantile(0.75)),
    }

# Сохранение
OUT.mkdir(parents=True, exist_ok=True)

with open(OUT / "feature_spec.json", "w") as f:
    json.dump(feature_spec, f, indent=2, ensure_ascii=False)

with open(OUT / "imputation_params.json", "w") as f:
    json.dump(imputation_params, f, indent=2, ensure_ascii=False)

with open(OUT / "barrier_stats.json", "w") as f:
    json.dump(barrier_stats, f, indent=2, ensure_ascii=False)

print(f"\nАртефакты сохранены в {OUT}/")
for fp in sorted(OUT.iterdir()):
    print(f"  {fp.name} ({fp.stat().st_size:,} B)")
