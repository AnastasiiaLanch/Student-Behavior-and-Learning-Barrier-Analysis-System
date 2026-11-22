"""Train XGBoostClassifier on train_imputed.csv, evaluate on test_imputed.csv.

Encodings:
  - gender: Label Encoding
  - highest_education: One-Hot Encoding
  - age_band: Ordinal Encoding (0-35→0, 35-55→1, 55<=→2)
  - disability: Label Encoding

Saves model + metrics to model/xgb/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb

from config import DATA_DIR as RSRC
from config import XGB_DIR as OUT
OUT.mkdir(parents=True, exist_ok=True)

RANDOM = 42

# Загрузка
df_train = pd.read_csv(RSRC / "train_imputed.csv")
df_test = pd.read_csv(RSRC / "test_imputed.csv")

# Идентификаторы
id_cols = ["code_module", "code_presentation", "id_student"]

# Утечка target
leak_cols = ["days_until_dropout", "dropout_week", "course_completion_ratio"]

# Все признаки кроме идентификаторов, leaks и target
feature_cols = [c for c in df_train.columns if c not in id_cols + leak_cols + ["target"]]

def encode(df):
    """Применяет кодирование категориальных признаков, возвращает X (DataFrame)."""
    X = df[feature_cols].copy()

    # 1. gender — Label Encoding (F→0, M→1 — по алфавиту)
    le = LabelEncoder()
    X["gender"] = le.fit_transform(X["gender"])
    print(f"  gender: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 2. highest_education — One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, dtype=np.int64)
    edu_encoded = ohe.fit_transform(X[["highest_education"]])
    edu_cols = [f"edu_{c}" for c in ohe.categories_[0]]
    edu_df = pd.DataFrame(edu_encoded, columns=edu_cols, index=X.index)
    X = pd.concat([X.drop(columns=["highest_education"]), edu_df], axis=1)
    print(f"  highest_education: {list(edu_cols)} ({len(edu_cols)} колонок)")

    # 3. age_band — Ordinal Encoding (ручной порядок)
    age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
    X["age_band"] = X["age_band"].map(age_map)
    print(f"  age_band: {age_map}")

    # 4. disability — Label Encoding (N→0, Y→1 по алфавиту)
    le2 = LabelEncoder()
    X["disability"] = le2.fit_transform(X["disability"])
    print(f"  disability: {dict(zip(le2.classes_, le2.transform(le2.classes_)))}")

    return X

print("Кодирование train:")
X_train_full = encode(df_train)
print(f"\nКодирование test:")
X_test = encode(df_test)

y_train_full = df_train["target"]
y_test = df_test["target"]

print(f"\nПризнаков после кодирования: {X_train_full.shape[1]}")
print(f"Train: {len(X_train_full)} строк, Test: {len(X_test)} строк")

# Hold-out валидация (тот же random_state)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM,
)
print(f"Train после split: {len(X_train)}, Val: {len(X_val)}")

# Обучение
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=RANDOM,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=200,
)

best_iter = model.best_iteration
print(f"\nBest iteration: {best_iter}")

# Оценка на test
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
}

print("\n── Test Metrics ──")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
print(f"\nConfusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Feature importance
fi_all = model.feature_importances_
fi = sorted(
    [{"feature": X_train.columns[i], "importance": round(float(fi_all[i]), 4)}
     for i in range(len(fi_all))],
    key=lambda x: x["importance"],
    reverse=True,
)

# Сохранение
model.save_model(str(OUT / "model.json"))

params_info = {
    "model": "XGBoostClassifier",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "best_iteration": int(best_iter),
    "scale_pos_weight": round(float(scale_pos_weight), 2),
    "early_stopping_rounds": 50,
    "random_state": RANDOM,
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "features_total": X_train.shape[1],
    "encoding": {
        "gender": "Label Encoding",
        "highest_education": "One-Hot Encoding",
        "age_band": "Ordinal Encoding (0-35→0, 35-55→1, 55<=→2)",
        "disability": "Label Encoding",
    },
}

with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUT / "confusion.json", "w") as f:
    json.dump(cm, f, indent=2)

with open(OUT / "feature_importance.json", "w") as f:
    json.dump(fi, f, indent=2)

with open(OUT / "params.json", "w") as f:
    json.dump(params_info, f, indent=2, ensure_ascii=False)

print(f"\nМодель и метрики сохранены в {OUT}")
