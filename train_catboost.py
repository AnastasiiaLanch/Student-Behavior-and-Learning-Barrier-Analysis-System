"""Train CatBoostClassifier on train_imputed.csv, evaluate on test_imputed.csv.

Usage:
    python3 train_catboost.py

Saves model + metrics to model/catboost/
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from config import DATA_DIR as RSRC
from config import CATBOOST_DIR as OUT
OUT.mkdir(parents=True, exist_ok=True)

RANDOM = 42

# Загрузка
df_train = pd.read_csv(RSRC / "train_imputed.csv")
df_test = pd.read_csv(RSRC / "test_imputed.csv")

# Идентификаторы - не признаки
id_cols = ["code_module", "code_presentation", "id_student"]

# Утечка target - эти признаки производные от целевой переменной
leak_cols = ["days_until_dropout", "dropout_week", "course_completion_ratio"]

# Категориальные признаки
cat_cols = ["gender", "highest_education", "age_band", "disability"]

# Все признаки кроме идентификаторов и target
feature_cols = [c for c in df_train.columns if c not in id_cols + leak_cols + ["target"]]

# Проверка: все ли категориальные есть в признаках
cat_cols = [c for c in cat_cols if c in feature_cols]

X_train_full = df_train[feature_cols]
y_train_full = df_train["target"]
X_test = df_test[feature_cols]
y_test = df_test["target"]

# Индексы категориальных признаков в X
cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]

print(f"Признаков: {len(feature_cols)}")
print(f"  из них категориальных: {len(cat_indices)} — {cat_cols}")
print(f"Train: {len(X_train_full)} строк, Test: {len(X_test)} строк")

# 2. Hold-out валидация из train
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM,
)
print(f"Train после split: {len(X_train)}, Val: {len(X_val)}")

# Обучение
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=RANDOM,
    auto_class_weights="Balanced",
    cat_features=cat_indices,
    verbose=200,
    thread_count=-1,
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    plot=False,
)

best_iter = model.get_best_iteration()
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

print("\n - Test Metrics -")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
print(f"\nConfusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Feature importance
importance = model.get_feature_importance()
fi = sorted(
    [{"feature": feature_cols[i], "importance": round(float(importance[i]), 4)}
     for i in range(len(feature_cols))],
    key=lambda x: x["importance"],
    reverse=True,
)

# Сохранение
model.save_model(str(OUT / "model.cbm"))

params_info = {
    "model": "CatBoostClassifier",
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "best_iteration": int(best_iter),
    "auto_class_weights": "Balanced",
    "early_stopping_rounds": 50,
    "random_state": RANDOM,
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "features_total": len(feature_cols),
    "cat_features": cat_cols,
}

with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUT / "confusion.json", "w") as f:
    json.dump(cm, f, indent=2)

with open(OUT / "feature_importance.json", "w") as f:
    json.dump(fi, f, indent=2)

with open(OUT / "params.json", "w") as f:
    json.dump(params_info, f, indent=2)

print(f"\nМодель и метрики сохранены в {OUT}")
