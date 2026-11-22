"""
Production prediction: extract features for selected users, impute, predict with CatBoost.

Usage:
    python3 production_predict.py

Output:
    data/prod_dataset_first_ss.csv  — features (3 rows × 45 cols)
    data/prod_predictions.json      — predictions + barriers per user
"""
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import extract_features, impute_srs
from db import DB_CONFIG


def run(max_session: int, user_ids: list[int], dataset_name: str, output_predictions: str):
    print(f"=== Step 1: extract_features(max_session={max_session}) ===")
    df = extract_features(DB_CONFIG, max_session=max_session)
    print(f"  All users: {len(df)}")

    # Filter to selected users
    df = df[df["user_id"].isin(user_ids)].copy()
    print(f"  Filtered to {user_ids}: {len(df)} rows")
    if len(df) < len(user_ids):
        found = set(df["user_id"].values)
        missing = set(user_ids) - found
        print(f"  WARNING: users not found: {missing}")

    if df.empty:
        print("  No data, aborting.")
        return

    print(f"\n=== Step 2: impute_srs ===")
    df = impute_srs(df, DB_CONFIG)

    # Save dataset
    out_dir = Path("data") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{dataset_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(df)} rows × {len(df.columns)} cols)")

    print(f"\n=== Step 3: CatBoost inference ===")
    model_path = Path("model/catboost/model.cbm")
    fi_path = Path("model/catboost/feature_importance.json")
    bs_path = Path("model/catboost/pipeline/barrier_stats.json")

    from catboost import CatBoostClassifier, Pool

    with open(fi_path) as f:
        fi = json.load(f)
    model_cols = [item["feature"] for item in fi]

    cat_features = ["gender", "highest_education", "age_band", "disability"]
    cat_indices = [model_cols.index(c) for c in cat_features]

    X = df[model_cols].copy()
    pool = Pool(X, cat_features=cat_indices)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    proba = model.predict_proba(pool)[:, 1]
    pred = model.predict(pool)

    # Barrier thresholds
    barrier_stats = {}
    if bs_path.exists():
        with open(bs_path) as f:
            barrier_stats = json.load(f)

    barrier_names = [
        "low_engagement_score",
        "knowledge_gap_score",
        "consistency_score",
        "cognitive_overload_score",
    ]
    barrier_labels = {
        "low_engagement_score": "Low Engagement",
        "knowledge_gap_score": "Knowledge Gap",
        "consistency_score": "Consistency",
        "cognitive_overload_score": "Cognitive Overload",
    }

    # Build per-user results
    predictions = []
    for i, uid in enumerate(df["user_id"].values):
        user_result = {
            "user_id": int(uid),
            "probability": round(float(proba[i]), 4),
            "prediction": int(pred[i]),
        }

        # Barriers
        barriers = []
        for bname in barrier_names:
            if bname in df.columns:
                val = float(df[bname].iloc[i])
                p50 = barrier_stats.get(bname, {}).get("dropout_p50", 0)
                p75 = barrier_stats.get(bname, {}).get("dropout_p75", 0)
                if val < p50:
                    level, icon, color = "none", "✓", "#238636"
                elif val < p75:
                    level, icon, color = "possible", "⚠", "#d29922"
                else:
                    level, icon, color = "formed", "🔴", "#f85149"
                barriers.append({
                    "label": barrier_labels.get(bname, bname),
                    "name": bname,
                    "value": round(val, 4),
                    "p50": round(p50, 4),
                    "p75": round(p75, 4),
                    "level": level,
                    "icon": icon,
                    "color": color,
                })
        user_result["barriers"] = barriers
        predictions.append(user_result)

    # Save predictions
    out_json = out_dir / output_predictions
    with open(out_json, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Predictions saved: {out_json}")

    # Print summary
    print(f"\n=== Results ===")
    for p in predictions:
        risk = "⚠ AT RISK" if p["probability"] >= 0.5 else "✓ on track"
        print(f"  user_id={p['user_id']}  prob={p['probability']:.2%}  {risk}")
        for b in p["barriers"]:
            if b["level"] != "none":
                print(f"    {b['icon']} {b['label']}: {b['value']:.4f} ({b['level']})")


if __name__ == "__main__":
    run(
        max_session=1,
        user_ids=[4, 6, 7, 8],
        dataset_name="prod_dataset_first_ss",
        output_predictions="prod_predictions_first_ss.json",
    )
