"""Полный пересчёт снапшотов из БД и предсказаний по рубежам сессий.

Запуск: python3 run_inference_all_ss.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import os


def load_env(path=None):
    from pathlib import Path as _P
    path = path or str(_P(__file__).resolve().parent / ".env")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


load_env()

import pymysql

from db import DB_CONFIG
from extract_features import extract_features, impute_srs
from production_predict import run as run_predict

ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
from config import DATA_DIR as DATA


def sessions_done() -> dict:
    """user_id -> число уникальных завершённых сессий."""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, COUNT(DISTINCT session_num) k FROM test_sessions "
                "WHERE completed_at IS NOT NULL GROUP BY user_id"
            )
            return {int(r["user_id"]): int(r["k"]) for r in cur.fetchall()}
    finally:
        conn.close()


def main():
    print(" Снапшоты по сессиям (все участники) ")
    for i in range(1, 7):
        df = extract_features(DB_CONFIG, max_session=i)
        df = impute_srs(df, DB_CONFIG)
        out = DATA / ("srs_features_imputed_%s_ss.csv" % ORD[i])
        df.to_csv(out, index=False)
        print("  %-40s %2d строк x %d колонок" % (out.name, len(df), len(df.columns)))

    k_by_user = sessions_done()
    print("\n Предсказания по рубежам (k-1) ")
    for i in range(1, 6):
        eligible = sorted(u for u, k in k_by_user.items() if k >= i + 1)
        if not eligible:
            print("  рубеж %d: нет участников, пропуск" % i)
            continue
        print("  рубеж %d: max_session=%d, users=%s" % (i, i, eligible))
        run_predict(
            max_session=i,
            user_ids=eligible,
            dataset_name="prod_dataset_%s_ss" % ORD[i],
            output_predictions="prod_predictions_%s_ss.json" % ORD[i],
        )

    print("\n Итог ")
    for i in range(1, 6):
        j = DATA / ("prod_predictions_%s_ss.json" % ORD[i])
        if not j.exists():
            continue
        with open(j) as f:
            preds = json.load(f)
        print("  %-34s %d записей" % (j.name, len(preds)))


if __name__ == "__main__":
    main()
