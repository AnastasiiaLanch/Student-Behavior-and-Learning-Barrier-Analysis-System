"""
Extract raw (pre-feature-engineering) features from SRS database.

Produces a DataFrame with one row per user, containing all features
that the OULAD build_features.py expects as input.

Usage:
    from extract_features import extract_features
    df = extract_features(db_config)
    df.to_csv('srs_raw_features.csv', index=False)
"""
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
from pathlib import Path


def extract_features(db_config: dict, max_session: int = None) -> pd.DataFrame:
    """Extract raw features from SRS database, one row per user.

    Args:
        db_config: MySQL connection config.
        max_session: If set, only include data up to this session number (inclusive).
    """

    conn = pymysql.connect(**db_config)
    cur = conn.cursor()

    # Load data
    cur.execute("SELECT * FROM student_info")
    si = pd.DataFrame(cur.fetchall())
    cur.execute("SELECT * FROM test_sessions")
    ts = pd.DataFrame(cur.fetchall())
    cur.execute(
        """SELECT ti.*, tt.block
           FROM task_instances ti
           JOIN task_templates tt ON ti.template_id = tt.id"""
    )
    ti = pd.DataFrame(cur.fetchall())
    cur.execute("SELECT * FROM user_events")
    ue = pd.DataFrame(cur.fetchall())
    conn.close()

    if si.empty:
        return pd.DataFrame()

    # Filter by max_session if specified
    if max_session is not None:
        if not ts.empty:
            ts = ts[ts["session_num"] <= max_session]
        if not ti.empty:
            ti = ti[ti["session_id"] <= max_session]
        if not ue.empty:
            ue = ue[ue["session_id"] <= max_session]

    # Constants
    si["studied_credits"] = 60

    # Temporal features (relative to registration date)
    si["created_at"] = pd.to_datetime(si["created_at"])
    reg_dates = dict(zip(si["user_id"], si["created_at"]))

    # date_registration = 0 (registration IS day 0 for each user)
    si["date_registration"] = 0

    # date_unregistration: last completed_at in relative days
    ts["completed_at"] = pd.to_datetime(ts["completed_at"])
    ts["rel_completed"] = ts.apply(
        lambda r: (r["completed_at"] - reg_dates.get(r["user_id"])).total_seconds() / 86400
        if pd.notna(r["completed_at"]) and r["user_id"] in reg_dates
        else np.nan,
        axis=1,
    )
    last_dates = ts.groupby("user_id")["rel_completed"].max().reset_index()
    last_dates.columns = ["user_id", "date_unregistration"]

    si = si.merge(last_dates, on="user_id", how="left")
    # days_registered: use max session_num × 30 to map into OULAD scale
    max_session = ts.groupby("user_id")["session_num"].max().reset_index()
    max_session.columns = ["user_id", "max_session_num"]
    si = si.merge(max_session, on="user_id", how="left")
    si["days_registered"] = si["max_session_num"].fillna(1).astype(float) * 30.0

    # Per-user attempts
    ts["attempt_num"] = pd.to_numeric(ts["attempt_num"], errors="coerce").fillna(1).astype(int)
    attempts = ts.groupby("user_id")["attempt_num"].max().reset_index()
    attempts.columns = ["user_id", "num_of_prev_attempts"]

    si = si.merge(attempts, on="user_id", how="left")
    si["num_of_prev_attempts"] = si["num_of_prev_attempts"].fillna(0).astype(int)

    # Assessment features (per-task, then aggregate)
    if not ti.empty:
        ti["created_at"] = pd.to_datetime(ti["created_at"])
        ti["rel_day"] = ti.apply(
            lambda r: (r["created_at"] - reg_dates.get(r["user_id"])).total_seconds() / 86400
            if r["user_id"] in reg_dates else 0,
            axis=1,
        )

        # Weighted score: is_correct × difficulty × 100 / 3
        ti["is_correct"] = pd.to_numeric(ti["is_correct"], errors="coerce").fillna(0)
        ti["difficulty"] = pd.to_numeric(ti["difficulty"], errors="coerce").fillna(1)
        ti["score"] = ti["is_correct"] * ti["difficulty"] * 100.0 / 3.0
        ti["date_submitted"] = ti["rel_day"]

        # Per-user aggregates
        perf_agg = ti.groupby("user_id").agg(
            avg_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            score_std=("score", "std"),
            num_assessments_completed=("id", "count"),
            num_low_scores=("score", lambda x: (x < 40).sum()),
            assessment_success_rate=("is_correct", "mean"),
        ).reset_index()

        # Score trend (slope of score over date_submitted)
        def _score_trend(g):
            g = g.sort_values("date_submitted")
            if len(g) < 2:
                return 0.0
            x = np.arange(len(g), dtype=float)
            y = g["score"].values.astype(float)
            slope, _ = np.polyfit(x, y, 1)
            return float(slope)

        trends = ti.groupby("user_id").apply(_score_trend).reset_index()
        trends.columns = ["user_id", "score_trend"]
        perf_agg = perf_agg.merge(trends, on="user_id", how="left")

        # Performance change: avg(last 2) - avg(first 2)
        def _perf_change(g):
            g = g.sort_values("date_submitted")
            if len(g) < 3:
                return 0.0
            first2 = g["score"].head(2).mean()
            last2 = g["score"].tail(2).mean()
            return float(last2 - first2)

        changes = ti.groupby("user_id").apply(_perf_change).reset_index()
        changes.columns = ["user_id", "performance_change"]
        perf_agg = perf_agg.merge(changes, on="user_id", how="left")

        si = si.merge(perf_agg, on="user_id", how="left")

        # Deadline per session: completed_at in relative days
        session_deadlines = ts[["user_id", "session_num", "rel_completed"]].copy()
        session_deadlines.columns = ["user_id", "session_id", "deadline"]

        ti_deadline = ti.merge(session_deadlines, on=["user_id", "session_id"], how="left")

        load_agg = ti_deadline.groupby("user_id").agg(
            num_assessments=("template_id", "nunique"),
            avg_assessment_weight=("difficulty", "mean"),
            max_assessment_weight=("difficulty", "max"),
            assessment_density=(
                "deadline",
                lambda d: d.nunique() / max(d.max() - d.min(), 1)
                if d.notna().any() and d.max() > d.min()
                else 0,
            ),
            deadline_density=(
                "deadline",
                lambda d: d.nunique() / max(d.max() - d.min(), 1)
                if d.notna().any() and d.max() > d.min()
                else 0,
            ),
        ).reset_index()

        si = si.merge(load_agg, on="user_id", how="left")

    # Engagement features (from user_events)
    if not ue.empty:
        ue["created_at"] = pd.to_datetime(ue["created_at"])

        # Filter to click-counting events only
        click_types = ["click_option", "input_change", "answer_submit"]
        ue_clicks = ue[ue["event_type"].isin(click_types)].copy()

        if not ue_clicks.empty:
            ue_clicks["rel_day"] = ue_clicks.apply(
                lambda r: (r["created_at"] - reg_dates.get(r["user_id"])).total_seconds() / 86400
                if r["user_id"] in reg_dates else 0,
                axis=1,
            )
            ue_clicks["day"] = ue_clicks["rel_day"].astype(int)

            engagement_agg = ue_clicks.groupby("user_id").agg(
                total_clicks=("counter", "sum"),
                active_days=("day", "nunique"),
            ).reset_index()
            engagement_agg["avg_clicks_per_day"] = (
                engagement_agg["total_clicks"]
                / engagement_agg["active_days"].replace(0, 1)
            )

            # num_resources_visited from task_instances
            if not ti.empty:
                resources = ti.groupby("user_id")["block"].nunique().reset_index()
                resources.columns = ["user_id", "num_resources_visited"]
                engagement_agg = engagement_agg.merge(resources, on="user_id", how="left")
            else:
                engagement_agg["num_resources_visited"] = 0

            # Early / late engagement (first 7 days, last 7 days)
            def _early_late(g):
                days = g["day"]
                min_day, max_day = days.min(), days.max()
                early = g.loc[g["day"] <= min_day + 7, "counter"].sum()
                late = g.loc[g["day"] >= max_day - 7, "counter"].sum()
                return pd.Series({"early_engagement": early, "late_engagement": late})

            el = ue_clicks.groupby("user_id").apply(_early_late).reset_index()
            engagement_agg = engagement_agg.merge(el, on="user_id", how="left")

            # Gap days
            def _gap_features(g):
                days = sorted(g["day"].unique())
                if len(days) < 2:
                    return pd.Series({"avg_gap_days": 0.0, "max_gap_days": 0.0})
                gaps = np.diff(days)
                return pd.Series({"avg_gap_days": float(gaps.mean()), "max_gap_days": float(gaps.max())})

            gaps = ue_clicks.groupby("user_id").apply(_gap_features).reset_index()
            engagement_agg = engagement_agg.merge(gaps, on="user_id", how="left")

            # Activity variance (variance of daily clicks)
            daily = ue_clicks.groupby(["user_id", "day"])["counter"].sum().reset_index()
            act_var = daily.groupby("user_id")["counter"].var().reset_index()
            act_var.columns = ["user_id", "activity_variance"]
            engagement_agg = engagement_agg.merge(act_var, on="user_id", how="left")

            # Inactive days ratio
            def _inactive_ratio(g):
                days = sorted(g["day"].unique())
                if len(days) < 1:
                    return 0.0
                total_range = max(days) - min(days) + 1
                return 1.0 - (len(days) / max(total_range, 1))

            inactive = ue_clicks.groupby("user_id").apply(_inactive_ratio).reset_index()
            inactive.columns = ["user_id", "inactive_days_ratio"]
            engagement_agg = engagement_agg.merge(inactive, on="user_id", how="left")

            # Weekly features
            ue_clicks["week"] = ue_clicks["day"] // 7
            weekly = ue_clicks.groupby(["user_id", "week"])["counter"].sum().reset_index()
            weekly_agg = weekly.groupby("user_id").agg(
                weekly_clicks_mean=("counter", "mean"),
                weekly_clicks_std=("counter", "std"),
            ).reset_index()
            engagement_agg = engagement_agg.merge(weekly_agg, on="user_id", how="left")

            # Activity slope
            def _activity_slope(g):
                daily_g = g.groupby("day")["counter"].sum().reset_index().sort_values("day")
                if len(daily_g) < 2:
                    return 0.0
                x = daily_g["day"].values.astype(float)
                y = daily_g["counter"].values.astype(float)
                slope, _ = np.polyfit(x, y, 1)
                return float(slope)

            slopes = ue_clicks.groupby("user_id").apply(_activity_slope).reset_index()
            slopes.columns = ["user_id", "activity_slope"]
            engagement_agg = engagement_agg.merge(slopes, on="user_id", how="left")

            # Activity drop ratio: 1 - (last week / first week)
            def _drop_ratio(g):
                weekly_g = g.groupby("week")["counter"].sum().reset_index().sort_values("week")
                if len(weekly_g) < 2:
                    return 0.0
                first = weekly_g["counter"].iloc[0]
                last = weekly_g["counter"].iloc[-1]
                if first == 0:
                    return 0.0
                return 1.0 - (last / first)

            drops = ue_clicks.groupby("user_id").apply(_drop_ratio).reset_index()
            drops.columns = ["user_id", "activity_drop_ratio"]
            engagement_agg = engagement_agg.merge(drops, on="user_id", how="left")

            # Engagement change: second half - first half
            def _engagement_change(g):
                g = g.sort_values("day")
                n = len(g)
                if n < 2:
                    return 0.0
                mid = n // 2
                first_half = g["counter"].head(mid).mean() if mid > 0 else 0.0
                second_half = g["counter"].tail(n - mid).mean() if (n - mid) > 0 else 0.0
                return float(second_half - first_half)

            eng_ch = ue_clicks.groupby("user_id").apply(_engagement_change).reset_index()
            eng_ch.columns = ["user_id", "engagement_change"]
            engagement_agg = engagement_agg.merge(eng_ch, on="user_id", how="left")

            si = si.merge(engagement_agg, on="user_id", how="left")

    # Engineered features
    # learning_efficiency
    si["learning_efficiency"] = np.where(
        si.get("total_clicks", pd.Series(0, index=si.index)) > 0,
        si.get("avg_score", 0) / si.get("total_clicks", 1),
        0,
    )
    # resource_diversity
    si["resource_diversity"] = si.get("num_resources_visited", 0)

    # Min-max normalizer (over current dataset)
    def _minmax(series):
        s = series.fillna(0)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    # Barrier scores (same weights as OULAD build_features.py)
    si["low_engagement_score"] = (
        _minmax(-si.get("activity_slope", 0)) * 0.25
        + _minmax(si.get("activity_drop_ratio", 0)) * 0.25
        + (1 - _minmax(si.get("active_days", 0))) * 0.25
        + _minmax(si.get("inactive_days_ratio", 0)) * 0.25
    )

    si["knowledge_gap_score"] = (
        (1 - _minmax(si.get("avg_score", 0))) * 0.30
        + _minmax(si.get("num_low_scores", 0)) * 0.30
        + (1 - _minmax(si.get("score_trend", 0))) * 0.20
        + (1 - _minmax(si.get("assessment_success_rate", 0))) * 0.20
    )

    si["consistency_score"] = (
        _minmax(si.get("avg_gap_days", 0)) * 0.35
        + _minmax(si.get("max_gap_days", 0)) * 0.35
        + _minmax(si.get("activity_variance", 0)) * 0.30
    )

    si["cognitive_overload_score"] = (
        (1 - _minmax(si.get("learning_efficiency", 0))) * 0.30
        + (1 - _minmax(si.get("avg_score", 0))) * 0.20
        + _minmax(si.get("assessment_density", 0)) * 0.25
        + _minmax(si.get("deadline_density", 0)) * 0.25
    )

    # Final column set
    feature_cols = [
        # Demographics
        "gender",
        "highest_education",
        "age_band",
        "disability",
        "num_of_prev_attempts",
        "studied_credits",
        # Temporal
        "days_registered",
        # Assessment (task-level raw)
        "avg_score",
        "min_score",
        "max_score",
        "score_std",
        "score_trend",
        "performance_change",
        "num_assessments_completed",
        "num_low_scores",
        "assessment_success_rate",
        # Course load
        "num_assessments",
        "avg_assessment_weight",
        "max_assessment_weight",
        "assessment_density",
        "deadline_density",
        # Engagement
        "total_clicks",
        "avg_clicks_per_day",
        "active_days",
        "num_resources_visited",
        "weekly_clicks_mean",
        "weekly_clicks_std",
        "activity_slope",
        "activity_drop_ratio",
        "engagement_change",
        "early_engagement",
        "late_engagement",
        "avg_gap_days",
        "max_gap_days",
        "activity_variance",
        "inactive_days_ratio",
        # Engineered
        "learning_efficiency",
        "resource_diversity",
        "low_engagement_score",
        "knowledge_gap_score",
        "consistency_score",
        "cognitive_overload_score",
    ]

    existing = [c for c in feature_cols if c in si.columns]
    # Keep user_id for joining
    result = si[["user_id"] + existing].copy()

    return result


def impute_srs(df: pd.DataFrame, db_config: dict) -> pd.DataFrame:
    """Impute SRS features using OULAD reference values from imputation_params.json."""
    import json
    from pathlib import Path

    from config import CATBOOST_PIPELINE_DIR
    params_path = CATBOOST_PIPELINE_DIR / "imputation_params.json"
    with open(params_path) as f:
        params = json.load(f)

    df = df.copy()
    nan_before = df.isnull().sum().sum()

    # VLE / Engagement
    vle_cols = list(params["vle_fill"].keys())
    vle_existing = [c for c in vle_cols if c in df.columns]
    df["is_no_activity"] = df[vle_existing].isnull().all(axis=1).astype(int)
    for col in vle_existing:
        df[col] = df[col].fillna(params["vle_fill"][col])

    # Consistency
    for col, val in params["consistency_fill"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Performance
    perf_cols = list(params["performance_fill"].keys())
    perf_existing = [c for c in perf_cols if c in df.columns]
    df["is_no_assessment"] = df[perf_existing].isnull().all(axis=1).astype(int)
    for col in perf_existing:
        df[col] = df[col].fillna(params["performance_fill"][col])

    # Temporal
    for col, val in params["temporal_fill"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Barrier + other engineered
    for cat in ["barrier_fill", "other_fill"]:
        for col, val in params.get(cat, {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

    # missing_barrier_flag
    barrier_cols = list(params.get("barrier_fill", {}).keys())
    barrier_existing = [c for c in barrier_cols if c in df.columns]
    df["missing_barrier_flag"] = 0

    nan_after = df.isnull().sum().sum()
    print(f"\nImpute: {nan_before} NaN → {nan_after} NaN filled")

    return df


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(__file__))
    from db import DB_CONFIG

    df = extract_features(DB_CONFIG)
    print(f"Users: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print()
    print(df.to_string())
    # Save for model page
    out = os.path.join(os.path.dirname(__file__), "data", "srs_features.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Impute and save
    imputed = impute_srs(df, DB_CONFIG)
    out2 = os.path.join(os.path.dirname(__file__), "data", "srs_features_imputed_first_ss.csv")
    imputed.to_csv(out2, index=False)
    print(f"Imputed saved to {out2}")
