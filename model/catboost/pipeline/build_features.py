"""
Объединение таблиц OULAD и вычисление признаков для модели.
Результат: merged_features.csv + сводка в консоль.
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE = '/home/srs/data'
OUT = f'{BASE}/merged_features.csv'

# ---- 1. Загрузка ----
print("=" * 60)
print("ЗАГРУЗКА ТАБЛИЦ")
print("=" * 60)

stu = pd.read_csv(f'{BASE}/studentInfo.csv')
reg = pd.read_csv(f'{BASE}/studentRegistration.csv')
# Приводим числовые поля
for c in ['date_registration', 'date_unregistration']:
    reg[c] = pd.to_numeric(reg[c], errors='coerce')
vle = pd.read_csv(f'{BASE}/studentVle.csv')
sas = pd.read_csv(f'{BASE}/studentAssessment.csv')
# Приводим score к числовому
sas['score'] = pd.to_numeric(sas['score'], errors='coerce')
sas['date_submitted'] = pd.to_numeric(sas['date_submitted'], errors='coerce')
asm = pd.read_csv(f'{BASE}/assessments.csv')
asm['date'] = pd.to_numeric(asm['date'], errors='coerce')
asm['weight'] = pd.to_numeric(asm['weight'], errors='coerce')

n_students = stu.shape[0]
print(f"studentInfo:          {stu.shape[0]:,} rows, {stu.shape[1]} cols")
print(f"studentRegistration:  {reg.shape[0]:,} rows, {reg.shape[1]} cols")
print(f"studentVle:           {vle.shape[0]:,} rows, {vle.shape[1]} cols")
print(f"studentAssessment:    {sas.shape[0]:,} rows, {sas.shape[1]} cols")
print(f"assessments:          {asm.shape[0]:,} rows, {asm.shape[1]} cols")
print()

# ---- 2. JOIN 1: studentInfo + studentRegistration (1:1) ----
merge_keys = ['code_module', 'code_presentation', 'id_student']
print("JOIN: studentInfo + studentRegistration (1:1)")
df = stu.merge(reg[merge_keys + ['date_registration', 'date_unregistration']],
               on=merge_keys, how='left')
# Замена '?' на NaN в date_unregistration
df['date_unregistration'] = pd.to_numeric(df['date_unregistration'], errors='coerce')
print(f"  rows: {df.shape[0]:,}")

# ---- 3. studentAssessment + assessments (привязка типа/веса/дедлайна) ----
print("JOIN: studentAssessment + assessments (id_assessment)")
sas_enriched = sas.merge(asm[['id_assessment', 'assessment_type', 'weight']],
                         on='id_assessment', how='left')
# Добавляем deadline из assessments (колонка 'date' в assessments — это дедлайн в днях от старта)
sas_enriched = sas_enriched.merge(asm[['id_assessment', 'date']].rename(columns={'date': 'assessment_deadline'}),
                                  on='id_assessment', how='left')
sas_enriched.rename(columns={'weight': 'assessment_weight'}, inplace=True)
# Приводим к числовым
for c in ['score', 'date_submitted', 'assessment_weight', 'assessment_deadline']:
    sas_enriched[c] = pd.to_numeric(sas_enriched[c], errors='coerce')
print(f"  rows: {sas_enriched.shape[0]:,}")

# ---- 4. Вычисляемые признаки из studentAssessment ----
print("\n--- Признаки: Performance ---")
as_grp = sas_enriched.groupby('id_student')
as_agg = as_grp.agg(
    avg_score=('score', 'mean'),
    min_score=('score', 'min'),
    max_score=('score', 'max'),
    score_std=('score', 'std'),
    num_assessments_completed=('score', 'count'),
    num_low_scores=('score', lambda x: (x < 40).sum()),
    assessment_success_rate=('score', lambda x: (x >= 40).mean()),
    first_submission=('date_submitted', 'min'),
    last_submission=('date_submitted', 'max'),
    # Course load: from assessments
    num_assessments=('id_assessment', 'nunique'),
    avg_assessment_weight=('assessment_weight', 'mean'),
    max_assessment_weight=('assessment_weight', 'max'),
    assessment_density=('assessment_deadline', lambda x: x.nunique() / max(x.max()-x.min(),1) if len(x)>1 else 0),
    deadline_density=('assessment_deadline', lambda x: x.nunique() / max(x.max()-x.min(),1) if len(x)>1 else 0),
).reset_index()

# score_trend: slope of scores over submissions
def compute_score_trend(df_g):
    df_g = df_g.sort_values('date_submitted')
    if len(df_g) < 2:
        return 0
    x = np.arange(len(df_g))
    y = df_g['score'].values.astype(float)
    # linear regression slope
    slope, _ = np.polyfit(x, y, 1)
    return slope

score_trends = as_grp.apply(compute_score_trend).reset_index()
score_trends.columns = ['id_student', 'score_trend']
as_agg = as_agg.merge(score_trends, on='id_student', how='left')

# performance_change: (avg of last 2) - (avg of first 2)
def compute_perf_change(df_g):
    df_g = df_g.sort_values('date_submitted')
    if len(df_g) < 3:
        return 0
    first2 = df_g['score'].head(2).mean()
    last2 = df_g['score'].tail(2).mean()
    return last2 - first2

perf_changes = as_grp.apply(compute_perf_change).reset_index()
perf_changes.columns = ['id_student', 'performance_change']
as_agg = as_agg.merge(perf_changes, on='id_student', how='left')

print(f"  Aggregated: {as_agg.shape[0]:,} students with assessment data")

# ---- 5. Вычисляемые признаки из studentVle ----
print("\n--- Признаки: Engagement + Consistency ---")
vle_grp = vle.groupby('id_student')

vle_agg = vle_grp.agg(
    total_clicks=('sum_click', 'sum'),
    active_days=('date', 'nunique'),
    num_resources_visited=('id_site', 'nunique'),
    avg_clicks_per_day=('sum_click', lambda x: x.sum() / max(x.nunique(), 1)),
).reset_index()

# activity_slope: regression slope of daily clicks over time
def compute_activity_slope(df_g):
    daily = df_g.groupby('date')['sum_click'].sum().reset_index().sort_values('date')
    if len(daily) < 2:
        return 0
    x = daily['date'].values.astype(float)
    y = daily['sum_click'].values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    return slope

vle_slopes = vle.groupby('id_student').apply(compute_activity_slope).reset_index()
vle_slopes.columns = ['id_student', 'activity_slope']
vle_agg = vle_agg.merge(vle_slopes, on='id_student', how='left')

# weekly aggregation for weekly features
vle['week'] = vle['date'] // 7
weekly = vle.groupby(['id_student', 'week'])['sum_click'].sum().reset_index()
weekly_grp = weekly.groupby('id_student')
weekly_agg = weekly_grp.agg(
    weekly_clicks_mean=('sum_click', 'mean'),
    weekly_clicks_std=('sum_click', 'std'),
).reset_index()
vle_agg = vle_agg.merge(weekly_agg, on='id_student', how='left')

# activity_drop_ratio: ratio of clicks in last week vs first week (for students with >=2 weeks)
def drop_ratio(df_g):
    weekly = df_g.groupby('week')['sum_click'].sum().reset_index().sort_values('week')
    if len(weekly) < 2:
        return 0
    first = weekly['sum_click'].iloc[0]
    last = weekly['sum_click'].iloc[-1]
    if first == 0:
        return 0
    return 1.0 - (last / first)  # 0 = stable, positive = drop

drop_ratios = vle.groupby('id_student').apply(drop_ratio).reset_index()
drop_ratios.columns = ['id_student', 'activity_drop_ratio']
vle_agg = vle_agg.merge(drop_ratios, on='id_student', how='left')

# engagement_change: difference in avg daily clicks between second and first half
def engagement_change_fn(df_g):
    df_g = df_g.sort_values('date')
    n = len(df_g)
    if n < 2:
        return 0
    mid = n // 2
    first_half = df_g['sum_click'].head(mid).mean() if mid > 0 else 0
    second_half = df_g['sum_click'].tail(n - mid).mean() if (n - mid) > 0 else 0
    return second_half - first_half

eng_changes = vle.groupby('id_student').apply(engagement_change_fn).reset_index()
eng_changes.columns = ['id_student', 'engagement_change']
vle_agg = vle_agg.merge(eng_changes, on='id_student', how='left')

# early_engagement: total clicks in first 7 days
early = vle[vle['date'] <= 7].groupby('id_student')['sum_click'].sum().reset_index()
early.columns = ['id_student', 'early_engagement']
# late_engagement: total clicks in last 7 days (per student's own last active day)
last_day = vle.groupby('id_student')['date'].max().reset_index()
last_day.columns = ['id_student', 'last_active_day']
late = vle.merge(last_day, on='id_student')
late = late[late['date'] >= late['last_active_day'] - 7]
late_agg = late.groupby('id_student')['sum_click'].sum().reset_index()
late_agg.columns = ['id_student', 'late_engagement']
vle_agg = vle_agg.merge(early, on='id_student', how='left')
vle_agg = vle_agg.merge(late_agg, on='id_student', how='left')

# Gap days between visits
def gap_features(df_g):
    days = sorted(df_g['date'].unique())
    if len(days) < 2:
        return pd.Series({'avg_gap_days': 0, 'max_gap_days': 0})
    gaps = np.diff(days)
    return pd.Series({'avg_gap_days': gaps.mean(), 'max_gap_days': gaps.max()})

gap_feats = vle.groupby('id_student').apply(gap_features).reset_index()
vle_agg = vle_agg.merge(gap_feats, on='id_student', how='left')

# activity_variance: variance of daily clicks
daily_var = vle.groupby(['id_student', 'date'])['sum_click'].sum().reset_index()
daily_var = daily_var.groupby('id_student')['sum_click'].var().reset_index()
daily_var.columns = ['id_student', 'activity_variance']
vle_agg = vle_agg.merge(daily_var, on='id_student', how='left')

# inactive_days_ratio
def inactive_ratio(df_g):
    days = sorted(df_g['date'].unique())
    if len(days) < 2:
        return 0 if len(days) == 0 else 0
    total_range = max(days) - min(days) + 1
    active = len(days)
    return 1.0 - (active / max(total_range, 1))

inactive_ratios = vle.groupby('id_student').apply(inactive_ratio).reset_index()
inactive_ratios.columns = ['id_student', 'inactive_days_ratio']
vle_agg = vle_agg.merge(inactive_ratios, on='id_student', how='left')

print(f"  Aggregated: {vle_agg.shape[0]:,} students with VLE data")

# ---- 6. JOIN student-level aggregations to main df ----
print("\n--- JOIN: student-level features to main ---")
df = df.merge(as_agg, on='id_student', how='left')
df = df.merge(vle_agg, on='id_student', how='left')
print(f"  rows: {df.shape[0]:,}, cols: {df.shape[1]}")

# ---- 7. Temporal features ----
print("\n--- Признаки: Temporal ---")

# days_registered: from date_registration (negative = before start)
df['days_registered'] = (df['date_unregistration'].fillna(df['last_submission'].fillna(0)) 
                         - df['date_registration'])

# days_until_dropout: for withdrawn students, days from registration to unregistration
df['days_until_dropout'] = np.where(
    df['final_result'] == 'Withdrawn',
    df['date_unregistration'] - df['date_registration'],
    np.nan
)

# dropout_week: week of dropout (for withdrawn students)
df['dropout_week'] = np.where(
    df['date_unregistration'].notna(),
    df['date_unregistration'] // 7,
    np.nan
)

# course_completion_ratio: fraction of course days the student stayed
# Course length from courses.csv could be used, but we approximate:
# For completed students: their last interaction date / typical course length
# Using max(date) per module-presentation from VLE as course length
course_len = vle.groupby(['code_module', 'code_presentation'])['date'].max().reset_index()
course_len.columns = ['code_module', 'code_presentation', 'course_end_day']
df = df.merge(course_len, on=['code_module', 'code_presentation'], how='left')
df['course_completion_ratio'] = 1.0
mask_wd = df['final_result'] == 'Withdrawn'
df.loc[mask_wd, 'course_completion_ratio'] = (
    df.loc[mask_wd, 'date_unregistration'] / df.loc[mask_wd, 'course_end_day']
).clip(0, 1)

# ---- 8. Learning Behavior features ----
print("\n--- Признаки: Learning Behavior ---")

# learning_efficiency: avg_score / total_clicks
df['learning_efficiency'] = np.where(
    df['total_clicks'] > 0,
    df['avg_score'] / df['total_clicks'],
    0
)

# resource_diversity: unique(id_site) — уже есть как num_resources_visited
df['resource_diversity'] = df['num_resources_visited']

# ---- 9. Barrier-oriented scores ----
print("\n--- Признаки: Barrier-Oriented ---")

# 9.1 low_engagement_score (из activity_slope, activity_drop_ratio, active_days, inactive_days_ratio)
# Нормализуем и комбинируем
def minmax(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0, index=series.index)
    return (series - mn) / (mx - mn)

df['low_engagement_score'] = (
    minmax(-df['activity_slope'].fillna(0)) * 0.25 +      # negative slope = bad
    minmax(df['activity_drop_ratio'].fillna(0)) * 0.25 +  # high drop = bad
    (1 - minmax(df['active_days'].fillna(0))) * 0.25 +    # low active days = bad
    minmax(df['inactive_days_ratio'].fillna(0)) * 0.25    # high inactive = bad
)

# 9.2 knowledge_gap_score (из avg_score, num_low_scores, score_trend, assessment_success_rate)
df['knowledge_gap_score'] = (
    (1 - minmax(df['avg_score'].fillna(0))) * 0.30 +
    minmax(df['num_low_scores'].fillna(0)) * 0.30 +
    (1 - minmax(df['score_trend'].fillna(0))) * 0.20 +    # low/negative trend = gap
    (1 - minmax(df['assessment_success_rate'].fillna(0))) * 0.20
)

# 9.3 consistency_score (из avg_gap_days, max_gap_days, activity_variance)
df['consistency_score'] = (
    minmax(df['avg_gap_days'].fillna(0)) * 0.35 +
    minmax(df['max_gap_days'].fillna(0)) * 0.35 +
    minmax(df['activity_variance'].fillna(0)) * 0.30
)

# 9.4 cognitive_overload_score (из learning_efficiency, avg_score, assessment_density, deadline_density)
df['cognitive_overload_score'] = (
    (1 - minmax(df['learning_efficiency'].fillna(0))) * 0.30 +
    (1 - minmax(df['avg_score'].fillna(0))) * 0.20 +
    minmax(df['assessment_density'].fillna(0)) * 0.25 +
    minmax(df['deadline_density'].fillna(0)) * 0.25
)

# ---- 10. Финальный набор признаков + target ----
feature_cols = [
    # Raw features
    'gender', 'highest_education', 'age_band', 'num_of_prev_attempts',
    'studied_credits', 'disability',
    # Engagement
    'total_clicks', 'avg_clicks_per_day', 'active_days', 'num_resources_visited',
    'weekly_clicks_mean', 'weekly_clicks_std', 'activity_slope', 'activity_drop_ratio',
    'engagement_change', 'early_engagement', 'late_engagement',
    # Consistency
    'avg_gap_days', 'max_gap_days', 'activity_variance', 'inactive_days_ratio',
    # Performance
    'avg_score', 'min_score', 'max_score', 'score_std', 'score_trend',
    'performance_change', 'num_assessments_completed', 'num_low_scores',
    'assessment_success_rate',
    # Temporal
    'days_registered', 'days_until_dropout', 'dropout_week', 'course_completion_ratio',
    # Course Load
    'num_assessments', 'avg_assessment_weight', 'max_assessment_weight',
    'assessment_density', 'deadline_density',
    # Learning Behavior
    'learning_efficiency', 'resource_diversity',
    # Barrier-Oriented
    'low_engagement_score', 'knowledge_gap_score', 'consistency_score',
    'cognitive_overload_score',
]

id_cols = ['code_module', 'code_presentation', 'id_student']

# Бинарный таргет: 1 = Withdrawn (дропаут), 0 = Pass/Fail/Distinction (завершил курс)
df['target'] = (df['final_result'] == 'Withdrawn').astype(int)

target_col = 'target'

all_cols = id_cols + feature_cols + ['final_result', target_col]
# Только существующие колонки
existing = [c for c in all_cols if c in df.columns]
df_out = df[existing].copy()

# ---- 11. Сохранение ----
df_out.to_csv(OUT, index=False)
print(f"\n{'='*60}")
print(f"ИТОГОВЫЙ ДАТАСЕТ сохранён: {OUT}")
print(f"Строк: {df_out.shape[0]:,} | Признаков: {len(feature_cols)} | Всего колонок: {df_out.shape[1]}")
print(f"Размер файла: {os.path.getsize(OUT)/1024/1024:.1f} MB")

# ---- Train/Test split (80/20, stratified) ----
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_out, test_size=0.2, stratify=df_out['target'], random_state=42)
train.to_csv(f'{BASE}/train.csv', index=False)
test.to_csv(f'{BASE}/test.csv', index=False)

print(f"\nTRAIN/TEST SPLIT (80/20, stratified):")
print(f"  Train: {train.shape[0]:,} ({train.shape[0]/df_out.shape[0]*100:.0f}%)")
print(f"  Test:  {test.shape[0]:,} ({test.shape[0]/df_out.shape[0]*100:.0f}%)")
print(f"  Train target 0: {train['target'].value_counts().get(0,0):,}  |  1: {train['target'].value_counts().get(1,0):,}")
print(f"  Test  target 0: {test['target'].value_counts().get(0,0):,}  |  1: {test['target'].value_counts().get(1,0):,}")

# ---- 12. Сводка ----
print(f"\n{'='*60}")
print("СВОДКА")
print("=" * 60)

print(f"\nСтудентов: {df_out.shape[0]:,}")
print(f"Признаков (включая target): {df_out.shape[1] - len(id_cols)}")

print(f"\nРаспределение final_result:")
print(df_out['final_result'].value_counts().to_string())

print(f"\nБинарный таргет (target):")
tc = df_out['target'].value_counts()
print(f"  0 (завершил): {tc[0]:,} ({tc[0]/len(df_out)*100:.1f}%)")
print(f"  1 (дропаут):  {tc[1]:,} ({tc[1]/len(df_out)*100:.1f}%)")
print(f"\nПропуски в признаках (%):")
missing = df_out[feature_cols].isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    for col, pct in missing.items():
        bar = '█' * int(pct / 5)
        print(f"  {col:35s} {pct:5.1f}% {bar}")
else:
    print("  Нет пропусков")

print(f"\n{'='*60}")
print("БАЗОВАЯ СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ")
print("=" * 60)
num_cols = df_out[feature_cols].select_dtypes(include=[np.number]).columns
stats = df_out[num_cols].describe().T[['mean', 'std', 'min', 'max']].round(2)
print(stats.to_string())
