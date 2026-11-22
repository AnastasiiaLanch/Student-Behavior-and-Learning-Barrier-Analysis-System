"""
Заполнение пропусков в train и test.
Все статистики вычисляются ТОЛЬКО на train, затем применяются к test.
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE = '/home/srs/data'
train = pd.read_csv(f'{BASE}/train.csv')
test = pd.read_csv(f'{BASE}/test.csv')

print(f"Train: {train.shape[0]:,} x {train.shape[1]}")
print(f"Test:  {test.shape[0]:,} x {test.shape[1]}")

# Считаем пропуски до
def missing_report(df, label):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n{label}: {len(missing)} колонок с пропусками, всего ячеек NaN: {df.isnull().sum().sum():,}")
    for c, n in missing.items():
        print(f"  {c:35s} {n:>6,} ({n/len(df)*100:.1f}%)")

missing_report(train, "Train до")
missing_report(test, "Test до")

# ================================================================
# ВСЕ СТАТИСТИКИ ВЫЧИСЛЯЮТСЯ ТОЛЬКО НА TRAIN
# ================================================================

# ---- 1. Демография ----
print("\n--- 1. Демография ---")

cat_demo = ['gender', 'highest_education', 'age_band', 'disability']
for col in cat_demo:
    train[col] = train[col].fillna('Unknown')
    test[col] = test[col].fillna('Unknown')
    # Новые категории в test → Unknown — и так уже обработано fillna

# Числовые демо: медиана из train
num_demo = ['num_of_prev_attempts', 'studied_credits']
for col in num_demo:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)
    print(f"  {col}: median={med:.2f}")

# ---- 2. VLE-признаки ----
print("\n--- 2. VLE / Engagement ---")

vle_cols = [
    'total_clicks', 'avg_clicks_per_day', 'active_days', 'num_resources_visited',
    'weekly_clicks_mean', 'weekly_clicks_std', 'activity_slope', 'activity_drop_ratio',
    'engagement_change', 'early_engagement', 'late_engagement',
]

# Флаг: не было активности
def add_no_activity_flag(df):
    no_act = df[vle_cols].isnull().all(axis=1).astype(int)
    # Также считаем: если все vle-признаки == 0 (после заполнения) — тоже не было активности
    # Но пока ставим только для тех, у кого были пропуски
    return no_act

train['is_no_activity'] = add_no_activity_flag(train)
test['is_no_activity'] = add_no_activity_flag(test)

for col in vle_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

print(f"  is_no_activity: train={train['is_no_activity'].sum()}, test={test['is_no_activity'].sum()}")

# ---- 3. Consistency ----
print("\n--- 3. Consistency ---")

# Используем max_gap_days из train как значение по умолчанию
max_gap_train = train['max_gap_days'].max()
print(f"  max_gap_days (train): {max_gap_train:.0f}")

consistency_cols = ['avg_gap_days', 'max_gap_days', 'activity_variance', 'inactive_days_ratio']
for col in consistency_cols:
    train[col] = train[col].fillna(max_gap_train if col in ('avg_gap_days', 'max_gap_days') else 0)
    test[col] = test[col].fillna(max_gap_train if col in ('avg_gap_days', 'max_gap_days') else 0)

# ---- 4. Оценки (Performance) ----
print("\n--- 4. Performance ---")

perf_cols = [
    'avg_score', 'min_score', 'max_score', 'score_std', 'score_trend',
    'performance_change', 'num_assessments_completed', 'num_low_scores',
    'assessment_success_rate',
]

def add_no_assessment_flag(df):
    return df[perf_cols].isnull().all(axis=1).astype(int)

train['is_no_assessment'] = add_no_assessment_flag(train)
test['is_no_assessment'] = add_no_assessment_flag(test)

for col in perf_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

print(f"  is_no_assessment: train={train['is_no_assessment'].sum()}, test={test['is_no_assessment'].sum()}")

# ---- 5. Временные ----
print("\n--- 5. Temporal ---")

temporal_cols = ['days_registered', 'days_until_dropout', 'dropout_week', 'course_completion_ratio']
for col in temporal_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# Особое правило: если студент не дропнулся (target=0) и days_until_dropout=0, ставим -1
train.loc[(train['target'] == 0) & (train['days_until_dropout'] == 0), 'days_until_dropout'] = -1
test.loc[(test['target'] == 0) & (test['days_until_dropout'] == 0), 'days_until_dropout'] = -1

print(f"  days_until_dropout=-1: train={(train['days_until_dropout']==-1).sum()}, test={(test['days_until_dropout']==-1).sum()}")

# ---- 6. Нагрузка курса — медиана по train, по каждому модулю-презентации ----
print("\n--- 6. Course Load ---")

course_cols = ['num_assessments', 'avg_assessment_weight', 'max_assessment_weight',
               'assessment_density', 'deadline_density']

# Медиана по train в разрезе модуль+презентация
for col in course_cols:
    med_by_course = train.groupby(['code_module', 'code_presentation'])[col].median()
    print(f"  {col}: глобальная медиана = {train[col].median():.3f}")

    # Применяем к train
    for (mod, pres), med_val in med_by_course.items():
        mask_train = (train['code_module'] == mod) & (train['code_presentation'] == pres) & (train[col].isnull())
        train.loc[mask_train, col] = med_val
        mask_test = (test['code_module'] == mod) & (test['code_presentation'] == pres) & (test[col].isnull())
        test.loc[mask_test, col] = med_val

    # Если остались пропуски (новая комбинация в test) — глобальная медиана из train
    global_med = train[col].median()
    train[col] = train[col].fillna(global_med)
    test[col] = test[col].fillna(global_med)

# ---- 7. Вычисленные признаки ----
print("\n--- 7. Computed (learning_efficiency, resource_diversity) ---")

# Эти признаки уже были заполнены в build_features (0 для пропусков), но проверим
train['learning_efficiency'] = train['learning_efficiency'].fillna(0)
test['learning_efficiency'] = test['learning_efficiency'].fillna(0)
train['resource_diversity'] = train['resource_diversity'].fillna(0)
test['resource_diversity'] = test['resource_diversity'].fillna(0)

# ---- 8. Barrier scores ----
print("\n--- 8. Barrier Scores ---")

barrier_cols = ['low_engagement_score', 'knowledge_gap_score', 'consistency_score',
                'cognitive_overload_score']

# Флаг пропуска для барьеров
train['missing_barrier_flag'] = train[barrier_cols].isnull().any(axis=1).astype(int)
test['missing_barrier_flag'] = test[barrier_cols].isnull().any(axis=1).astype(int)

for col in barrier_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

print(f"  missing_barrier_flag: train={train['missing_barrier_flag'].sum()}, test={test['missing_barrier_flag'].sum()}")

# ---- Финальная проверка ----
print(f"\n{'='*60}")
print("ПОСЛЕ ЗАПОЛНЕНИЯ")
print(f"{'='*60}")
missing_report(train, "Train после")
missing_report(test, "Test после")

# Проверка, что нигде не осталось NaN
train_nan = train.isnull().sum().sum()
test_nan = test.isnull().sum().sum()
print(f"\nОсталось NaN: train={train_nan}, test={test_nan}")
assert train_nan == 0, f"Train has {train_nan} NaN!"
assert test_nan == 0, f"Test has {test_nan} NaN!"

# ---- Сохранение ----
train.to_csv(f'{BASE}/train_imputed.csv', index=False)
test.to_csv(f'{BASE}/test_imputed.csv', index=False)

print(f"\nСохранено: train_imputed.csv ({train.shape[0]:,} x {train.shape[1]})")
print(f"Сохранено: test_imputed.csv ({test.shape[0]:,} x {test.shape[1]})")

# Сводка новых колонок
new_cols = [c for c in train.columns if c not in pd.read_csv(f'{BASE}/train.csv').columns]
print(f"\nНовые признаки: {new_cols}")
print(f"Итого признаков: {len(train.columns) - 4}")  # минус 3 id + target + final_result
