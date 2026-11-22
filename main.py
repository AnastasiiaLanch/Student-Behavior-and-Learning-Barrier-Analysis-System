import os
import json
import logging
from datetime import datetime
from pathlib import Path

import jinja2
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

import config
from auth import authenticate, create_session, get_session_user, delete_session
import pymysql

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("srs")

app = FastAPI(title="SRS", docs_url=None, redoc_url=None)

_templates_dir = Path(__file__).parent / "templates"
_jinja_loader = jinja2.FileSystemLoader(str(_templates_dir))
_jinja_env = jinja2.Environment(loader=_jinja_loader, autoescape=True)


def render(template_name: str, **context) -> str:
    return _jinja_env.get_template(template_name).render(**context)


_DB_CFG = dict(
    host=os.environ.get('SRS_DB_HOST', 'localhost'),
    port=3306,
    user=os.environ.get('SRS_DB_USER', ''),
    database=os.environ.get('SRS_DB_NAME', 'SRS'),
    password=os.environ.get('SRS_DB_PASS', ''),
    cursorclass=pymysql.cursors.DictCursor
)

from task_generator import TaskGenerator
task_gen = TaskGenerator(_DB_CFG)


def _has_student_info(user_id: int) -> bool:
    """Check if user has filled out demographic info."""
    conn = pymysql.connect(**_DB_CFG)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM student_info WHERE user_id=%s", (user_id,))
        return cur.fetchone() is not None
    finally:
        conn.close()


SRS_COOKIE = ''

PUBLIC_PATHS = ''

NO_STUDENT_INFO_PATHS = ''


async def check_srs_auth(request: Request):
    path = request.url.path.rstrip("/")
    if path == "":
        path = "/"


    for p in PUBLIC_PATHS:
        if path == p or path.startswith(p + "/"):
            return None

    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token)
    if user_id is None:
        return RedirectResponse(url="/srs/enter", status_code=302)

    if path.startswith("/srs/test"):
        if not _has_student_info(user_id):
            return RedirectResponse(url="/srs/register", status_code=302)

    return None


class SrsAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        result = await check_srs_auth(request)
        if result is not None:
            return result
        return await call_next(request)


app.add_middleware(SrsAuthMiddleware)

app.mount("/srs/static", StaticFiles(directory=str(_templates_dir)), name="srs_static")


def compute_outliers(df):
    """Вычисляет выбросы по IQR (1.5×IQR) для всех числовых признаков."""
    skip = {
        'code_module', 'code_presentation', 'id_student', 'target',
        'gender', 'highest_education', 'age_band', 'disability',
        'is_no_activity', 'is_no_assessment', 'missing_barrier_flag',
        'final_result',  # на случай если остался
    }
    numeric = [c for c in df.columns if c not in skip and df[c].dtype in ('int64', 'float64')]
    results = []
    for col in numeric:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            results.append({
                'feature': col, 'q1': round(q1, 2), 'q3': round(q3, 2), 'iqr': 0,
                'lower': None, 'upper': None,
                'low_count': 0, 'high_count': 0, 'total_outliers': 0, 'pct': 0,
            })
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        low = int((df[col] < lower).sum())
        high = int((df[col] > upper).sum())
        total = low + high
        pct = round(total / len(df) * 100, 2)
        results.append({
            'feature': col, 'q1': round(q1, 2), 'q3': round(q3, 2), 'iqr': round(iqr, 2),
            'lower': round(lower, 2), 'upper': round(upper, 2),
            'low_count': low, 'high_count': high, 'total_outliers': total, 'pct': pct,
        })
    results.sort(key=lambda x: x['pct'], reverse=True)
    return results


@app.get("/srs/model", response_class=HTMLResponse)
async def srs_model(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token)
    if not user_id:
        return RedirectResponse(url="/srs/enter", status_code=302)
    conn = pymysql.connect(**_DB_CFG)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT username, is_admin FROM auth_users WHERE id=%s AND section='srs'",
            (user_id,),
        )
        row = cur.fetchone()
        if not row or not (row.get('is_admin') or row['username'] == 'Admin'):
            return RedirectResponse(url="/srs/test", status_code=302)
    finally:
        conn.close()
    return await _render_model(request)


def _project_tree(root: str = None, max_depth: int = 5) -> str:
    import os

    root = root or str(config.BASE_DIR)
    SKIP = set(config.TREE_SKIP)

    NOTES = {
        # каталоги
        "srs/": "корень проекта",
        "data/": "датасеты и скрипты подготовки данных",
        "predictions/": "предсказания по сессиям и их датасеты",
        "model/": "обученные модели",
        "model/catboost/": "модель CatBoost (градиентный бустинг)",
        "model/xgb/": "модель XGBoost",
        "templates/": "HTML-шаблоны страниц",
        # код
        "main.py": "FastAPI-приложение: маршруты, логика теста, страница модели",
        "auth.py": "авторизация: хеши паролей, сессии, проверка cookie",
        "db.py": "подключение к MySQL (параметры из .env)",
        "task_generator.py": "генерация заданий по шаблонам: параметры, ответ, дистракторы",
        "setup_tables.py": "создание схемы БД (DDL)",
        "create_tables.py": "утилита создания таблиц",
        "extract_features.py": "сборка признаков из данных теста",
        "save_pipeline.py": "сохранение пайплайна подготовки признаков",
        "train_catboost.py": "обучение модели CatBoost",
        "train_xgb.py": "обучение модели XGBoost",
        "production_predict.py": "предсказания на актуальных данных",
        "demo_generator.py": "демо-данные для проверки",
        "install.sh": "установка зависимостей и окружения",
        "apply_nginx.sh": "прописывание конфига nginx для /srs",
        "srs-api.service": "systemd-юнит запуска приложения (порт 8003)",
        ".env": "конфиг: параметры БД и админ-пароль",
        "config.py": "пути и параметры из переменных окружения",
        "requirements.txt": "зависимости Python с фиксированными версиями",
        ".gitignore": "что не попадает в git",
        "README.md": "описание проекта (рус.)",
        "README.md": "описание проекта (англ.)",
        # данные
        "merged_features.csv": "объединённый набор признаков (основной датасет)",
        "train.csv": "обучающая выборка (сырая)",
        "test.csv": "тестовая выборка (сырая)",
        "train_imputed.csv": "обучающая выборка после заполнения пропусков",
        "test_imputed.csv": "тестовая выборка после заполнения пропусков",
        "srs_features.csv": "признаки, снятые с теста SRS",
        "OULAD.names": "описание полей датасета OULAD",
        "oulad.zip": "исходный архив датасета OULAD (UCI)",
        "studentInfo.csv": "OULAD: профили студентов",
        "studentAssessment.csv": "OULAD: результаты оцениваний",
        "studentRegistration.csv": "OULAD: регистрация на курсы",
        "assessments.csv": "OULAD: список оцениваний по курсам",
        "courses.csv": "OULAD: справочник курсов",
        "vle.csv": "OULAD: материалы виртуальной среды обучения",
        "build_features.py": "скрипт сборки признаков из OULAD",
        "impute.py": "заполнение пропусков в признаках",
        # шаблоны
        "enter.html": "страница входа",
        "register.html": "форма профиля участника",
        "test.html": "страница прохождения теста",
        "model.html": "страница модели и результатов",
        "index.html": "стартовая страница раздела",
        "schema_diagram.svg": "схема связей в виде картинки",
        "pipeline/": "пайплайн подготовки признаков",
        "feature_spec.json": "описание признаков модели",
        "imputation_params.json": "эталонные значения для заполнения пропусков",
        "barrier_stats.json": "пороги p50 и p75 для шкал-барьеров",
        "barrier_feature_refs.json": "привязка шкал-барьеров к признакам",
        "model.cbm": "веса модели CatBoost",
        "model.json": "веса модели XGBoost",
        "metrics.json": "метрики качества модели",
        "confusion.json": "матрица ошибок",
        "feature_importance.json": "важность признаков",
        "params.json": "гиперпараметры модели",
    }

    def note_for(rel: str, is_dir: bool) -> str:
        key = rel + "/" if is_dir else rel
        if key in NOTES:
            return NOTES[key]
        base = rel.rsplit("/", 1)[-1]
        if is_dir:
            base += "/"
        if base in NOTES:
            return NOTES[base]
        PREFIX = (
            ("prod_dataset_", "данные для production-предсказаний (по сессиям)"),
            ("prod_predictions_", "результаты production-предсказаний"),
            ("srs_features_imputed_", "признаки SRS после импутации (по сессиям)"),
            ("srs_features_", "признаки, снятые с теста SRS"),
        )
        for pfx, text in PREFIX:
            if base.startswith(pfx):
                return text
        return ""

    def walk(path: str, prefix: str, depth: int, lines: list, rel_prefix: str = ""):
        try:
            entries = sorted(
                [e for e in os.listdir(path) if e not in SKIP and not e.endswith(".pyc") and ".bak" not in e]
            )
        except OSError:
            return
        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
        files = [e for e in entries if not os.path.isdir(os.path.join(path, e))]
        ordered = dirs + files
        for i, name in enumerate(ordered):
            last = i == len(ordered) - 1
            full = os.path.join(path, name)
            is_dir = os.path.isdir(full)
            rel = (rel_prefix + name).lstrip("/")
            marker = "/" if is_dir else ""
            size = ""
            if not is_dir:
                try:
                    b = os.path.getsize(full)
                    if b >= 1048576:
                        size = "  (%.1f MB)" % (b / 1048576)
                    else:
                        size = "  (%d KB)" % max(1, b // 1024)
                except OSError:
                    size = ""
            note = note_for(rel, is_dir)
            tail = ("  # " + note) if note else ""
            lines.append(prefix + ("\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 ") + name + marker + size + tail)
            if is_dir and depth < max_depth:
                walk(full, prefix + ("    " if last else "\u2502   "), depth + 1, lines, rel + "/")

    lines = ["srs/  # " + NOTES["srs/"]]
    walk(root, "", 1, lines)
    return "\n".join(lines)


async def _render_model(request: Request):
    import json

    rsrc = config.DATA_DIR
    train_outliers = []
    test_outliers = []
    for fname, target_list in [('train_imputed.csv', train_outliers), ('test_imputed.csv', test_outliers)]:
        fp = rsrc / fname
        if fp.exists():
            df = pd.read_csv(fp)
            target_list.extend(compute_outliers(df))

    # Сканируем директорию моделей
    models_dir = config.MODEL_DIR
    models = []
    barrier_stats = None
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            mid = model_dir.name
            if mid.lower() == "xgb":
                continue
            m = {
                'id': mid,
                'name': mid.replace('_', ' ').title(),
                'metrics': None,
                'params': None,
                'confusion': None,
                'feature_importance': None,
                'barrier_stats': None,
            }
            for fname, key in [('metrics.json', 'metrics'), ('params.json', 'params'),
                               ('confusion.json', 'confusion'),
                               ('feature_importance.json', 'feature_importance')]:
                fp = model_dir / fname
                if fp.exists():
                    with open(fp) as f:
                        m[key] = json.load(f)
            bs_fp = model_dir / 'pipeline' / 'barrier_stats.json'
            if bs_fp.exists():
                with open(bs_fp) as f:
                    m['barrier_stats'] = json.load(f)
                    if barrier_stats is None:
                        barrier_stats = m['barrier_stats']
            models.append(m)

    srs_csv = config.DATA_DIR / 'srs_features.csv'
    srs_features = None
    srs_columns = []
    srs_rows = 0
    if srs_csv.exists():
        df = pd.read_csv(srs_csv)
        srs_columns = list(df.columns)
        srs_rows = len(df)
        srs_features = df.head(5).to_dict(orient='records')
        if 'user_id' in srs_columns:
            srs_columns.remove('user_id')
            for row in srs_features:
                row.pop('user_id', None)

    # Imputed snapshots (first_ss, second_ss, ...)
    srs_snapshots = []
    data_dir = config.DATA_DIR
    for ss_file in sorted(data_dir.glob('srs_features_imputed_*_ss.csv'),
                          key=lambda p: ['first','second','third','fourth','fifth','sixth'].index(
                              p.stem.replace('srs_features_imputed_','').replace('_ss',''))):
        name = ss_file.stem.replace('srs_features_imputed_', '')
        df2 = pd.read_csv(ss_file)
        cols = [c for c in df2.columns if c != 'user_id']
        rows = df2.head(5).to_dict(orient='records')
        for row in rows:
            row.pop('user_id', None)
        user_ids = list(df2['user_id'].values) if 'user_id' in df2.columns else []
        snapshot = {
            'name': name,
            'columns': cols,
            'rows_data': rows,
            'num_rows': len(df2),
            'num_cols': len(cols),
        }
        # Inference
        try:
            from catboost import CatBoostClassifier, Pool
            model_path = config.CATBOOST_DIR / 'model.cbm'
            fi_path = config.CATBOOST_DIR / 'feature_importance.json'
            if model_path.exists() and fi_path.exists():
                with open(fi_path) as f:
                    fi = json.load(f)
                model_cols = [item['feature'] for item in fi]
                cat_features = ['gender', 'highest_education', 'age_band', 'disability']
                cat_indices = [model_cols.index(c) for c in cat_features]
                X = df2[model_cols].copy()
                pool = Pool(X, cat_features=cat_indices)
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                proba = model.predict_proba(pool)[:, 1]
                pred = model.predict(pool)
                snapshot['prediction'] = {
                    'probabilities': [round(float(p), 4) for p in proba],
                    'predictions': [int(p) for p in pred],
                    'num_sessions': int(df2['num_assessments_completed'].iloc[0]) // 9 if 'num_assessments_completed' in df2.columns else 1,
                }

        except Exception as e:
            logger.warning(f"Inference failed for {name}: {e}")
        # Barriers
        bs_path = config.CATBOOST_PIPELINE_DIR / 'barrier_stats.json'
        if bs_path.exists():
            with open(bs_path) as f:
                barrier_stats_data = json.load(f)
            barrier_names = ['low_engagement_score', 'knowledge_gap_score', 'consistency_score', 'cognitive_overload_score']
            barrier_labels = {
                'low_engagement_score': 'Low Engagement',
                'knowledge_gap_score': 'Knowledge Gap',
                'consistency_score': 'Consistency',
                'cognitive_overload_score': 'Cognitive Overload',
            }
            barriers = []
            for bname in barrier_names:
                if bname in df2.columns:
                    val = float(df2[bname].iloc[0])
                    p50 = barrier_stats_data[bname]['dropout_p50']
                    p75 = barrier_stats_data[bname]['dropout_p75']
                    if val < p50:
                        level, icon, color = 'none', '✓', '#238636'
                    elif val < p75:
                        level, icon, color = 'possible', '⚠', '#d29922'
                    else:
                        level, icon, color = 'formed', '🔴', '#f85149'
                    barriers.append({
                        'label': barrier_labels.get(bname, bname),
                        'name': bname,
                        'value': round(val, 4),
                        'p50': round(p50, 4),
                        'p75': round(p75, 4),
                        'level': level,
                        'icon': icon,
                        'color': color,
                    })
            snapshot['barriers'] = barriers
        srs_snapshots.append(snapshot)

    # Production predictions (prod_predictions_*.json → prod_dataset_*.csv)
    prod_predictions = []
    data_dir2 = config.PREDICTIONS_DIR
    if not data_dir2.exists():
        data_dir2 = config.DATA_DIR
    ORDER = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6}
    for pred_file in sorted(data_dir2.glob('prod_predictions_*.json')):
        ds_name = pred_file.stem.replace('prod_predictions_', '')
        with open(pred_file) as f:
            pred_data = json.load(f)
        # Attach dataset row count for label
        ds_csv = data_dir2 / f'prod_dataset_{ds_name}.csv'
        ds_rows = 0
        if ds_csv.exists():
            ds_rows = len(pd.read_csv(ds_csv))
        prod_predictions.append({
            'name': ds_name,
            'order': ORDER.get(ds_name.split('_')[0], 99),
            'rows': ds_rows,
            'predictions': pred_data,
        })
    prod_predictions.sort(key=lambda x: x['order'])

    return HTMLResponse(render("model.html", title="Обучение модели — SRS",
                               train_outliers=train_outliers, test_outliers=test_outliers,
                               models=models, barrier_stats=barrier_stats,
                               srs_features=srs_features, srs_columns=srs_columns, srs_rows=srs_rows,
                               srs_snapshots=srs_snapshots,
                               prod_predictions=prod_predictions,
                               project_tree=_project_tree()))


# Session-system: 6 sessions × 9 tasks
_SESSION_TEMPLATES = {
    1: [1,2,3, 7,8,9, 13,14,15],   # odd sessions → these templates
    0: [4,5,6, 10,11,12, 16,17,18], # even sessions → these templates
}

def _session_templates(session_num: int) -> list:
    """Return template IDs for a given session number (1-6)."""
    return _SESSION_TEMPLATES[session_num % 2]

def _session_difficulty(session_num: int) -> int:
    """Difficulty level for a given session number."""
    return (session_num + 1) // 2  # 1→1, 2→1, 3→2, 4→2, 5→3, 6→3


def _is_admin(cur, user_id: int) -> bool:
    cur.execute(
        "SELECT 1 FROM auth_users WHERE id=%s AND section='srs' AND username='Admin'",
        (user_id,),
    )
    return cur.fetchone() is not None


def _get_or_create_session(cur, user_id: int, retry_session: int = None) -> dict:
    """Get the user's current open session, or create the next one.
    If retry_session is set, create a new attempt of that session."""
    cur.execute(
        "SELECT * FROM test_sessions WHERE user_id=%s AND completed_at IS NULL ORDER BY session_num, attempt_num LIMIT 1",
        (user_id,)
    )
    sess = cur.fetchone()
    if sess:
        return sess

    if retry_session:
        # Delete old task instances and events for this session before retrying
        cur.execute(
            "DELETE FROM task_instances WHERE user_id=%s AND session_id=%s",
            (user_id, retry_session)
        )
        cur.execute(
            "DELETE FROM user_events WHERE user_id=%s AND session_id=%s",
            (user_id, retry_session)
        )
        # Create new attempt of the same session
        cur.execute(
            "SELECT COALESCE(MAX(attempt_num),0)+1 AS na FROM test_sessions WHERE user_id=%s AND session_num=%s",
            (user_id, retry_session)
        )
        attempt = cur.fetchone()['na']
        diff = _session_difficulty(retry_session)
        cur.execute(
            "INSERT INTO test_sessions (user_id, session_num, attempt_num, difficulty) VALUES (%s,%s,%s,%s)",
            (user_id, retry_session, attempt, diff)
        )
        conn = cur.connection
        conn.commit()
        cur.execute("SELECT * FROM test_sessions WHERE id=LAST_INSERT_ID()")
        return cur.fetchone()

    # Find next session number
    cur.execute(
        "SELECT COALESCE(MAX(session_num),0)+1 AS next FROM test_sessions WHERE user_id=%s",
        (user_id,)
    )
    next_num = cur.fetchone()['next']
    if next_num > 6:
        return None  # All sessions done
    diff = _session_difficulty(next_num)
    cur.execute(
        "INSERT INTO test_sessions (user_id, session_num, attempt_num, difficulty) VALUES (%s,%s,1,%s)",
        (user_id, next_num, diff)
    )
    conn = cur.connection
    conn.commit()
    cur.execute("SELECT * FROM test_sessions WHERE id=LAST_INSERT_ID()")
    return cur.fetchone()


def _get_next_undone_template(cur, user_id: int, session_num: int) -> int | None:
    """Find the next template in the session that hasn't been answered yet."""
    templates = _session_templates(session_num)
    cur.execute(
        "SELECT template_id FROM task_instances WHERE session_id=%s AND user_id=%s AND user_answer IS NOT NULL",
        (session_num, user_id)
    )
    done = {r['template_id'] for r in cur.fetchall()}
    for tid in templates:
        if tid not in done:
            return tid
    return None



@app.post("/srs/event")
async def srs_event(request: Request):
    """Receive user interaction events. UPSERT with counter logic."""
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token)
    if not user_id:
        return {"ok": False, "error": "no session"}
    data = await request.json()
    event_type = data.get("type", "").strip()
    if not event_type:
        return {"ok": False, "error": "missing type"}

    session_id = data.get("session_id") or None
    task_instance_id = data.get("task_instance_id")

    # Validate task_instance_id exists
    if task_instance_id:
        conn = pymysql.connect(**_DB_CFG)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM task_instances WHERE id=%s", (task_instance_id,))
            if not cur.fetchone():
                task_instance_id = None
        finally:
            conn.close()

    # tab events: no task/session binding
    if event_type in ("tab_hidden", "tab_visible"):
        task_instance_id = None
        session_id = None

    # Determine task_type for counter rules
    task_type = None
    if task_instance_id:
        conn = pymysql.connect(**_DB_CFG)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT tt.task_type FROM task_instances ti "
                "JOIN task_templates tt ON ti.template_id = tt.id "
                "WHERE ti.id = %s",
                (task_instance_id,),
            )
            row = cur.fetchone()
            task_type = row["task_type"] if row else None
        finally:
            conn.close()

    # Decision: should we increment the counter?
    incr = 0

    if event_type == "answer_submit":
        # Only count once — first INSERT, subsequent calls are no-op
        incr = 0
        conn = pymysql.connect(**_DB_CFG)
        try:
            cur = conn.cursor()
            if task_instance_id is not None:
                cur.execute(
                    "SELECT 1 FROM user_events "
                    "WHERE user_id=%s AND session_id<=>%s AND task_instance_id=%s AND event_type=%s",
                    (user_id, session_id, task_instance_id, event_type),
                )
            else:
                cur.execute(
                    "SELECT 1 FROM user_events "
                    "WHERE user_id=%s AND session_id<=>%s AND task_instance_id IS NULL AND event_type=%s",
                    (user_id, session_id, event_type),
                )
            if not cur.fetchone():
                incr = 1  # first submit only
        finally:
            conn.close()
    elif event_type in ("click_option", "input_change"):
        conn = pymysql.connect(**_DB_CFG)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT counter, updated_at FROM user_events "
                "WHERE user_id=%s AND session_id<=>%s AND task_instance_id<=>%s AND event_type=%s",
                (user_id, session_id, task_instance_id, event_type),
            )
            row = cur.fetchone()
            if not row:
                incr = 1  # first interaction → +1
            else:
                cur.execute("SELECT NOW()")
                server_now = cur.fetchone()["NOW()"]
                delta = (server_now - row["updated_at"]).total_seconds()
                if event_type == "click_option" and task_type == "single_choice":
                    incr = 1  # always +1 for any (re)selection
                elif event_type == "click_option" and task_type == "multiple_choice":
                    ev_data = data.get("data") or {}
                    if not ev_data.get("checked", True):
                        incr = 1  # uncheck → +1, no timer
                    elif delta >= 4:
                        incr = 1  # check → +1 if ≥4 sec since last change
                elif event_type == "input_change":
                    if delta >= 4:
                        incr = 1  # ≥4 sec since last change → +1
        finally:
            conn.close()

    # UPSERT
    conn = pymysql.connect(**_DB_CFG)
    try:
        cur = conn.cursor()
        # Find existing row (handle NULL-safe comparison for task_instance_id)
        if task_instance_id is not None:
            cur.execute(
                "SELECT id FROM user_events "
                "WHERE user_id=%s AND session_id<=>%s AND task_instance_id=%s AND event_type=%s",
                (user_id, session_id, task_instance_id, event_type),
            )
        else:
            cur.execute(
                "SELECT id FROM user_events "
                "WHERE user_id=%s AND session_id<=>%s AND task_instance_id IS NULL AND event_type=%s",
                (user_id, session_id, event_type),
            )
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE user_events SET counter = counter + %s, updated_at = NOW() WHERE id = %s",
                (incr, row["id"]),
            )
        else:
            cur.execute(
                "INSERT INTO user_events (user_id, session_id, task_instance_id, event_type, counter, created_at, updated_at)"
                " VALUES (%s, %s, %s, %s, %s, NOW(), NOW())",
                (user_id, session_id, task_instance_id, event_type, incr),
            )
        conn.commit()
    finally:
        conn.close()

    return {"ok": True}



@app.get("/srs/test", response_class=HTMLResponse)
async def srs_test(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token) or 1
    review_id = request.query_params.get("review")
    review_id = int(review_id) if review_id and review_id.isdigit() else None

    test_task = None
    feedback = None
    session_done = None
    all_done = False
    start_prompt = None
    session_info = None
    all_tasks = None
    review_mode = False

    conn = pymysql.connect(**_DB_CFG)
    try:
        cur = conn.cursor()
        # Check for an open (incomplete) session first — do NOT auto-create
        cur.execute(
            "SELECT * FROM test_sessions WHERE user_id=%s AND completed_at IS NULL ORDER BY session_num LIMIT 1",
            (user_id,)
        )
        sess = cur.fetchone()

        if sess:
            # Open session exists — show next unanswered task (or review)
            sn = sess['session_num']
            cur.execute(
                "SELECT COUNT(*) AS c FROM task_instances WHERE session_id=%s AND user_id=%s AND user_answer IS NOT NULL",
                (sn, user_id)
            )
            answered = cur.fetchone()['c']
            session_info = {'num': sn, 'answered': answered, 'total': 9}

            # Fetch all tasks for this session (for navigation bar)
            cur.execute(
                "SELECT ti.id, ti.user_answer, ti.is_correct FROM task_instances ti "
                "WHERE ti.session_id=%s AND ti.user_id=%s ORDER BY ti.id",
                (sn, user_id)
            )
            all_task_rows = cur.fetchall()
            all_tasks = [{'id': r['id'], 'has_answer': r['user_answer'] is not None, 'is_correct': bool(r['is_correct'])} for r in all_task_rows]

            # Review mode: show a previously answered question
            if review_id:
                cur.execute(
                    "SELECT ti.*, tt.task_type, tt.answer_type, tt.question_tpl, tt.solution_tpl, "
                    "tt.answer_expr, tt.param_schema "
                    "FROM task_instances ti JOIN task_templates tt ON ti.template_id = tt.id "
                    "WHERE ti.id=%s AND ti.user_id=%s AND ti.session_id=%s AND ti.user_answer IS NOT NULL",
                    (review_id, user_id, sn)
                )
                rev = cur.fetchone()
                if rev:
                    import json
                    params = json.loads(rev['params_used'])
                    ua = rev['user_answer']
                    ua_checked = set()
                    if rev['answer_type'] == 'multiple_choice' and ua:
                        try:
                            ua_checked = set(json.loads(ua))
                        except (json.JSONDecodeError, TypeError):
                            pass
                    test_task = {
                        'id': rev['id'],
                        'template_id': rev['template_id'],
                        'difficulty': rev['difficulty'],
                        'task_type': rev['task_type'],
                        'answer_type': rev['answer_type'],
                        'question': _render_task_question(rev, params),
                        'options': json.loads(rev['options_shown']) if rev.get('options_shown') else None,
                        'solution': _render_task_solution(rev, params),
                        'user_answer': ua,
                        'user_answer_checked': ua_checked,
                    }
                    review_mode = True
            
            if review_mode:
                pass  # showing review — don't generate new task
            else:
                tid = _get_next_undone_template(cur, user_id, sn)
                if tid is None:
                    # All 9 answered for this session — mark complete
                    cur.execute(
                        "SELECT SUM(time_spent_ms) AS t FROM task_instances WHERE session_id=%s AND user_id=%s",
                        (sn, user_id)
                    )
                    total_time = cur.fetchone()['t'] or 0
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM task_instances WHERE session_id=%s AND user_id=%s AND is_correct=1",
                        (sn, user_id)
                    )
                    correct = cur.fetchone()['c']
                    cur.execute(
                        "UPDATE test_sessions SET completed_at=NOW(), total_correct=%s, total_time_ms=%s WHERE id=%s",
                        (correct, total_time, sess['id'])
                    )
                    conn.commit()
                    session_done = {
                        'num': sn,
                        'correct': correct,
                        'total': 9,
                        'time_ms': total_time,
                    }
                else:
                    test_task = task_gen.generate(
                        user_id=user_id, template_id=tid,
                        difficulty=sess['difficulty']
                    )
                    # Tag the instance with session number
                    cur.execute(
                        "UPDATE task_instances SET session_id=%s WHERE id=%s",
                        (sn, test_task['id'])
                    )
                    conn.commit()
        else:
            # No open session — check overall progress
            cur.execute(
                "SELECT MAX(session_num) AS max_sn FROM test_sessions WHERE user_id=%s",
                (user_id,)
            )
            row = cur.fetchone()
            max_sn = row['max_sn'] if row and row['max_sn'] else 0
            if max_sn >= 6:
                all_done = True
            elif max_sn == 0:
                start_prompt = {'num': 1}
            else:
                # Check 12h rule: if < 12h → show retry for last session, else → next session
                admin = _is_admin(cur, user_id)
                if admin:
                    start_prompt = {'num': max_sn + 1}
                else:
                    cur.execute(
                        "SELECT started_at, total_correct, total_time_ms FROM test_sessions "
                        "WHERE user_id=%s AND session_num=%s AND attempt_num=1 LIMIT 1",
                        (user_id, max_sn),
                    )
                    last = cur.fetchone()
                    if last:
                        import datetime as _dt
                        next_avail = last['started_at'] + _dt.timedelta(hours=12)
                        if _dt.datetime.now() < next_avail:
                            session_done = {
                                'num': max_sn,
                                'correct': last['total_correct'] or 0,
                                'total': 9,
                                'time_ms': last['total_time_ms'] or 0,
                            }
                        else:
                            start_prompt = {'num': max_sn + 1}
                    else:
                        start_prompt = {'num': max_sn + 1}
    finally:
        conn.close()

    # Check if admin for UI permissions
    conn2 = pymysql.connect(**_DB_CFG)
    try:
        cur2 = conn2.cursor()
        is_admin = _is_admin(cur2, user_id)
    finally:
        conn2.close()

    return HTMLResponse(render("test.html", title="Тест — SRS",
                               test_task=test_task, feedback=feedback,
                               session_done=session_done, all_done=all_done,
                               start_prompt=start_prompt,
                               session_info=session_info,
                               is_admin=is_admin,
                               all_tasks=all_tasks,
                               review_mode=review_mode))


@app.post("/srs/test/start", response_class=HTMLResponse)
async def srs_test_start(request: Request):
    """Start a new session (on button click), or retry a completed session."""
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token) or 1

    data = await request.form()
    retry_session = data.get("retry_session")
    retry_session = int(retry_session) if retry_session else None

    conn = pymysql.connect(**_DB_CFG)
    try:
        cur = conn.cursor()

        # Check 12h rule for new (non-retry) sessions
        import datetime as _dt
        admin = _is_admin(cur, user_id)
        if not admin:
            cur.execute(
                "SELECT MAX(session_num) AS max_sn FROM test_sessions WHERE user_id=%s",
                (user_id,),
            )
            row = cur.fetchone()
            last_sn = row['max_sn'] if row and row['max_sn'] else 0
            if last_sn > 0:
                cur.execute(
                    "SELECT started_at FROM test_sessions WHERE user_id=%s AND session_num=%s AND attempt_num=1 LIMIT 1",
                    (user_id, last_sn),
                )
                last = cur.fetchone()
                if last:
                    if retry_session is None:
                        # New session: must be >= 12h since last session started
                        if _dt.datetime.now() < last['started_at'] + _dt.timedelta(hours=12):
                            return RedirectResponse(url="/srs/test", status_code=302)
                    else:
                        # Retry: only allowed for max_sn, and only if < 12h since start
                        if retry_session != last_sn:
                            return RedirectResponse(url="/srs/test", status_code=302)
                        if _dt.datetime.now() >= last['started_at'] + _dt.timedelta(hours=12):
                            return RedirectResponse(url="/srs/test", status_code=302)

        sess = _get_or_create_session(cur, user_id, retry_session=retry_session)
        if not sess:
            all_done = True
            return HTMLResponse(render("test.html", title="Тест — SRS",
                                       test_task=None, feedback=None,
                                       session_done=None, all_done=True,
                                       start_prompt=None, session_info=None))

        sn = sess['session_num']
        tid = _get_next_undone_template(cur, user_id, sn)
        if tid is None:
            return RedirectResponse(url="/srs/test", status_code=302)

        test_task = task_gen.generate(
            user_id=user_id, template_id=tid,
            difficulty=sess['difficulty']
        )
        cur.execute(
            "UPDATE task_instances SET session_id=%s WHERE id=%s",
            (sn, test_task['id'])
        )
        conn.commit()

        cur.execute(
            "SELECT COUNT(*) AS c FROM task_instances WHERE session_id=%s AND user_answer IS NOT NULL",
            (sn,)
        )
        answered = cur.fetchone()['c']
        session_info = {'num': sn, 'answered': answered, 'total': 9}
    finally:
        conn.close()

    return HTMLResponse(render("test.html", title="Тест — SRS",
                               test_task=test_task, feedback=None,
                               session_done=None, all_done=False,
                               start_prompt=None, session_info=session_info))


@app.post("/srs/test", response_class=HTMLResponse)
async def srs_test_post(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token) or 1

    data = await request.form()
    instance_id = int(data.get("instance_id", 0))
    user_answer = data.get("user_answer", "").strip()
    time_spent_ms = data.get("time_spent_ms", "")
    is_review = data.get("review") == "1"
    mc_answers = data.getlist("mc_answer")
    if mc_answers:
        user_answer = json.dumps(sorted(mc_answers))

    feedback = None
    test_task = None
    session_done = None
    all_done = False
    session_info = None
    all_tasks = None
    review_mode = False

    if instance_id:
        conn = pymysql.connect(**_DB_CFG)
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM task_instances WHERE id = %s", (instance_id,))
            row = cur.fetchone()
            if row:
                correct = str(row['correct_answer']).strip()
                # Normalize numeric answers: comma→dot, trim whitespace
                ua_norm = user_answer.replace(',', '.').strip()
                try:
                    ua_parsed = json.loads(user_answer) if user_answer.startswith('[') else [ua_norm]
                    ca_parsed = json.loads(correct) if correct.startswith('[') else [correct]
                    is_correct = set(map(str, ua_parsed)) == set(map(str, ca_parsed))
                except (json.JSONDecodeError, ValueError):
                    is_correct = ua_norm == correct
                tms = int(time_spent_ms) if time_spent_ms else 0

                if is_review:
                    # Review: accumulate time, update answer
                    cur.execute(
                        "UPDATE task_instances SET user_answer=%s, is_correct=%s, time_spent_ms = time_spent_ms + %s WHERE id=%s",
                        (user_answer, 1 if is_correct else 0, tms, instance_id)
                    )
                    conn.commit()
                    # Build feedback then redirect to current question
                    cur.execute("SELECT * FROM task_templates WHERE id=%s", (row['template_id'],))
                    tpl = cur.fetchone()
                    params = json.loads(row['params_used'])
                    solution = _render_task_solution(tpl, params) if tpl else ''
                    feedback = {
                        'correct': is_correct,
                        'user_answer': user_answer,
                        'correct_answer': correct,
                        'solution': solution,
                    }
                    test_task = {
                        'id': row['id'],
                        'template_id': row['template_id'],
                        'difficulty': row['difficulty'],
                        'task_type': tpl['task_type'] if tpl else 'numeric_input',
                        'answer_type': tpl['answer_type'] if tpl else 'numeric',
                        'question': _render_task_question(tpl, params) if tpl else '',
                        'options': json.loads(row['options_shown']) if row.get('options_shown') else None,
                        'solution': solution,
                        'user_answer': user_answer,
                    }
                    review_mode = True
                else:
                    cur.execute(
                        "UPDATE task_instances SET user_answer=%s, is_correct=%s, time_spent_ms=%s WHERE id=%s",
                        (user_answer, 1 if is_correct else 0, tms, instance_id)
                    )
                    conn.commit()

                    # Build feedback
                    cur.execute("SELECT * FROM task_templates WHERE id=%s", (row['template_id'],))
                    tpl = cur.fetchone()
                    params = json.loads(row['params_used'])
                    solution = _render_task_solution(tpl, params) if tpl else ''
                    feedback = {
                        'correct': is_correct,
                        'user_answer': user_answer,
                        'correct_answer': correct,
                        'solution': solution,
                    }

                    # Show the answered task
                    test_task = {
                        'id': row['id'],
                        'template_id': row['template_id'],
                        'difficulty': row['difficulty'],
                        'task_type': tpl['task_type'] if tpl else 'numeric_input',
                        'answer_type': tpl['answer_type'] if tpl else 'numeric',
                        'question': _render_task_question(tpl, params) if tpl else '',
                        'options': json.loads(row['options_shown']) if row.get('options_shown') else None,
                        'solution': solution,
                    }

                    # Check if session is complete: all 9 templates answered
                    sn = row.get('session_id')
                    if sn:
                        expected = set(_session_templates(sn))
                        cur.execute(
                            "SELECT template_id FROM task_instances WHERE session_id=%s AND user_id=%s AND user_answer IS NOT NULL",
                            (sn, user_id)
                        )
                        answered_set = {r['template_id'] for r in cur.fetchall()}
                        if expected == answered_set:
                            cur.execute(
                                "SELECT COUNT(*) AS c FROM task_instances WHERE session_id=%s AND user_id=%s AND is_correct=1",
                                (sn, user_id)
                            )
                            correct_cnt = cur.fetchone()['c']
                            cur.execute(
                                "SELECT SUM(time_spent_ms) AS t FROM task_instances WHERE session_id=%s AND user_id=%s",
                                (sn, user_id)
                            )
                            total_time = cur.fetchone()['t']
                            cur.execute(
                                "UPDATE test_sessions SET completed_at=NOW(), total_correct=%s, total_time_ms=%s WHERE user_id=%s AND session_num=%s",
                                (correct_cnt, total_time or 0, user_id, sn)
                            )
                            conn.commit()
                            session_done = {
                                'num': sn,
                                'correct': correct_cnt,
                                'total': 9,
                                'time_ms': total_time or 0,
                            }
                # Get session info + all_tasks for nav bar
                sn_val = row.get('session_id')
                if sn_val:
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM task_instances WHERE session_id=%s AND user_id=%s AND user_answer IS NOT NULL",
                        (sn_val, user_id)
                    )
                    answered = cur.fetchone()['c']
                    session_info = {'num': sn_val, 'answered': answered, 'total': 9}
                    # All tasks for nav bar
                    cur.execute(
                        "SELECT ti.id, ti.user_answer, ti.is_correct FROM task_instances ti "
                        "WHERE ti.session_id=%s AND ti.user_id=%s ORDER BY ti.id",
                        (sn_val, user_id)
                    )
                    all_task_rows = cur.fetchall()
                    all_tasks = [{'id': r['id'], 'has_answer': r['user_answer'] is not None, 'is_correct': bool(r['is_correct'])} for r in all_task_rows]
        finally:
            conn.close()

    # In review mode, redirect to current question (don't stay on review page)
    if is_review and instance_id:
        return RedirectResponse(url="/srs/test", status_code=302)

    return HTMLResponse(render("test.html", title="Тест — SRS",
                               test_task=test_task, feedback=feedback,
                               session_done=session_done, all_done=all_done,
                               start_prompt=None,
                               session_info=session_info,
                               all_tasks=all_tasks,
                               review_mode=review_mode))



def _render_task_question(tpl, params):
    """Render question_tpl with given params."""
    text = tpl['question_tpl']
    for k, v in params.items():
        text = text.replace(f'{{{k}}}', str(v))
    return text


def _render_task_solution(tpl, params):
    """Render solution_tpl with params and computed steps."""
    if not tpl or not tpl.get('solution_tpl'):
        return ''
    from task_generator import compute_arithmetic_steps, ENGINE_FUNCTIONS, _ARITHMETIC_STEPS
    text = tpl['solution_tpl']
    # Build context
    ctx = dict(params)
    # Compute arithmetic steps
    tid = tpl['id']
    if tid in _ARITHMETIC_STEPS:
        steps = compute_arithmetic_steps(tid, params)
        ctx.update(steps)
    # Evaluate answer expression for result and extra context
    expr = tpl['answer_expr']
    if expr in ENGINE_FUNCTIONS:
        fn = ENGINE_FUNCTIONS[expr]
        result = fn({k: _to_num_safe(v) for k, v in params.items()})
        if isinstance(result, dict):
            ctx.update(result)
    else:
        try:
            val = eval(expr, {"__builtins__": {}}, {k: _to_num_safe(v) for k, v in params.items()})
            ctx['result'] = val
        except Exception:
            pass
    # Replace placeholders
    for k, v in ctx.items():
        if isinstance(v, float):
            v = str(round(v, 10)).rstrip('0').rstrip('.')
        text = text.replace(f'{{{k}}}', str(v))
    return text


def _to_num_safe(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        try:
            return float(s)
        except (ValueError, TypeError):
            return s


@app.get("/srs/register", response_class=HTMLResponse)
async def srs_register(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token)
    if _has_student_info(user_id):
        return RedirectResponse(url="/srs/test", status_code=302)
    return HTMLResponse(render("register.html", title="Registration — SRS", error=None))


@app.post("/srs/register")
async def srs_register_post(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token)
    if _has_student_info(user_id):
        return RedirectResponse(url="/srs/test", status_code=302)

    data = await request.form()
    gender = data.get("gender", "").strip()
    highest_education = data.get("highest_education", "").strip()
    age_band = data.get("age_band", "").strip()
    disability = data.get("disability", "N").strip()

    if not gender or not highest_education or not age_band:
        return HTMLResponse(render("register.html", title="Registration — SRS",
                                   error="All fields are required"))

    conn = pymysql.connect(**_DB_CFG)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO student_info (user_id, gender, highest_education, age_band, disability) "
            "VALUES (%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE gender=VALUES(gender), highest_education=VALUES(highest_education), "
            "age_band=VALUES(age_band), disability=VALUES(disability)",
            (user_id, gender, highest_education, age_band, disability)
        )
        conn.commit()
    finally:
        conn.close()

    return RedirectResponse(url="/srs/test", status_code=302)


@app.get("/srs", response_class=HTMLResponse)
async def srs_home(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    user_id = get_session_user(token)
    if user_id and _has_student_info(user_id):
        return RedirectResponse(url="/srs/test", status_code=302)
    return RedirectResponse(url="/srs/register", status_code=302)


@app.get("/srs/enter", response_class=HTMLResponse)
async def srs_enter(request: Request, error: str = None):
    token = request.cookies.get(SRS_COOKIE)
    if get_session_user(token):
        return RedirectResponse(url="/srs", status_code=302)
    return HTMLResponse(render("enter.html", title="Вход — SRS", error=error))


@app.post("/srs/enter")
async def srs_enter_post(request: Request):
    data = await request.form()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    user_id = authenticate(username, password)
    if not user_id:
        return RedirectResponse(url="/srs/enter?error=1", status_code=302)

    token = create_session(user_id)
    next_url = "/srs/test" if _has_student_info(user_id) else "/srs/register"
    resp = RedirectResponse(url=next_url, status_code=302)
    resp.set_cookie(
        key=SRS_COOKIE,
        value=token,
        max_age=30 * 24 * 3600,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return resp


@app.get("/srs/logout")
async def srs_logout(request: Request):
    token = request.cookies.get(SRS_COOKIE)
    if token:
        delete_session(token)
    resp = RedirectResponse(url="/srs/enter", status_code=302)
    resp.delete_cookie(SRS_COOKIE, path="/")
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
