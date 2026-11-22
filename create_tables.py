"""Создание таблиц task_templates и template_difficulty в БД SRS."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from db import execute

# task_templates
execute("""
    CREATE TABLE IF NOT EXISTS task_templates (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        block           VARCHAR(50)     NOT NULL COMMENT 'Арифметика, Логика, Последовательности, ...',
        task_type       VARCHAR(30)     NOT NULL COMMENT 'numeric_input, multiple_choice, sequence_input',
        answer_type     VARCHAR(30)     NOT NULL COMMENT 'numeric, expression, choice_index',
        param_schema    JSON            NOT NULL COMMENT 'описание генерируемых параметров',
        question_tpl    TEXT            NOT NULL COMMENT 'шаблон текста вопроса с {placeholders}',
        answer_expr     TEXT            NOT NULL COMMENT 'Python-выражение или имя функции для вычисления ответа',
        distractors_expr TEXT           NULL COMMENT 'правила генерации дистракторов (NULL если не multiple_choice)',
        solution_tpl    TEXT            NULL COMMENT 'шаблон пошагового объяснения',
        time_limit_sec  INT             NOT NULL DEFAULT 120 COMMENT 'время на задачу в секундах',
        created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
        active          TINYINT(1)      NOT NULL DEFAULT 1 COMMENT '0=выключен, 1=активен',
        INDEX idx_block (block),
        INDEX idx_active (active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
""")

print("✓ task_templates создана")

# template_difficulty
execute("""
    CREATE TABLE IF NOT EXISTS template_difficulty (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        template_id     INT             NOT NULL,
        difficulty      TINYINT         NOT NULL COMMENT 'уровень сложности: 1, 2, 3',
        param_overrides JSON            NOT NULL COMMENT 'переопределение параметров и/или таймаута для этой сложности',
        created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_template_difficulty (template_id, difficulty),
        CONSTRAINT fk_difficulty_template
            FOREIGN KEY (template_id) REFERENCES task_templates(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
""")

print("✓ template_difficulty создана")


from db import query
print("\nСозданные таблицы:")
all_tables = query("SHOW TABLES")
for t in all_tables:
    name = list(t.values())[0]
    if name.startswith('task_') or name.startswith('template_'):
        cols = query(f"SHOW COLUMNS FROM `{name}`")
        print(f"  {name}: {len(cols)} полей")
        for c in cols:
            null = 'NULL' if c['Null'] == 'YES' else 'NOT NULL'
            default = f" DEFAULT {c['Default']}" if c['Default'] is not None else ''
            comment = f" — {c['Comment']}" if c.get('Comment') else ''
            print(f"    {c['Field']:20s} {c['Type']:15s} {null:8s}{default}{comment}")
