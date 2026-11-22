"""
Task Generator Engine for SRS.

Reads task_templates + template_difficulty from DB,
generates concrete task instances for users.

Usage:
    gen = TaskGenerator(db_config)
    instance = gen.generate(user_id=1, template_id=7, difficulty=2)
    # → task_instances row created, returns dict with question, options, etc.
"""
import json
import random
import re
from typing import Any, Optional

import pymysql
from pymysql.cursors import DictCursor



def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _to_num(s):
    """Convert string to int/float if possible, else keep as string."""
    if not isinstance(s, str):
        return s
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def odd_one_out(params: dict) -> dict:
    """id=8: find the number with different parity among A-E.
    Returns index (0-4) as result, actual value as odd_value."""
    vals = {}
    for key in 'ABCDE':
        if key in params:
            vals[key] = params[key]
    evens = [v for v in vals.values() if v % 2 == 0]
    odds = [v for v in vals.values() if v % 2 == 1]
    if len(evens) == 1:
        odd_val = evens[0]
    else:
        odd_val = odds[0]
    idx = list(vals.values()).index(odd_val)
    return {'result': idx, 'odd_value': odd_val}


def find_divisible(params: dict) -> dict:
    """id=9: find which of A-E is divisible by F.
    Returns index (0-4) as result, actual values as divisible_value + quotient."""
    F = params['F']
    for i, key in enumerate('ABCDE'):
        if key in params and params[key] % F == 0:
            return {'result': i, 'divisible_value': params[key], 'quotient': params[key] // F}
    return {'result': None}


def find_primes(params: dict) -> dict:
    """id=10: find all primes among A-E. Returns indices."""
    primes = []
    indices = []
    for i, key in enumerate('ABCDE'):
        if key in params and _is_prime(params[key]):
            primes.append(params[key])
            indices.append(i)
    return {'result': indices, 'primes': ', '.join(map(str, primes)) if primes else 'none'}


def find_odd_category(params: dict) -> dict:
    """id=11: find the item that doesn't belong (vehicle among colors)."""
    colors_set = {'Red', 'Blue', 'Green', 'Yellow', 'Black', 'White',
                  'Orange', 'Purple', 'Pink', 'Brown'}
    for key in 'ABCD':
        if key in params and params[key] not in colors_set:
            # This is the vehicle — it's the odd one
            idx = 'ABCD'.index(key)
            return {'result': idx, 'odd_one': params[key]}
    return {'result': None}


def compare_sum_product(params: dict) -> dict:
    """id=12: compare A+B vs C×D, return index of greater expression."""
    s = params['A'] + params['B']
    p = params['C'] * params['D']
    expr1 = f'{params["A"]} + {params["B"]}'
    expr2 = f'{params["C"]} × {params["D"]}'
    if s > p:
        return {
            'result': 0,
            'sum': s, 'product': p,
            'greater_expr': expr1,
            'expr1': expr1, 'expr2': expr2
        }
    else:
        return {
            'result': 1,
            'sum': s, 'product': p,
            'greater_expr': expr2,
            'expr1': expr1, 'expr2': expr2
        }


def card_prob(params: dict) -> dict:
    """id=15: probability of drawing a specific card from 52-card deck."""
    ranks = {'Ace': 4, 'King': 4, 'Queen': 4, 'Jack': 4}
    suits = {'Heart': 13, 'Diamond': 13, 'Club': 13, 'Spade': 13}
    A = params['A']
    count = ranks.get(A, suits.get(A, 0))
    return {'result': round(count / 52, 2), 'count': count}


def most_likely_prob(params: dict) -> dict:
    """id=16: probability of drawing the most common ball color."""
    colors = ['red', 'blue', 'green']
    vals = [params['A'], params['B'], params['C']]
    total = sum(vals)
    max_count = max(vals)
    max_idx = vals.index(max_count)
    return {
        'result': round(max_count / total, 2),
        'total': total,
        'max_color': colors[max_idx],
        'max_count': max_count
    }


# Registry of named functions
ENGINE_FUNCTIONS = {
    'find_divisible': find_divisible,
    'find_primes': find_primes,
    'find_odd_category': find_odd_category,
    'compare_sum_product': compare_sum_product,
    'card_prob': card_prob,
    'most_likely_prob': most_likely_prob,
    'odd_one_out': odd_one_out,
}


# ═══════════════════════════════════════════════════════════════════
# Arithmetic Step-by-Step Evaluator (templates 1-6)
# ═══════════════════════════════════════════════════════════════════

# Each entry: list of (step_name, expression) where expression uses
# original params + previously computed steps
_ARITHMETIC_STEPS = {
    1: [  # A + B * C
        ('step1', 'B * C'),
    ],
    2: [  # (A + B) * (C - D)
        ('step1', 'A + B'),
        ('step2', 'C - D'),
    ],
    3: [  # (A + B * C) / D
        ('step1', 'B * C'),
        ('step2', 'A + step1'),
    ],
    4: [  # (A*D + B*C) / (B*D) — fraction addition
        ('step1', 'A*D + B*C'),
        ('step2', 'B*D'),
    ],
    5: [  # (A * B + C * D) / 100 — percentages
        ('step1', 'A * B / 100'),
        ('step2', 'C * D / 100'),
    ],
    6: [  # ((A + B) * C - D) / E
        ('step1', 'A + B'),
        ('step2', 'step1 * C'),
        ('step3', 'step2 - D'),
    ],
}


def compute_arithmetic_steps(template_id: int, params: dict) -> dict:
    """Compute intermediate steps for arithmetic templates.
    Returns dict with 'step1', 'step2', ... and 'result'.
    """
    if template_id not in _ARITHMETIC_STEPS:
        return {}
    ctx = dict(params)
    for step_name, expr in _ARITHMETIC_STEPS[template_id]:
        val = eval(expr, {"__builtins__": {}}, ctx)
        ctx[step_name] = val
    return {k: v for k, v in ctx.items() if k.startswith('step')}


class ParamGenerator:
    """Parses param_schema JSON and generates valid parameter sets."""

    MAX_RETRIES = 1000

    def parse_range(self, spec: Any, known_params: dict = None) -> list:
        """Parse a value specification into a list of possible values.
        'int[1,10]' → [1,2,...,10] (ints)
        'int[1,B-1]' → resolved using known_params (e.g., B=5 → [1,2,3,4])
        'set[2,3,4]' → [2,3,4] (ints)
        'set[Ace,King]' → ['Ace','King'] (strings)
        ['Red','Blue'] → ['Red','Blue'] (already a list)
        [1,5] → [1,2,3,4,5] (min,max range, ints)
        """
        if isinstance(spec, str):
            # Try int[min,max] — may contain param references like B-1
            m_int = re.match(r'int\[(.+)\s*,\s*(.+)\]', spec)
            if m_int:
                lo_str, hi_str = m_int.group(1), m_int.group(2)
                # Resolve parameter references
                lo = self._resolve_bound(lo_str, known_params or {})
                hi = self._resolve_bound(hi_str, known_params or {})
                if lo is not None and hi is not None:
                    return list(range(lo, hi + 1))
            m_set = re.match(r'set\[(.+)\]', spec)
            if m_set:
                items = [s.strip() for s in m_set.group(1).split(',')]
                # Convert to ints where possible
                return [_to_num(i) for i in items]
            return [spec]  # literal string
        if isinstance(spec, list):
            if len(spec) == 2 and isinstance(spec[0], int) and isinstance(spec[1], int):
                return list(range(spec[0], spec[1] + 1))
            return spec  # list of literals
        return [spec]

    def _resolve_bound(self, expr: str, known: dict) -> Optional[int]:
        """Resolve a bound expression like '20' or 'B-1' or 'B'."""
        expr = expr.strip()
        try:
            return int(expr)
        except ValueError:
            pass
        # Try as simple param reference: 'B' → known['B']
        if expr in known:
            return known[expr]
        # Try as expression like 'B-1' or '2*B'
        try:
            return int(eval(expr, {"__builtins__": {}}, known))
        except Exception:
            return None

    def generate(self, param_schema: dict) -> dict:
        """Generate a random valid parameter set from the schema.
        Retries on constraint violations.
        """
        # Separate regular params from meta keys
        meta_keys = {'colors', 'vehicles', 'prob_map', 'constraint', 'constraints',
                     'seq_length', 'count', 'range', 'pattern'}
        param_keys = {k for k in param_schema if k not in meta_keys}

        # Special: pattern-based generation (template 8: odd_one_out)
        if 'pattern' in param_schema and param_schema.get('pattern', '').startswith('4 same parity'):
            return self._gen_parity_pattern(param_schema)

        # Resolve reference lists (colors, vehicles)
        refs = {}
        for ref_key in ['colors', 'vehicles']:
            if ref_key in param_schema:
                refs[ref_key] = self.parse_range(param_schema[ref_key])

        # Count how many keys reference 'colors' — if >1, pick unique values
        color_keys = [k for k in param_keys
                      if isinstance(param_schema[k], str) and param_schema[k] == 'colors']
        unique_colors = None
        if len(color_keys) > 1:
            unique_colors = random.sample(refs['colors'], min(len(color_keys), len(refs['colors'])))

        # Sort keys so that params WITHOUT dependencies generate first
        ordered_keys = self._order_by_dependency(param_keys, param_schema)

        for attempt in range(self.MAX_RETRIES):
            params = {}
            color_idx = 0
            for key in ordered_keys:
                spec = param_schema[key]
                if isinstance(spec, str) and spec in refs:
                    if spec == 'colors' and unique_colors is not None:
                        params[key] = unique_colors[color_idx % len(unique_colors)]
                        color_idx += 1
                    else:
                        params[key] = random.choice(refs[spec])
                else:
                    values = self.parse_range(spec, params)
                    params[key] = random.choice(values)

            # Ensure all values are proper types (ints where possible)
            params = {k: _to_num(v) for k, v in params.items()}

            # Ensure uniqueness for templates with A-E keys (avoid duplicates)
            letter_keys = [k for k in params if len(k) == 1 and k.isalpha() and k.isupper()]
            if len(letter_keys) >= 3:
                vals = [params[k] for k in letter_keys]
                if len(vals) != len(set(vals)):
                    continue  # retry

            # Compute sequence values if seq_length is present (template 7)
            if 'seq_length' in param_schema:
                params = self._compute_sequence(params, param_schema['seq_length'])

            # Check constraints
            ok = self._check_constraints(params, param_schema, refs)
            if ok:
                return params

        raise RuntimeError(f"Failed to generate valid params after {self.MAX_RETRIES} attempts")

    def _gen_parity_pattern(self, schema: dict) -> dict:
        """Generate params for '4 same parity, 1 different' pattern."""
        range_spec = schema.get('range', 'int[4,100]')
        values = self.parse_range(range_spec)
        # Pick a parity (even or odd) for the majority
        majority_parity = random.choice([0, 1])  # 0=even, 1=odd
        # Generate 4 numbers with majority parity, all unique
        majority_vals = []
        seen = set()
        for _ in range(4):
            while True:
                v = random.choice(values)
                if v % 2 == majority_parity and v not in seen:
                    majority_vals.append(v)
                    seen.add(v)
                    break
        # Generate 1 number with opposite parity
        while True:
            v = random.choice(values)
            if v % 2 != majority_parity:
                odd_val = v
                break
        # Combine and shuffle into A-E
        all_vals = majority_vals + [odd_val]
        random.shuffle(all_vals)
        return {label: all_vals[i] for i, label in enumerate('ABCDE')}

    def _order_by_dependency(self, keys: set, schema: dict) -> list:
        """Order keys so that keys without param references come first."""
        deps = []
        nodeps = []
        for key in keys:
            spec = str(schema[key])
            has_dep = any(
                ref in spec and ref != key
                for ref in keys
                if len(ref) == 1 and ref.isalpha()  # single-letter param refs
            )
            if has_dep:
                deps.append(key)
            else:
                nodeps.append(key)
        return nodeps + deps

    def _compute_sequence(self, params: dict, seq_length: int) -> dict:
        """Compute geometric sequence: A, A*k, A*k², ...
        Populates B, C, D, ... from A and k.
        """
        label = 'A'
        current = params['A']
        k = params['k']
        for i in range(1, seq_length):
            current = current * k
            next_label = chr(ord(label) + 1)  # B, C, D, ...
            params[next_label] = current
            label = next_label
        return params

    def _check_constraints(self, params: dict, schema: dict, refs: dict) -> bool:
        """Check all constraints on generated parameters."""
        constraints = schema.get('constraint', '')
        if isinstance(constraints, str):
            constraints = constraints.strip()
        if not constraints:
            return True

        # Split by both ',' and ';' — some schemas use commas
        raw_parts = []
        for part in re.split(r'[;,]', str(constraints)):
            part = part.strip()
            if part:
                raw_parts.append(part)
        # Re-join comma-separated relational constraints as single statements
        statements = self._merge_relational_parts(raw_parts)

        for c in statements:
            if not self._check_one(c, params):
                return False
        return True

    def _merge_relational_parts(self, parts: list) -> list:
        """Merge comma-separated relational constraints like 'A < B, C < D'."""
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if re.match(r'^\w+\s*[<>=!]+\s*\w+$', part) and i + 1 < len(parts):
                # Might be part of a comma-separated relational chain
                merged = part
                j = i + 1
                while j < len(parts) and re.match(r'^\w+\s*[<>=!]+\s*\w+$', parts[j]):
                    merged += ', ' + parts[j]
                    j += 1
                result.append(merged)
                i = j
            else:
                result.append(part)
                i += 1
        return result

    def _check_one(self, constraint: str, params: dict) -> bool:
        """Check a single constraint against params."""
        c = constraint.strip()

        # Relational: 'A < B' or 'A < B, C < D'
        if re.match(r'^\w+\s*[<>=!]+\s*\w+', c):
            relations = [r.strip() for r in c.split(',')]
            for rel in relations:
                try:
                    if not eval(rel, {"__builtins__": {}}, params):
                        return False
                except Exception:
                    return False
            return True

        # Math: 'gcd(A,B)=1', '(A*B)%100==0'
        if '=' in c or '%' in c or 'gcd' in c.lower():
            try:
                # Replace gcd with math.gcd
                import math
                local_params = dict(params)
                local_params['gcd'] = math.gcd
                # Support both 'expr = val' and Python equality 'expr == val'
                expr = c.replace(' = ', ' == ').replace('=', '==', 1) if '==' not in c else c
                if '==' in expr:
                    result = eval(expr, {"__builtins__": {}}, local_params)
                    return bool(result)
                return True  # Can't evaluate, assume ok
            except Exception:
                return False

        # Named constraints
        if c == 'all_different':
            vals = list(params.values())
            return len(vals) == len(set(vals))

        elif c == 'exactly_one_divisible_by_F':
            F = params.get('F')
            if F is None:
                return False
            count = sum(1 for k in 'ABCDE' if k in params and int(params[k]) % int(F) == 0)
            return count == 1

        elif c == 'at_least_one_prime':
            return any(_is_prime(int(params[k])) for k in 'ABCDE' if k in params)

        elif c == 'sum_neq_product_abs_diff_5_30':
            s = params['A'] + params['B']
            p = params['C'] * params['D']
            diff = abs(s - p)
            return s != p and 5 <= diff <= 30

        elif c == '3_colors_1_vehicle_shuffled':
            pass  # Handled by the ref system

        return True



class AnswerEvaluator:
    """Evaluates answer_expr with given parameters."""

    def evaluate(self, answer_expr: str, params: dict) -> tuple:
        """Returns (correct_answer_value, extra_context_dict).
        correct_answer is a string representation for storage in DB.
        extra_context has additional variables for solution_tpl rendering.
        """
        if not answer_expr or not answer_expr.strip():
            return '', {}

        expr = answer_expr.strip()

        # Try named function first
        if expr in ENGINE_FUNCTIONS:
            fn = ENGINE_FUNCTIONS[expr]
            result = fn(params)
            if isinstance(result, dict):
                answer = result.get('result', '')
                extra = dict(result)  # keep all keys including 'result'
            else:
                answer = result
                extra = {}
            return self._serialize(answer), extra

        # Treat as Python expression
        try:
            safe_builtins = {
                'abs': abs, 'min': min, 'max': max, 'int': int, 'float': float,
                'round': round, 'True': True, 'False': False, 'None': None,
                'len': len, 'range': range, 'sum': sum,
            }
            result = eval(expr, {"__builtins__": safe_builtins}, params)
            return self._serialize(result), {'result': result}
        except Exception as e:
            raise ValueError(f"Failed to evaluate answer_expr '{expr}': {e}") from e

    def _serialize(self, value: Any) -> str:
        """Convert answer to string for DB storage."""
        if isinstance(value, list):
            return json.dumps(value)
        if isinstance(value, float):
            # Round to 4 decimal places, remove trailing zeros
            return str(round(value, 10)).rstrip('0').rstrip('.')
        if isinstance(value, int):
            return str(value)
        return str(value)



class DistractorGenerator:
    """Generates plausible wrong answers for single_choice questions."""

    def generate(self, distractors_expr: Optional[str], params: dict,
                 correct_answer: str, evaluator: AnswerEvaluator) -> list:
        """Return a list of incorrect options as strings."""
        if not distractors_expr:
            return []

        correct_val = self._parse_answer(correct_answer)

        if distractors_expr == 'geo_distractors':
            return self._geo_distractors(correct_val, params)
        else:
            # Try evaluating as expression
            try:
                result = evaluator.evaluate(distractors_expr, params)
                return result if isinstance(result, list) else [str(result)]
            except Exception:
                return []

    def _parse_answer(self, answer_str: str) -> Any:
        """Try to parse answer back to int/float for comparison."""
        try:
            return int(answer_str)
        except ValueError:
            try:
                return float(answer_str)
            except ValueError:
                return answer_str

    def _geo_distractors(self, correct: Any, params: dict) -> list:
        """Generate geometrically/arithmetically close wrong answers."""
        if not isinstance(correct, (int, float)):
            return []
        distractors = set()
        k = params.get('k', 2)
        base = params.get('A', correct)
        # Generate nearby values
        for offset in [-1, -2, 1, 2]:
            d = correct + offset * k
            if d > 0 and d != correct:
                distractors.add(str(d))
        return list(distractors)[:3]



class TaskGenerator:
    """Main entry point for generating task instances."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.param_gen = ParamGenerator()
        self.evaluator = AnswerEvaluator()
        self.distractor_gen = DistractorGenerator()

    def _get_conn(self):
        cfg = dict(self.db_config)
        if 'cursorclass' not in cfg:
            cfg['cursorclass'] = DictCursor
        return pymysql.connect(**cfg)

    def generate(self, user_id: int, template_id: int, difficulty: int = 2) -> dict:
        """Generate a task instance for a user. Returns dict with question details."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # 1. Load template
            cur.execute(
                "SELECT * FROM task_templates WHERE id = %s AND active = 1",
                (template_id,)
            )
            tpl = cur.fetchone()
            if not tpl:
                raise ValueError(f"Template {template_id} not found or inactive")

            # 2. Load difficulty overrides
            cur.execute(
                "SELECT param_overrides FROM template_difficulty "
                "WHERE template_id = %s AND difficulty = %s",
                (template_id, difficulty)
            )
            diff_row = cur.fetchone()

            # 3. Build effective param_schema (merge overrides)
            base_schema = json.loads(tpl['param_schema'])
            if diff_row:
                overrides = json.loads(diff_row['param_overrides'])
                for k, v in overrides.items():
                    if k in base_schema:
                        # Override just the values, keep the spec structure
                        base_schema[k] = v

            # 4. Generate parameters
            params = self.param_gen.generate(base_schema)

            # 5. Evaluate answer
            correct_answer, extra_context = self.evaluator.evaluate(
                tpl['answer_expr'], params
            )

            # 5b. Compute arithmetic steps for templates 1-6
            if template_id in _ARITHMETIC_STEPS:
                steps = compute_arithmetic_steps(template_id, params)
                extra_context = {**extra_context, **steps}

            # 6. Generate options for single_choice
            options = None
            if tpl['task_type'] in ('single_choice', 'multiple_choice'):
                options = self._build_options(tpl, params, correct_answer)

            # 7. Render question text
            question_text = self._render_template(tpl['question_tpl'], params, extra_context)
            solution_text = self._render_template(tpl['solution_tpl'] or '', params, extra_context)

            # 8. Insert task_instance
            cur.execute(
                """INSERT INTO task_instances
                   (user_id, template_id, difficulty, params_used, correct_answer,
                    options_shown)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    user_id, template_id, difficulty,
                    json.dumps(params),
                    correct_answer,
                    json.dumps(options) if options else None
                )
            )
            conn.commit()
            instance_id = cur.lastrowid

            return {
                'id': instance_id,
                'template_id': template_id,
                'difficulty': difficulty,
                'task_type': tpl['task_type'],
                'answer_type': tpl['answer_type'],
                'question': question_text,
                'options': options,
                'solution': solution_text,
                'params': params,
            }
        finally:
            conn.close()

    def _build_options(self, tpl: dict, params: dict, correct_answer: str) -> list:
        """Build options list for single_choice / multiple_choice questions."""
        answer_type = tpl['answer_type']
        task_type = tpl['task_type']

        if answer_type == 'choice_index' and task_type == 'single_choice':
            # Check if this is a comparison template (A+B vs C×D)
            if tpl['answer_expr'] == 'compare_sum_product':
                return [
                    f'{params["A"]} + {params["B"]}',
                    f'{params["C"]} × {params["D"]}'
                ]
            # Default: options are the parameter values (A, B, C, D, E)
            options = []
            for key in 'ABCDE':
                if key in params:
                    options.append(str(params[key]))
            return options

        elif answer_type == 'multiple_choice':
            # All values are options, can select multiple
            options = []
            for key in 'ABCDE':
                if key in params:
                    options.append(str(params[key]))
            return options

        return None

    def _render_template(self, tpl: str, params: dict, extra: dict) -> str:
        """Render a template string with {param} placeholders."""
        if not tpl:
            return tpl
        context = {**params}
        if extra:
            context.update(extra)
        # Try to evaluate {expr} patterns like {6-A}, {2**A}, etc.
        result = tpl
        # Find all {placeholders}
        for match in re.finditer(r'\{([^}]+)\}', result):
            placeholder = match.group(1)
            value = context.get(placeholder)
            if value is None and placeholder in params:
                value = params[placeholder]
            if value is None:
                # Try evaluating as an expression
                try:
                    value = eval(placeholder, {"__builtins__": {}}, context)
                except Exception:
                    value = f'{{{placeholder}}}'  # leave as-is
            if isinstance(value, float):
                value = str(round(value, 10)).rstrip('0').rstrip('.')
            result = result.replace(f'{{{placeholder}}}', str(value), 1)
        return result
