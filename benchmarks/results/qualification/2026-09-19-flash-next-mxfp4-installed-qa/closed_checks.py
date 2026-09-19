"""Unmodified checker functions from the owned frozen QA helper."""
import re
from dataclasses import fields
from checkers import CheckResult, check_exact_answer, check_regex, run_all_checks
from prompt_def import QaPrompt
PROMPT_FIELD_NAMES = {item.name for item in fields(QaPrompt)}

def item_to_prompt(item: dict) -> QaPrompt:
    payload = {}
    for name in PROMPT_FIELD_NAMES:
        if name in item:
            payload[name] = item[name]
    checker = item.get('checker') or {}
    if 'exact_answer' not in payload and checker.get('value') is not None:
        payload['exact_answer'] = str(checker['value'])
    if 'exact_answer_aliases' not in payload:
        payload['exact_answer_aliases'] = list(checker.get('aliases') or [])
    if 'exact_match' not in payload:
        payload['exact_match'] = checker.get('match') or 'auto'
    if 'regex_patterns' not in payload:
        payload['regex_patterns'] = list(checker.get('patterns') or [])
    payload.setdefault('keywords', [])
    payload.setdefault('regex_patterns', [])
    payload.setdefault('min_length', 1)
    payload.setdefault('max_repetition_ratio', 0.3)
    payload.setdefault('description', '')
    payload.setdefault('min_test_count', 0)
    payload.setdefault('json_expected_total', None)
    payload.setdefault('exact_answer_aliases', [])
    payload.setdefault('exact_match', 'auto')
    payload.setdefault('system', None)
    return QaPrompt(**payload)

def checker_spec(item: dict) -> dict:
    spec = dict(item.get('checker') or {})
    if not spec:
        spec = {
            'type': 'exact_answer',
            'value': item.get('exact_answer'),
            'aliases': list(item.get('exact_answer_aliases') or []),
            'match': item.get('exact_match') or 'auto',
        }
    spec.setdefault('type', 'exact_answer')
    if spec.get('value') is None:
        spec['value'] = item.get('exact_answer')
    spec.setdefault('aliases', list(item.get('exact_answer_aliases') or []))
    spec.setdefault('match', item.get('exact_match') or 'auto')
    spec.setdefault('patterns', list(item.get('regex_patterns') or []))
    return spec

def _last_line(text: str) -> str:
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1] if lines else ''

def _parse_set(text: str) -> set[str]:
    last = _last_line(text).strip('{}[]()')
    parts = [part.strip() for part in re.split(r'[,;]+', last) if part.strip()]
    if not parts:
        parts = [part for part in last.split() if part]
    return set(parts)

def _parse_ordered(text: str, sep: str) -> list[str]:
    last = _last_line(text)
    if sep == '\t':
        return [part.strip() for part in last.split('\t')]
    if sep == '|':
        return [part.strip() for part in last.split('|') if part.strip()]
    if sep == '-':
        return [part.strip() for part in last.split('-') if part.strip()]
    return [part.strip() for part in last.split(sep) if part.strip()]

def evaluate_checker(text: str, item: dict) -> CheckResult:
    """Closed-answer spec on top of qa/checkers.py primitives."""
    spec = checker_spec(item)
    kind = spec['type']
    prompt = item_to_prompt(item)
    if kind == 'regex':
        return check_regex(text, prompt)
    if kind == 'set':
        members = {str(m) for m in spec.get('members') or []}
        if not members:
            members = {part.strip() for part in str(spec.get('value') or '').split(',') if part.strip()}
        parsed = _parse_set(text)
        passed = parsed == members
        return CheckResult(
            'set',
            passed,
            f'parsed={sorted(parsed)} expected={sorted(members)}',
            1.0 if passed else 0.0,
            hard=True,
        )
    if kind == 'ordered_list':
        expected = [str(x) for x in spec.get('items') or []]
        sep = spec.get('sep', ',')
        parsed = _parse_ordered(text, sep) if expected else []
        if expected and parsed == expected:
            return CheckResult('ordered_list', True, f'list={parsed}', 1.0, hard=True)
        return check_exact_answer(text, prompt)
    if kind == 'integer':
        exact = check_exact_answer(text, prompt)
        if exact.passed:
            return exact
        found = re.findall(r'-?\d+', text or '')
        want = str(spec.get('value'))
        passed = bool(found) and found[-1] == want
        return CheckResult(
            'integer',
            passed,
            f'last_int={found[-1] if found else None} expected={want}',
            1.0 if passed else 0.0,
            hard=True,
        )
    return check_exact_answer(text, prompt)
