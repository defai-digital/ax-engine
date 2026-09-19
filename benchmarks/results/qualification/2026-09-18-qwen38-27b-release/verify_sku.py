"""Independently verify hosted-wheel qualification and retained QA grading."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
repo = next(parent for parent in root.parents if (parent / 'qa/checkers.py').is_file())
sys.path.insert(0, str(repo / 'qa'))
from checkers import run_response_checks
from prompts import get_prompt_by_id

def read(path):
    return json.loads((root / path).read_text())

if (root / 'publication.json').is_file():
    for name, expected in read('publication.json')['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

build = read('build-manifest-product-final-installed.json')
candidate = read('candidate/candidate-manifest.json')
assert build['source_commit'] == candidate['git_commit']
assert build['wheel_sha256'] == candidate['wheel']['sha256']
assert build['dirty'] is False
assert not any(p['name'].lower() == 'mlx' for p in read('installed-packages.json'))
q = read('qualification/qualification.json')
assert q['schema'] == 4 and q['status'] == 'passed'
assert q['source_commit'] == build['source_commit'] and q['build_manifest'] == build
assert hashlib.sha256((root / 'build-manifest-product-final-installed.json').read_bytes()).hexdigest() == q['build_manifest_sha256']
assert q['host']['chip'] == 'Apple M4 Pro' and q['host']['model'] == 'Mac16,11'
assert q['host']['memory_bytes'] == 64 * 1024**3 and q['runtime_overrides'] == []
assert q['mtp_certification'] == {k: 'not_assessed' for k in ('MTP-S', 'MTP-P', 'MTP-D')}

def verify_report(report, count):
    assert report['totals']['items'] == report['totals']['hard_passed'] == count
    assert report['checker_contract'] == 'ax.qa.complete_answers.v2'
    assert len(report['commit']) >= 8 and build['source_commit'].startswith(report['commit'])
    for row in report['results']:
        response = row['response']
        assert response['finish_reason'] == 'stop' and not response['error']
        grading = run_response_checks(response['text'], get_prompt_by_id(row['prompt_id']), response['finish_reason'])
        assert grading.auto_pass == row['hard_pass']
        assert [vars(check) for check in grading.checks] == row['report']['checks']

cells = [q['default_route'], *q['cells']]
assert [c['mode'] for c in cells] == ['ngram', 'direct', 'mtp']
for cell in cells:
    mode = cell['mode']
    assert cell['status'] == 'ok' and cell['surface_passed'] is True
    assert cell['qa_items'] == cell['qa_hard_passed'] == 32
    verify_report(read(f'qualification/report-{mode}-qwen3.8-27b.json'), 32)
    surface = read(f'qualification/surface-{mode}-qwen3.8-27b.json')
    assert all(r['passed'] and not r['skipped'] for r in surface['results'] if r['hard'])
    route = read(f'qualification/server-route-{mode}-qwen3.8-27b.json')
    assert route['status'] == 'finished' and len(route['output_tokens']) == 64
    mtp = route['performance']['mtp']
    assert mtp['requested'] is (mode == 'mtp') and mtp['active'] is (mode == 'mtp')
    assert mtp['direct_fallback_steps'] == 0

rows = read('product-final-lifecycle/results.json')
assert {r['mode'] for r in rows} == {'default', 'mtp'}
for row in rows:
    mode = row['mode']
    assert row['source_commit'] == build['source_commit']
    assert row['wheel_sha256'] == build['wheel_sha256']
    assert row['server_sha256'] == build['server_sha256']
    assert row['passed'] and row['owned_servers_reaped'] and row['recovery_passed']
    assert row['soak_exit_code'] == row['full_qa_exit_code'] == 0
    assert row['cold']['output_tokens'] == row['warm']['output_tokens'] == row['recovery']['output_tokens']
    assert len(row['cold']['output_tokens']) == 64
    for phase in ('cold', 'warm', 'recovery'):
        assert row[phase]['metrics']['ax_engine_mtp_direct_fallback_steps_total'] == 0
        assert row[phase]['mtp']['requested'] is (mode == 'mtp')
        assert row[phase]['mtp']['active'] is (mode == 'mtp')
        if mode == 'default':
            assert row[phase]['metrics']['ax_engine_mtp_draft_tokens_total'] == 0
    report = read(f'product-final-lifecycle/{mode}-full-qa.json')
    verify_report(report, 158)
    assert report['bank_size'] == report['sample_size'] == 79
    for prompt in report['sampled_ids']:
        assert sorted(r['stream'] for r in report['results'] if r['prompt_id'] == prompt) == [False, True]
    soak = read(f'product-final-lifecycle/{mode}-soak.json')
    assert soak['verdict'] == 'pass' and soak['quiescent'] is True
    assert soak['failure_reasons'] == [] and len(soak['outcomes']) == 15
    assert soak['workload']['rounds'] == 3 and soak['workload']['require_backpressure'] is True
    assert soak['workload']['stalled_output_tokens'] == 8192
    assert soak['workload']['stalled_hold_ms'] == 180000
    assert all(r['output_tokens'] > 0 for r in soak['outcomes'] if r['kind'] == 'stalled')
    assert soak['metrics_after']['ax_engine_mtp_direct_fallback_steps_total'] == 0
    assert soak['metric_peaks']['ax_engine_generation_buffered_stream_events'] >= 1 or soak['counter_deltas']['ax_engine_generation_stream_backlog_overflows_total'] > 0
if (root / 'release-acceptance.json').is_file():
    acceptance = read('release-acceptance.json')
    release = read('released-manifest.json')
    assert acceptance['published'] and acceptance['independent_verification_passed']
    assert acceptance['source_commit'] == release['git_commit'] == build['source_commit']
    assert acceptance['wheel']['sha256'] == build['wheel_sha256']
    assert acceptance['archive'] == release['archive']
    assert hashlib.sha256((root / 'released-manifest.json').read_bytes()).hexdigest() == acceptance['released_manifest_sha256']
    assert read('public-wheel-doctor.json')['result'] == 'ready'
    assert {p['name'] for p in read('public-wheel-installed-packages.json')} == {'ax-engine', 'pip'}
    assert read('standalone-verification/payload-hashes.json') == acceptance['published_standalone']['payload_sha256']
    helper = read('standalone-verification/helper-python.json')
    assert not helper['ax_engine'] and not helper['mlx']
    standalone = read('standalone-verification/results.json')
    assert {r['mode'] for r in standalone} == {'default', 'mtp', 'local_directory'}
    for result in standalone:
        mode = result['mode']
        assert result['passed'] and result['owned_server_reaped']
        assert result['output_tokens'] == 64 and result['mtp']['direct_fallback_steps'] == 0
        assert result['mtp']['active'] is (mode == 'mtp')
        response = read(f'standalone-verification/{mode}-response.json')
        baseline = next(r for r in rows if r['mode'] == ('mtp' if mode == 'mtp' else 'default'))
        assert response['output_tokens'] == baseline['cold']['output_tokens']
print('Hosted-wheel SKU receipts and all 412 QA grades verified; MTP certification remains separate.')
