#!/usr/bin/env python3
"""Verify published receipts offline; does not execute the model or certify MTP."""
import argparse
import hashlib
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def verify():
    publication = read('publication.json')
    for name, expected in publication['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    q = read('qualification.json')
    build = read('build-manifest.json')
    installed = read('build-manifest-installed.json')
    assert hashlib.sha256((ROOT / 'build-manifest-installed.json').read_bytes()).hexdigest() == q['build_manifest_sha256']
    assert all(installed[key] == build[key] for key in build)
    assert len(installed['cli_sha256']) == 64 and installed['cli_attestation']
    assessment = read('assessment.json')
    assert assessment['assessment'] == 'product_health_candidate_passed'
    assert assessment['source_commit'] == q['source_commit']
    assert assessment['wheel_sha256'] == build['wheel_sha256']
    assert assessment['publication']['published'] is False
    assert q['schema'] == 4 and q['status'] == 'passed'
    assert q['source_commit'] == build['source_commit'] and build['dirty'] is False
    assert q['host']['chip'] == 'Apple M4 Pro' and q['host']['model'] == 'Mac16,11'
    assert q['host']['memory_bytes'] == 64 * 1024**3
    assert q['runtime_overrides'] == []
    assert q['mtp_certification'] == {key: 'not_assessed' for key in ('MTP-S', 'MTP-P', 'MTP-D')}
    assert len(build['model_files']) == 22
    inventory = json.dumps(build['model_files'], sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(inventory).hexdigest() == q['contract']['model_manifest_sha256']
    cells = [q['default_route'], *q['cells']]
    assert [c['mode'] for c in cells] == ['ngram', 'direct', 'mtp']
    for cell in cells:
        mode = cell['mode']
        assert cell['status'] == 'ok' and cell['surface_passed'] is True
        assert cell['qa_items'] == cell['qa_hard_passed'] == 32
        report = read(f'qualification/report-{mode}-qwen3.8-27b.json')
        assert report['totals']['items'] == report['totals']['hard_passed'] == 32
        assert len(report['commit']) >= 8 and q['source_commit'].startswith(report['commit'])
        assert all(r['response']['finish_reason'] == 'stop' and not r['response']['error'] for r in report['results'])
        assert all(any(c['name'] == 'completion' and c['hard'] and c['passed'] for c in r['report']['checks']) for r in report['results'])
        surface = read(f'qualification/surface-{mode}-qwen3.8-27b.json')
        assert all(r['passed'] and not r['skipped'] for r in surface['results'] if r['hard'])
        route = read(f'qualification/server-route-{mode}-qwen3.8-27b.json')
        assert route['status'] == 'finished' and len(route['output_tokens']) == 64
        mtp = route['performance']['mtp']
        assert mtp['requested'] is (mode == 'mtp') and mtp['active'] is (mode == 'mtp')
        assert mtp['direct_fallback_steps'] == 0
        if mode == 'mtp':
            assert cell['mtp_draft_tokens'] > 0 and cell['mtp_verify_tokens'] > 0
        else:
            assert cell['mtp_draft_tokens'] == cell['mtp_verify_tokens'] == 0
    assert q['paired_greedy']['release_blocking'] is False
    before = {r['mode']: r for r in read('before-default.json')}
    assert before['default']['response']['performance']['mtp']['requested'] is True
    for row in read('before-after-controls.json'):
        assert row['same_input_sampling_budget'] is True
        assert row['before_tokens'] == row['after_tokens'] == 64
        assert not row['differences'], row['after_mode']
    lifecycle = read('lifecycle/results.json')
    assert {r['mode'] for r in lifecycle} == {'default', 'mtp'}
    for row in lifecycle:
        mode = row['mode']
        assert row['source_commit'] == q['source_commit']
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
                assert row[phase]['metrics']['ax_engine_mtp_accepted_tokens_total'] == 0
        command = row['full_qa_command']
        assert command[command.index('--max-tokens') + 1] == '1024'
        assert command[command.index('--streams') + 1] == 'both'
        assert '--all' in command
        report = read(f'lifecycle/{mode}-full-qa.json')
        assert report['totals']['items'] == report['totals']['hard_passed'] == 158
        assert len(report['commit']) >= 8 and q['source_commit'].startswith(report['commit'])
        assert all(r['response']['finish_reason'] == 'stop' and not r['response']['error'] for r in report['results'])
        assert all(any(c['name'] == 'completion' and c['hard'] and c['passed'] for c in r['report']['checks']) for r in report['results'])
        assert len({r['prompt_id'] for r in report['results']}) == 79
        assert report['bank_size'] == report['sample_size'] == 79
        for prompt in report['sampled_ids']:
            assert sorted(r['stream'] for r in report['results'] if r['prompt_id'] == prompt) == [False, True]
        soak = read(f'lifecycle/{mode}-soak.json')
        assert assessment['lifecycle']['buffered_event_peak'][mode] == soak['metric_peaks']['ax_engine_generation_buffered_stream_events']
        assert assessment['lifecycle']['backlog_overflow_delta'][mode] == soak['counter_deltas']['ax_engine_generation_stream_backlog_overflows_total']
        assert soak['schema_version'] == 'ax.native_generation_fault_soak.v2'
        assert q['source_commit'].startswith(soak['provenance']['repo_revision'])
        assert soak['provenance']['git_tracked_dirty'] is False
        assert soak['workload']['stalled_output_tokens'] == 8192
        assert soak['workload']['stalled_hold_ms'] == 180000
        assert all(r['output_tokens'] > 0 and r['receive_buffer_bytes'] > 0 for r in soak['outcomes'] if r['kind'] == 'stalled')
        assert soak['verdict'] == 'pass' and soak['quiescent'] is True
        assert soak['metrics_before']['ax_engine_mtp_direct_fallback_steps_total'] == 0
        assert soak['metrics_after']['ax_engine_mtp_direct_fallback_steps_total'] == 0
        assert soak['failure_reasons'] == [] and len(soak['outcomes']) == 15
        assert soak['workload']['rounds'] == 3 and soak['workload']['require_backpressure'] is True
        assert any(r['outcome'] == 'expected_disconnect' for r in soak['outcomes'])
        assert soak['metric_peaks']['ax_engine_generation_buffered_stream_events'] >= 1 or soak['counter_deltas']['ax_engine_generation_stream_backlog_overflows_total'] > 0
    audit = read('quality-contract-audit.json')
    assert audit['checker_contract'] == 'ax.qa.complete_answers.v2'
    assert audit['question_count'] == 79 and audit['prompts_golds_aliases_patterns_unchanged'] is True
    for arm in audit['arms']:
        original = read(f"original/{arm['mode']}-full-qa.json")
        assert original['totals'] == arm['old_totals']
        assert original['totals']['hard_passed'] == 154
        incomplete = [r for r in original['results'] if r['response']['finish_reason'] != 'stop']
        assert len(incomplete) == len(arm['incomplete'])
        assert arm['regraded_passed'] == (152 if arm['mode'] == 'default' else 154)
        old_soak = read(f"original/{arm['mode']}-soak.json")
        assert old_soak['verdict'] == 'fail'
        assert old_soak['failure_reasons'] == ['required backpressure was not observed in buffered-event or overflow metrics']
    packaging = read('packaging.json')
    assert packaging['native_import'] == 'passed'
    assert not any(p['name'].lower() == 'mlx' for p in packaging['installed_packages'])
    assert read('lifecycle/doctor-outside-source.json')['result'] == 'ready'
    assert all(r['exit_code'] == 0 for r in read('validation.json')['checks'])
    assert read('validation.json')['strict_clippy_exit'] == 101
    assert read('validation.json')['production_clippy_exit'] == 0
    harness = read('harness-source.json')
    assert harness['runtime_and_executed_harness_commit'] == q['source_commit']
    assert all(row['identical'] for row in harness['files'] if row['path'] not in ('qa/reporter.py', 'scripts/run_native_generation_fault_soak.py'))
    followups = read('validation-followups.json')
    assert followups['exit_code'] == 0 and followups['runtime_changes_since_candidate'] == ''
    print('Published product qualification receipts verified; no hosted release or MTP certification asserted.')

def verify_source(source_root):
    sys.path.insert(0, str(source_root / 'qa'))
    from checkers import run_response_checks
    from prompts import get_prompt_by_id
    audit = read('quality-contract-audit.json')
    assert hashlib.sha256((source_root / 'qa/question_bank.py').read_bytes()).hexdigest() == audit['revised_bank_sha256']
    for mode in ('default', 'mtp'):
        for row in read(f'lifecycle/{mode}-full-qa.json')['results']:
            response = row['response']
            result = run_response_checks(response['text'], get_prompt_by_id(row['prompt_id']), response['finish_reason'])
            assert result.auto_pass == row['hard_pass']
            assert [vars(check) for check in result.checks] == row['report']['checks']
        original = read(f'original/{mode}-full-qa.json')
        passed = sum(run_response_checks(row['response']['text'], get_prompt_by_id(row['prompt_id']), row['response']['finish_reason']).auto_pass for row in original['results'])
        assert passed == next(arm['regraded_passed'] for arm in audit['arms'] if arm['mode'] == mode)
    print('Current source reproduces all final check details and both historical regrades.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    verify()
    if args.source_root:
        verify_source(args.source_root.resolve())
