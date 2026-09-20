"""Recompute matched-workload results and validate retained raw evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
EXTENSION = '4b4d26a19cbdd9e475f777e491c28869a421a897a4928fb2183583230de2bcb6'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def verify_row(row, budget):
    assert len(row['output_tokens']) == budget
    events = row['emissions']
    assert events and sum(n for _, n in events) == budget
    assert 0 < events[0][0] < events[-1][0]
    assert all(n > 0 for _, n in events)
    assert all(a[0] < b[0] for a, b in zip(events, events[1:]))
    first, first_n = events[0]
    last = events[-1][0]
    expected = dict(ttft_s=first, completion_s=last, completion_tok_s=budget / last,
                    first_batch_tokens=first_n, decode_tokens=budget - first_n,
                    decode_s=last - first, decode_tok_s=(budget - first_n) / (last - first))
    for key, value in expected.items():
        assert math.isfinite(row[key]) and math.isclose(row[key], value, rel_tol=1e-9, abs_tol=1e-9), key
    assert row['api_return_s'] >= last


def summarize(runs):
    cells = {}
    for run in runs:
        for case in run['cases']:
            key = (run['model'], case['case'], run['arm'])
            cells.setdefault(key, []).extend(case['trials'])
    return {key: dict(completion_tok_s=median(r['completion_tok_s'] for r in rows),
                      ttft_ms=1000 * median(r['ttft_s'] for r in rows),
                      decode_tok_s=median(r['decode_tok_s'] for r in rows),
                      min_completion_tok_s=min(r['completion_tok_s'] for r in rows),
                      max_completion_tok_s=max(r['completion_tok_s'] for r in rows),
                      output_variants=len({tuple(r['output_tokens']) for r in rows}),
                      samples=len(rows)) for key, rows in cells.items()}


def verify(data, check_source=False):
    assert digest((HERE / 'bench_native_peer.py').read_bytes()) == data['harness_sha256']
    if check_source:
        for name, expected in data['source_sha256'].items():
            assert digest((HERE.parents[4] / name).read_bytes()) == expected, name
    assert data['benchmark_build']['sha256'] == EXTENSION
    assert not data['test_only_rebuild']['different']
    assert data['test_only_rebuild']['before'] == data['test_only_rebuild']['after']
    probe = data['python_mode_probe']
    assert probe['required_off_rejected']
    assert probe['rows'][0]['active_bytes'] > 2 * probe['rows'][1]['active_bytes']
    assert abs(probe['rows'][0]['active_bytes'] - probe['rows'][2]['active_bytes']) < 128 * 2**20
    for model, cases in data['workloads'].items():
        assert len(cases) == 3
        for case in cases:
            assert digest(json.dumps(case['token_ids'], separators=(',', ':')).encode()) == case['sha256']
    count = warmups = 0
    assert set(data['hosts']) == {'m5', 'm4', 'm2', 'm3'}
    for host, entry in data['hosts'].items():
        assert entry['runtime_assets'] == data['hosts']['m5']['runtime_assets']
        assert entry['runtime_assets']['_ax_engine.abi3.so'] == EXTENSION
        assert len(entry['preflight']) == len(entry['postflight']) == 2
        for pre, post in zip(entry['preflight'], entry['postflight']):
            assert pre['model'] == post['model'] and pre['files'] == post['files']
            assert all(f['sha256'] == f['expected'] for f in post['files'])
            assert not pre['detected_delta'] and not post['detected_delta']
            assert all(n['unchanged'] for n in pre['norms'] + post['norms'])
        arms = ('ax', 'sustained', 'turbo') if host == 'm5' else ('ax', 'sustained', 'turbo', 'ax-unwired')
        expected = {(b, m, a) for b in (0, 1) for m in ('tiel', 'cyber') for a in arms}
        runs = entry['runs']
        assert len(runs) == len(expected)
        assert {(r['block'], r['model'], r['arm']) for r in runs} == expected
        assert all(a['ended'] <= b['started'] for a, b in zip(runs, runs[1:]))
        for model in ('tiel', 'cyber'):
            forward = [r['arm'] for r in runs if r['model'] == model and r['block'] == 0]
            reverse = [r['arm'] for r in runs if r['model'] == model and r['block'] == 1]
            assert forward == reverse[::-1]
        for run in runs:
            assert run['complete'] and run['exit_code'] == 0
            assert run['greedy'] and not run['prefix_cache']
            assert run['depth_cap'] == 3 and run['conservative'] == 0
            assert run['stream_experts'] == 'off'
            assert run['warmups'] == 2 and run['repetitions'] == 3 and run['cooldown_s'] == 3
            assert run['mlx_version'] == '0.32.2'
            assert run['python_version'].split()[0] == entry['python_version']
            assert run['engine'] == ('ax' if run['arm'].startswith('ax') else 'mtplx')
            assert run['profile'] == ('throughput' if run['engine'] == 'ax' else run['arm'])
            for state in run['conditions'].values():
                assert state['chip'] == {'m5': 'Apple M5 Max', 'm4': 'Apple M4 Pro', 'm2': 'Apple M2 Ultra', 'm3': 'Apple M3 Ultra'}[host]
                assert int(state['memory_bytes']) == {'m5': 128, 'm4': 64, 'm2': 192, 'm3': 512}[host] * 2**30
            if run['engine'] == 'ax':
                assert run['extension_sha256'] == EXTENSION
                assert run['environment'].get('AX_MLX_WIRED_LIMIT_SCALE') == ('0' if run['arm'] == 'ax-unwired' else None)
                assert run['environment'].get('AX_STREAM_EXPERTS') == (None if host == 'm5' else 'off')
            assert len(run['cases']) == 3
            for case, workload in zip(run['cases'], data['workloads'][run['model']]):
                assert case['prompt_sha256'] == workload['sha256']
                assert case['prompt_tokens'] == len(workload['token_ids'])
                assert case['generation_tokens'] == workload['generation_tokens']
                assert len(case['trials']) == 3 and len(case['warmup_trials']) == 2
                for measured, rows in ((True, case['trials']), (False, case['warmup_trials'])):
                    for row in rows:
                        assert row['measured'] == measured
                        verify_row(row, case['generation_tokens'])
                        if run['engine'] == 'ax':
                            c = row['counters']
                            assert row['backend'] == 'mlx'
                            assert c['ax_mtp_requested'] == c['ax_mtp_correctness_mode'] == 1
                            assert c['ax_mtp_draft_tokens'] > 0
                            assert c['ax_mtp_drafted_depth2'] > 0
                            assert c['ax_mtp_direct_fallback_steps'] == c['ax_mtp_optimistic_steps'] == 0
                            assert c['ax_mtp_ngram_proposed_tokens'] == 0
                            assert c['core_prefix_reuse_disabled'] == c['ax_mlx_prefix_cache_blocked'] == 1
                        else:
                            stats = row['native']
                            assert stats['drafted_tokens'] > 0 and stats['speculative_depth'] == 3
                            assert stats['cached_tokens'] == 0 and not stats['session_cache_hit']
                            assert stats['context_copy_drafted_tokens'] == 0
                            assert stats['drafted_by_depth'][2] > 0
                count += 3
                warmups += 2
        print(host)
        for key, values in summarize(runs).items():
            print(' / '.join(key), json.dumps(values, sort_keys=True))
    diagnostic = data['m4_auto_diagnostic']
    assert len(diagnostic) == 6
    assert {(r['model'], r['arm']) for r in diagnostic} == {
        (m, a) for m in ('tiel', 'cyber') for a in ('ax-auto', 'ax-off', 'sustained')}
    assert all(a['ended'] <= b['started'] for a, b in zip(diagnostic, diagnostic[1:]))
    for run in diagnostic:
        assert run['complete'] and run['exit_code'] == 0
        assert run['repetitions'] == run['warmups'] == 1
        assert run['stream_experts'] == ('auto' if run['arm'] == 'ax-auto' else 'off')
        assert run['greedy'] and not run['prefix_cache'] and run['conservative'] == 0
        assert run['mlx_version'] == '0.32.2' and run['cooldown_s'] == 3
        if run['engine'] == 'ax':
            assert run['extension_sha256'] == EXTENSION
        assert len(run['cases']) == 1
        case = run['cases'][0]
        assert case['case'] == 'python-lru' and case['generation_tokens'] == 16
        assert case['prompt_sha256'] == data['workloads'][run['model']][1]['sha256']
        assert len(case['trials']) == len(case['warmup_trials']) == 1
        for row in case['trials'] + case['warmup_trials']:
            verify_row(row, 16)
            if run['engine'] == 'ax':
                assert row['backend'] == 'mlx'
                assert row['counters']['ax_mtp_requested'] == 1
                assert row['counters']['ax_mtp_draft_tokens'] > 0
            else:
                assert row['native']['drafted_tokens'] > 0
    assert count == 540 and warmups == 360
    print('PASS: separate M4 Auto diagnostic, 6 measured requests + 6 warmups.')
    print(f'PASS: {count} measured requests + {warmups} warmups; matched inputs, output counts, timing boundaries, MTP and cold KV.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-source', action='store_true')
    verify(json.loads((HERE / 'trials.json').read_text()), parser.parse_args().check_source)
