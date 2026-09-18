"""Verify the frozen prefix-prefill diagnosis; this does not qualify a release."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT, check_publication=True):
    if check_publication:
        for name, digest in load(root / 'publication.json')['files'].items():
            assert sha(root / name) == digest, name
    analysis = load(root / 'analysis.json')
    assert analysis['release_ready'] is False
    assert analysis['production_arithmetic_changed'] is False
    case = load(root.parent / '2026-09-18-qwen38-27b-first-split-replay/case.json')
    expected = case['baseline_direct_output']
    assert len(case['prompt']) == 409 and len(expected) == 192
    off = load(root / 'cache-off/cache-off-direct-response.json')['output_tokens']
    assert len(off) == 192 and off[:117] == expected[:117]
    assert (expected[117], off[117]) == (40278, 2849)
    rows = load(root / 'live-controls.json')
    assert len(rows) == 5
    for row in rows:
        folder, name = row['folder'], row['name']
        for kind in ['request', 'response']:
            assert sha(root / folder / f'{name}-{kind}.json') == row[kind + '_body_sha256']
        request = load(root / folder / f'{name}-request.json')
        response = load(root / folder / f'{name}-response.json')
        assert request == case['baseline_request']
        assert response['output_tokens'] == (off if name == 'cache-off-direct' else expected)
        assert response['prompt_tokens'] == case['prompt']
        assert row['http_status'] == 200 and not row['watchdog_expired']
        assert row['cleanup']['returncode'] == 0
        mtp = response['performance']['mtp']
        assert not mtp['active'] and mtp['direct_fallback_steps'] == 0
        assert row['environment'] == ({'AX_MLX_PREFIX_CACHE_MAX_BYTES': '0'} if name == 'cache-off-direct' else {})
    states = {name: load(root / 'states' / f'{name}.json') for name in ['server', 'unsplit']}
    boundaries = [409, 410, 411, 522, 524, 525, 526]
    for events in states.values():
        assert [e['value']['seq_len'] for e in events] == boundaries
        for event in events:
            assert event['event'] == 'cache'
            arrays = event['value']['logical_arrays']
            assert len(arrays) == 128 and len({(x['layer'], x['kind']) for x in arrays}) == 128
            assert Counter(x['kind'] for x in arrays) == {'conv': 48, 'recurrent': 48, 'key': 16, 'value': 16}
            for item in arrays:
                a = item['array']
                assert a['finite'] and a['elements'] == math.prod(a['shape'])
                assert len(a['f32_bits_le_sha256']) == 64
    for index, (left, right) in enumerate(zip(states['server'], states['unsplit'])):
        differences = sum(a != b for a, b in zip(left['value']['logical_arrays'], right['value']['logical_arrays']))
        assert differences == (128 if index < 3 else 127)
    summary = load(root / 'live/summary.json')
    assert summary['full_192_fidelity'] and summary['seven_cache_boundaries'] and summary['prior_cache522_524_525_equal']
    assert hashlib.sha256((json.dumps(states['server'], indent=2) + '\n').encode()).hexdigest() == summary['events_sha256']
    prior = load(root.parent / '2026-09-18-qwen38-27b-direct-history/events.json')[0]
    for boundary in [522, 524, 525]:
        assert next(e['value'] for e in states['server'] if e['value']['seq_len'] == boundary) == prior[f'cache_{boundary}']
    observations = load(root / 'observations.json')
    assert len(observations) == 10
    for row in observations:
        assert not row['timed_out']
        split = row['label'] in ['prefix16', 'runner-prefix16', 'prefix16-repeat']
        off_reference = row['label'] == 'baseline-off-reference'
        assert row['state_set'] == ('server' if split else 'unsplit')
        assert hashlib.sha256(json.dumps(states[row['state_set']], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == row['canonical_state_sha256']
        assert row['expected_tokens'] == (off if off_reference else expected)
        assert row['exit_code'] == (0 if split or off_reference else 1)
        lines = row['diagnostic_lines']
        if row['exit_code'] == 0:
            result = json.loads(next(line.removeprefix('history_validation ') for line in lines if line.startswith('history_validation ')))
            assert result['validated_tokens'] == row['expected_tokens']
        else:
            assert any('generated index 117: expected 40278, got 2849' in line for line in lines)
        assert len(row['loaded_mlx_paths']) == 2
        assert all('/forced-replay-venv/' in p for p in row['loaded_mlx_paths'])
    final = load(root / 'final-cli/results.json')
    assert len(final) == 5
    for row in final:
        log = root / 'final-cli' / (row['label'] + '.log')
        assert sha(log) == row['published_log_sha256'] and not row['timed_out']
        text = log.read_text()
        split = row['label'] in ['prefix16', 'prefix16-repeat']
        off_reference = row['label'] == 'baseline-off-reference'
        assert row['exit_code'] == (0 if split or off_reference else 1)
        assert list(map(int, row['command'][2].split(','))) == case['prompt']
        assert list(map(int, row['command'][3].split(','))) == row['expected_tokens']
        assert row['expected_tokens'] == (off if off_reference else expected)
        if row['exit_code'] == 0:
            result = json.loads(next(line.removeprefix('history_validation ') for line in text.splitlines() if line.startswith('history_validation ')))
            assert result['validated_tokens'] == row['expected_tokens']
            assert result['prefix_cache_block_size'] == (16 if split else None)
            assert result['prefill_capture_head'] == (400 if split else None)
            assert result['server_cache_identity_established'] is False
            assert result['cache_seq_len'] == 600
        elif row['label'] == 'invalid-block':
            assert 'positive integer' in text
        else:
            assert 'generated index 117: expected 40278, got 2849' in text
    identity = load(root / 'controls/receipt.json')
    assert identity['hardware'] == {'machdep.cpu.brand_string': 'Apple M4 Pro', 'hw.memsize': '68719476736'}
    assert len(identity['model_files']) == 22
    for name in ['prefix-controls/receipt.json', 'final-cli/receipt.json', 'live/provenance.json', 'cache-off/provenance.json']:
        receipt = load(root / name)
        for field in ['model_files', 'native_files', 'case_sha256']:
            assert receipt[field] == identity[field]
    for name, patch in [('final-build.json', 'final.patch'), ('probe-build.json', 'observation.patch'), ('server-build.json', 'observation.patch'), ('prefix-control-build.json', 'prefix-observation.patch')]:
        build = load(root / name)
        assert build['exit_code'] == 0 and build['source_base'] == analysis['source_base']
        assert build['patch_sha256'] == sha(root / patch)
    assert load(root / 'final-cli/receipt.json')['build'] == load(root / 'final-build.json')
    assert load(root / 'prefix-controls/receipt.json')['build'] == load(root / 'prefix-control-build.json')
    assert load(root / 'live/provenance.json')['observer_build'] == load(root / 'server-build.json')
    validation = load(root / 'validation.json')
    assert validation['source_before'] == validation['source_after']
    for path, digest in load(root / 'final-build.json')['source_files'].items():
        assert validation['source_after'][path] == digest
    assert len(validation['results']) == 11
    for row in validation['results']:
        assert row['exit_code'] == (101 if row['check'] == 'strict-clippy' else 0)
        assert sha(root / 'validation' / (row['check'] + '.log')) == row['published_log_sha256']
    assert '7 passed; 0 failed' in (root / 'validation/history-regression.log').read_text()
    assert 'array is not row-contiguous' in (root / 'validation/strided-before.log').read_text()
    assert load(root / 'initial-observer-failure.json')['admitted'] is False
    assert all(row['exit_code'] == 0 and not row['timed_out'] for row in load(root / 'review-receipts.json'))
    return {'verified': True, 'release_ready': False, 'live_controls': 5, 'observations': 10, 'final_cli_controls': 5}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
