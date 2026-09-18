#!/usr/bin/env python3
"""Verify retained history diagnostics without a model or network."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '2026-09-18-qwen38-27b-first-split-replay'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    publication = read('publication.json')
    for name, digest in publication['files'].items():
        require(sha(ROOT / name) == digest, 'Artifact hash: ' + name)
    for name, digest in publication['previous_files'].items():
        require(sha(PREVIOUS / name) == digest, 'Prior artifact hash: ' + name)
    case = json.loads((PREVIOUS / 'case.json').read_text())
    require(len(case['prompt']) == 409 and len(case['baseline_direct_output']) == 192, 'Case extent')
    build = read('build.json')
    require(build['exit_code'] == 0 and sha(ROOT / 'probe.patch') == build['patch_sha256'], 'Probe build')
    observer = read('observer-build.json')
    require(observer['exit_code'] == 0 and sha(ROOT / 'observer.patch') == observer['patch_sha256'], 'Observer build')
    validation = read('validation.json')
    require(validation['source_before'] == validation['source_after'], 'Stable tested source')
    source = 'crates/ax-engine-mlx/src/bin/linear_mtp_state_oracle_probe.rs'
    require(validation['source_after'][source] == build['source_file_sha256'], 'Tested probe source')
    require(len(validation['checks']) == 11, 'Required checks')
    require([(r['check'], r['exit_code']) for r in validation['checks'] if r['exit_code']] == [('strict-clippy', 101)], 'Preserved strict Clippy failure')
    replays = read('replays.json')
    require(replays['hardware'] == {'machdep.cpu.brand_string': 'Apple M4 Pro', 'hw.memsize': '68719476736'}, 'SKU')
    require(len(replays['model_files']) == 22, 'Model file coverage')
    rows = {r['label']: r for r in replays['results']}
    require(len(rows) == len(replays['results']) == 7, 'Seven cold diagnostic arms')
    for name in ['pipeline-192', 'pipeline-192-repeat', 'synchronous-192']:
        row = rows[name]
        require(row['exit_code'] == 1 and not row['timed_out'], 'Expected full-prefix failure')
        require(row['expected_tokens'] == case['baseline_direct_output'], 'Actual direct reference')
        require(any('generated index 117: expected 40278, got 2849' in line for line in row['diagnostic_lines']), 'Mismatch boundary')
    for name, length in [('pipeline-one', 1), ('pipeline-two', 2), ('pipeline-reject', 118)]:
        row = rows[name]
        require(row['exit_code'] == 0 and not row['timed_out'], 'Accepted bounded prefix')
        reports = [json.loads(line.removeprefix('history_validation ')) for line in row['diagnostic_lines'] if line.startswith('history_validation ')]
        require(len(reports) == 1, 'Unique history result')
        report = reports[0]
        require(report['validated_tokens'] == row['expected_tokens'] and len(row['expected_tokens']) == length, 'Validated token coverage')
        require(report['cache_seq_len'] == 409 + length - 1 and not report['server_cache_identity_established'], 'Cache boundary and scope')
    negative = rows['pipeline-invalid-zero']
    require(negative['exit_code'] == 1 and not negative['timed_out'], 'Negative control')
    require(any('generated index 117: expected 0, got 2849' in line for line in negative['diagnostic_lines']), 'Invalid token rejected')
    live = read('live.json')
    require(live['full_192_fidelity'] and live['live_observer_valid'], 'Live admission')
    require(live['provenance']['model_files'] == replays['model_files'] and live['provenance']['native_files'] == replays['native_files'], 'Identical model and native files')
    require(live['provenance']['observer_build'] == observer, 'Live observer identity')
    require(live['loaded_native_basenames'] == {n: [n] for n in ['libmlx.dylib', 'libjaccl.dylib']}, 'Loaded native libraries')
    require([r['name'] for r in live['results']] == ['clean-direct', 'observer-direct', 'clean-direct-repeat'], 'Cold order')
    for row in live['results']:
        name = row['name']
        for suffix, key in [('request.json', 'request_body_sha256'), ('response.json', 'response_body_sha256')]:
            require(sha(ROOT / (name + '-' + suffix)) == row[key], 'Exact HTTP body')
        require(row['http_status'] == 200 and not row['watchdog_expired'] and row['cleanup']['returncode'] == 0, 'Completion and cleanup')
        require(read(name + '-request.json') == case['baseline_request'], 'Exact case request')
        response = read(name + '-response.json')
        require(response['output_tokens'] == case['baseline_direct_output'] and response['status'] == 'finished' and response['finish_reason'] == 'max_output_tokens', 'Full192 output fidelity')
        mtp = response['performance']['mtp']
        require(not mtp['active'] and mtp['direct_fallback_steps'] == 0, 'Direct route')
        expected = observer['binary_sha256'] if name == 'observer-direct' else live['provenance']['clean_server_sha256']
        require(row['binary_sha256'] == expected, 'Served executable')
    events = read('events.json')
    require(len(events) == 12 and not any(e['event'] == 'direct_invalid' for e in events), 'Observer events')
    window = next(e for e in events if e['event'] == 'direct_window')
    require(window['offset'] == 524 and window['input'] == 314 and window['generated_tokens'] == case['common_generated_prefix'][:115], 'Live boundary')
    require(all(window[k] is False for k in ['exact', 'target', 'relaxed', 'whole_trace']), 'Live direct scopes')
    arrays = {e['label']: e for e in events if e['event'] == 'array'}
    require(set(arrays) == {'live_direct', 'same_lazy_replay', 'host_singleton'}, 'Three actual logit arrays')
    hashes = {e['metadata']['f32_bits_le_sha256'] for e in arrays.values()}
    require(len(hashes) == 1 and all(e['argmax'] == 279 and e['metadata']['elements'] == 248320 for e in arrays.values()), 'Full logits and materialized argmax equality')
    comparisons = [e for e in events if e['event'] == 'logit_comparison']
    require(len(comparisons) == 2 and all(e['unequal'] == 0 and e['max_abs'] == 0 and e['elements'] == 248320 for e in comparisons), 'Full equality counts')
    controls = [e for e in events if e['event'] == 'control_cache']
    require(len(controls) == 2 and all(e['cache'] == window['cache_525'] for e in controls), 'Post-step state equality')
    valid = next(e for e in events if e['event'] == 'direct_valid')
    require(valid['cache_unchanged'] and not valid['production_return_replaced'] and valid['next_pending_token'] == 279, 'Non-mutating observer')
    old_events = [json.loads(line.split('first_split_probe ', 1)[1]) for line in (PREVIOUS / 'observer.log').read_text().splitlines() if 'first_split_probe ' in line]
    old = next(e for e in old_events if e['event'] == 'cache_before')['snapshot']
    current = window['cache_522']
    require({k:v for k,v in current.items() if k != 'logical_arrays'} == {k:v for k,v in old.items() if k != 'logical_arrays'}, 'Aligned cache boundary metadata')
    a = {(x['layer'], x['kind']): x['array'] for x in current['logical_arrays']}
    b = {(x['layer'], x['kind']): x['array'] for x in old['logical_arrays']}
    require(len(a) == len(b) == 128 and a.keys() == b.keys(), 'Full logical cache coverage')
    require(all({k:v for k,v in a[x].items() if k != 'f32_bits_le_sha256'} == {k:v for k,v in b[x].items() if k != 'f32_bits_le_sha256'} for x in a), 'Matching shape/dtype metadata')
    different = [{'layer': x[0], 'kind': x[1]} for x in a if a[x]['f32_bits_le_sha256'] != b[x]['f32_bits_le_sha256']]
    analysis = read('analysis.json')
    require(len(different) == 128 and different == analysis['cache522_unequal_arrays'], 'Observed unequal histories')
    require(analysis['historical_mtp_log_sha256'] == sha(PREVIOUS / 'observer.log'), 'Prior witness identity')
    print('Evidence verified; expected standalone mismatches retained; release gates remain open.')


if __name__ == '__main__':
    main()
