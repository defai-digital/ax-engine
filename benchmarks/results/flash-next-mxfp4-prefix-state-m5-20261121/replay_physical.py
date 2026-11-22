"""Replay the follow-up M5 HTTP physical-cut diagnostic."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
CUT = [512, 512, 512, 511, 1]
LOGICAL_PROGRESS = [(2048, 2048, 0), (2112, 64, 1)]
STARTUP = {'cold': [], 'l1': [31, 1] * 6}
READY = re.compile(
    r'^\S+\s+INFO ax-engine-server preview listening bind_address=\S+ '
    r'grpc_bind_address=None model_id=qwen3\.8-flash-next:axq support_tier=MlxPreview$'
)
PHYSICAL = re.compile(r'^AX_PREFILL_TIME_DEBUG chunk_tokens=(\d+) ')
REUSED = 'ax_mlx_prefix_cache_reused_tokens'


def extract_log(raw):
    lines = raw.decode().splitlines()
    ready = [line_no for line_no, line in enumerate(lines, 1) if READY.fullmatch(line)]
    forwards = []
    for line_no, line in enumerate(lines, 1):
        if line.startswith('AX_PREFILL_TIME_DEBUG chunk_tokens='):
            match = PHYSICAL.match(line)
            assert match, f'Malformed physical event at line {line_no}'
            forwards.append([line_no, int(match[1])])
    return {'source_sha256': sha256(raw).hexdigest(), 'line_count': len(lines),
            'ready_lines': ready, 'forwards': forwards}


def check_trace(arm, trace, count):
    assert len(trace['ready_lines']) == 1, f'{arm}: expected one structured listener boundary'
    ready_line, = trace['ready_lines']
    assert 1 <= ready_line <= trace['line_count']
    previous = 0
    for line_no, _ in trace['forwards']:
        assert previous < line_no <= trace['line_count'] and line_no != ready_line
        previous = line_no
    startup = [width for line_no, width in trace['forwards'] if line_no < ready_line]
    measured = [width for line_no, width in trace['forwards'] if line_no > ready_line]
    assert startup == STARTUP[arm], f'{arm}: unexpected startup widths {startup}'
    assert measured == CUT * count, f'{arm}: unexpected HTTP forwards {measured}'
    return {'startup_widths': startup, 'structured_listen_line': ready_line,
            'measured_cuts': count, 'measured_forwards': len(measured)}


def progress(reply):
    return [(row['cursor'], row['advanced_prompt_tokens'], row['delta_token_count'])
            for row in reply['prefill_progress']]


def reused(reply):
    return max((event['decisions'].get(REUSED, 0) for event in reply['cache_events']), default=0)


def check_http(current, prior):
    for key in ('binary_sha256', 'pack_revision', 'inventory_sha256', 'manifest_sha256', 'prompts'):
        assert current[key] == prior[key], key
    assert current['status'] == prior['status'] == 'complete'
    assert current['exact_prompt_reuses_and_matches'] is True
    assert current['strict_extensions_cold_prefill_and_match'] is True
    assert current['warm_overlap_proven'] is False

    cold = current['arms']['cold']['outputs']
    assert cold.keys() == prior['arms']['cold']['outputs'].keys()
    for label, reply in cold.items():
        assert reply['output_tokens'] == prior['arms']['cold']['outputs'][label]['output_tokens']
        assert reused(reply) == 0

    l1 = current['arms']['l1']
    assert l1['both_seed_stores_reported'] is True
    assert reused(l1['exact-replay']) == 2048
    assert l1['exact-replay']['output_tokens'] == cold['exact-replay']['output_tokens']
    assert l1['replacement_tokens_match'] is True
    assert l1['replacement_l1_restore_2048_proven'] is False
    cases = {'target': l1['paired']['target'], 'cancelled': l1['disconnect']['target'],
             'replacement': l1['replacement']}
    observed_counts = {}
    for name, reply in cases.items():
        assert progress(reply) == progress(cold[name]) == LOGICAL_PROGRESS
        assert reused(reply) == 0
        if name == 'cancelled':
            observed = reply['first_output_tokens']
            assert len(observed) == 1
            expected = cold[name]['output_tokens'][:1]
        else:
            observed = reply['output_tokens']
            expected = cold[name]['output_tokens']
        assert observed and observed == expected, name
        observed_counts[name] = len(observed)

    for arm, peer in (('paired', 'survivor-a'), ('disconnect', 'survivor-b')):
        pair = l1[arm]
        checks = pair['checks']
        assert checks['l1_restore_2048_proven'] is False
        assert checks['target_tokens_match'] is True and checks['peer_tokens_match'] is True
        assert checks['peer_no_restored_tokens'] is True
        assert reused(pair['peer']) == 0
        assert pair['peer']['output_tokens'] == cold[peer]['output_tokens']
        steps = pair['target']['prefill_progress']
        assert [step['step_id'] for step in steps] == checks['shared_prefill_steps_with_ongoing_decode']
        assert all(step['scheduled_requests'] == 2 for step in steps)
    assert cold['cancelled']['output_tokens'][0] != l1['seed-b']['output_tokens'][0]
    return observed_counts


def replay(root=ROOT):
    raw = gzip.decompress((root / 'physical-http-control.json.gz').read_bytes())
    baseline = gzip.decompress((root / 'http-control.json.gz').read_bytes())
    current = json.loads(raw)
    output_counts = check_http(current, json.loads(baseline))
    evidence = json.loads((root / 'physical-traces.json').read_bytes())
    arms = {}
    for arm, trace in evidence.items():
        arms[arm] = check_trace(arm, trace, len(output_counts))
        arms[arm]['log_sha256'] = trace['source_sha256']
        samples = [json.loads(line) for line in gzip.decompress(
            (root / f'physical-{arm}-samples.jsonl.gz').read_bytes()).splitlines()]
        assert samples and all(sample['pressure'] == 1 for sample in samples)
        peak = max(sample['footprint_bytes'] for sample in samples)
        assert peak == current['arms'][arm]['peak_footprint_bytes']
        arms[arm].update(normal_pressure_samples=len(samples), peak_footprint_bytes=peak)
    assert arms.keys() == STARTUP.keys()
    return {
        'scope': 'Observed cold physical forwards in the strict-extension HTTP controls',
        'receipt_sha256': sha256(raw).hexdigest(), 'baseline_sha256': sha256(baseline).hexdigest(),
        'physical_widths_per_cut': CUT, 'logical_widths': [2048, 64],
        'exact_prompt_reused_tokens': 2048, 'l1_strict_extension_output_ids': output_counts,
        'strict_extension_serving_warm_reuse_proven': False, 'release_ready': False, 'arms': arms,
    }


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
