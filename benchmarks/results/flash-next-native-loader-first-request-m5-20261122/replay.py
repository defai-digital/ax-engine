"""Audit the M5 first-request and warmed weight reader comparison."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
from statistics import median
import tarfile


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'flash-next-native-loader-m5-20261121' / 'result.json'
VOCAB = 248320
PROMPTS = (512, 2048, 8192)
ARMS = {'name-short': ('name-order', 512), 'lazy-short': ('lazy-cpu', 512),
        'lazy-long': ('lazy-cpu', 8192), 'name-long': ('name-order', 8192)}


def workload(first):
    other = 8192 if first == 512 else 512
    return [(first, 0, 'first'), (512, 100, 'warmup'),
            *((length, run, 'warm') for run in (1, 2, 3) for length in PROMPTS),
            (other, 0, 'parity'), (2048, 0, 'parity'), (32736, 0, 'boundary')]


def prompt_digest(length, run):
    rng = random.Random(f'flash-next-fixed-workload:{length}:{run}')
    ids = [rng.randrange(VOCAB) for _ in range(length)]
    return sha256(json.dumps(ids).encode()).hexdigest()


def measure(rows):
    ratios = []
    for row in rows:
        timing = row['timing']
        ttft = timing['ttft_seconds']
        decode = timing['decode_seconds']
        if not all(value > 0 and isfinite(value) for value in (ttft, decode)) or timing['decode_tokens'] != 31:
            raise ValueError('Invalid client timing in the loader control')
        ratios.append((ttft, 31 / decode))
    return {'ttft_seconds': median(row[0] for row in ratios),
            'decode_tokens_per_second': median(row[1] for row in ratios)}


def body(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f'Missing receipt data {name}')
    return stream.read()


def audit(archive, prior):
    raw = body(archive, 'receipt.json')
    receipt = json.loads(raw)
    build = json.loads(body(archive, 'server-build-receipt.json'))
    assert receipt['build'] == build == prior['build']
    input_hashes = {name: digest for digest, name in (line.split() for line in
                    body(archive, 'inputs.sha256').decode().splitlines())}
    assert input_hashes['run_loader_followup.py'] == receipt['runner_sha256']
    selection = body(archive, 'resident-page-selection.json')
    assert sha256(selection).hexdigest() == receipt['resident_page_selection_sha256']
    assert sha256(body(archive, 'server-build-receipt.json')).hexdigest() == input_hashes['server-build-receipt.json']
    for key in ('model_manifest_sha256', 'inventory_sha256', 'footprint_probe_sha256',
                'footprint_probe_source_sha256', 'resident_page_selection_sha256', 'libmlx_sha256'):
        assert receipt[key] == prior[key]
    assert receipt['pack_revision'] == prior['pack_revision'] == 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
    assert receipt['launch'] == 'ordinary-32k-resident-single-request'
    assert list(receipt['arms']) == list(ARMS)
    assert receipt['status'] in ('complete', 'failed')
    assert len(receipt['workloads']) == 4
    observed = {}
    results = {}
    sources = json.loads(body(archive, 'sources.json'))
    for label, (mode, first) in ARMS.items():
        arm = receipt['arms'][label]
        ready = arm['ready_seconds']
        assert ready > 0 and isfinite(ready)
        assert arm['loader_mode'] == mode and (arm['shard_count'], arm['tensor_count']) == (15, 1942)
        assert arm['model'] == {'id': 'qwen3.8-flash-next', 'context_length': 32768}
        assert arm['profile'] == {'effective_sequence_cap': 1, 'expert_residency': 'resident'}
        assert arm['health_ready'] == arm['health_after_requests'] == 'ok'
        assert arm['context_overflow'] == {'http_status': 400, 'error_code': 'context_length_exceeded'}
        assert all(item['resident_pages'] == 1 for item in arm['initial_selected_page_samples'])
        assert len(arm['initial_selected_page_samples']) == 3
        log = body(archive, f'{label}-server.txt')
        assert sha256(log).hexdigest() == sources['server_logs'][label]['portable_sha256']
        assert b'decision=resident' in log and b'context_tokens=32768' in log
        assert f'loader_mode="{mode}" shard_count=15 tensor_count=1942'.encode() in log
        assert receipt['workloads'][label] == [list(item) for item in workload(first)]
        rows = arm['cases']
        assert [(r['prompt_tokens'], r['trial'], r['phase']) for r in rows] == workload(first)
        for row in rows:
            digest = prompt_digest(row['prompt_tokens'], row['trial'])
            assert digest == row['prompt_hash']
            ids = row['output_tokens']
            assert len(ids) == 32 and observed.setdefault(digest, ids) == ids
            assert row['verified_steps'] == row['route_errors'] == 0 and row['verifier_schedule'] == 1
            assert row['status'] == 'finished' and row['finish_reason'] == 'max_output_tokens'
        samples = [json.loads(line) for line in body(archive, f'{label}-samples.jsonl').splitlines()]
        assert samples and sorted({s['pressure'] for s in samples}) == arm['pressure_levels']
        assert max(s['reference_rss_bytes'] for s in samples if 'reference_rss_bytes' in s) == arm['peak_sampled_rss_bytes']
        for name in ('physical_footprint_bytes', 'kernel_lifetime_peak_footprint_bytes'):
            values = [s[name] for s in (*samples, arm['process_before_stop']) if name in s]
            assert values and max(values) == arm[f'max_observed_{name}']
        pre = arm['quiet_baseline']['cpu_idle_percent']
        post = arm['post_run_cpu_idle_percent']
        first_prompt = measure(rows[:1])
        results[label] = {
            'readiness_seconds': ready, 'resident_load_seconds': arm['load_phase_seconds'],
            'first_prompt': first, 'first_request': first_prompt,
            'ready_plus_first_ttft_seconds': ready + first_prompt['ttft_seconds'],
            'warm': {length: measure([row for row in rows if row['phase'] == 'warm' and row['prompt_tokens'] == length])
                     for length in PROMPTS},
            'pressure_samples': len(samples), 'post_run_cpu_idle_percent': post,
            'measurement_eligible': 90 <= pre <= 100 and 90 <= post <= 100 and arm['pressure_levels'] == [1],
        }
    pairs = {}
    for suffix in ('short', 'long'):
        control = results[f'name-{suffix}']
        candidate = results[f'lazy-{suffix}']
        pairs[suffix] = {length: {metric: candidate['warm'][length][metric] / value
                                  for metric, value in control['warm'][length].items()} for length in PROMPTS}
    eligible = all(arm['measurement_eligible'] for arm in results.values())
    assert (receipt['status'] == 'complete') == eligible
    assert body(archive, 'followup-status.txt') == (b'loader-followup 0\n' if eligible else b'loader-followup 1\n')
    warm_band = all(row['ttft_seconds'] <= 1.03 and row['decode_tokens_per_second'] >= 0.97
                    for pair in pairs.values() for row in pair.values())
    return {
        'scope': 'First-request and warmed native loader diagnostic',
        'receipt_sha256': sha256(raw).hexdigest(), 'terminal_status': receipt['status'],
        'matched_ids_across_arms': 32 * sum(len(arm['cases']) for arm in receipt['arms'].values()),
        'all_measurements_eligible': eligible, 'all_warmed_bands_met': warm_band,
        'eligible_for_runtime_review': eligible and warm_band,
        'arms': results, 'warm_candidate_to_control': pairs,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence.tar.xz')
    parser.add_argument('--prior', type=Path, default=PRIOR)
    args = parser.parse_args()
    with tarfile.open(args.archive) as archive:
        result = audit(archive, json.loads(args.prior.read_bytes()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
