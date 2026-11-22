"""Audit the same-binary M5 guarded mmap read-ahead diagnostic."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
import re
from statistics import median
import tarfile


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'flash-next-mxfp4-startup-phases-m5-20261121' / 'evidence.tar.xz'
BUILD_COMMIT = 'c2358dd814f5bb4a11fce0f22d9d79c2809f63ea'
VOCAB = 248320
PROMPTS = (512, 2048, 8192)
ARMS = {'mmap-short': ('mmap', 512), 'advised-short': ('advised', 512),
        'advised-long': ('advised', 8192), 'mmap-long': ('mmap', 8192)}



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


def audit(archive, expected_selection):
    raw = body(archive, 'receipt.json')
    receipt = json.loads(raw)
    build = json.loads(body(archive, 'server-build-receipt.json'))
    assert receipt['build'] == build and build['commit'] == BUILD_COMMIT
    assert build['profile'] == 'release-server' and build['mlx'] == build['mlx_metal'] == '0.32.4'
    input_hashes = {name: digest for digest, name in (line.split() for line in
                    body(archive, 'inputs.sha256').decode().splitlines())}
    assert input_hashes['run_read_ahead.py'] == receipt['runner_sha256']
    selection = body(archive, 'expected-resident-selection.json')
    assert sha256(selection).hexdigest() == build['m5_selection_sha256']
    assert json.loads(selection) == expected_selection
    assert sha256(body(archive, 'server-build-receipt.json')).hexdigest() == input_hashes['server-build-receipt.json']
    assert input_hashes['ax-engine-server-read-ahead'] == build['server_sha256']
    assert receipt['inventory_sha256'] == input_hashes['mxfp4-inventory.json'] == '97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6'
    assert receipt['pack_revision'] == 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
    assert receipt['scope'] == 'guarded mmap advisory startup diagnostic'
    assert receipt['launch'] == 'ordinary-32k-resident-single-request'
    assert tuple(receipt['arms']) == tuple(ARMS)
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
        hints = trace_hints(log.decode(), expected_selection, mode, arm['load_phase_seconds'])
        assert hints == arm['read_ahead']
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
            'pressure_samples': len(samples), 'post_run_cpu_idle_percent': post, 'hints': hints,
            'measurement_eligible': 90 <= pre <= 100 and 90 <= post <= 100 and arm['pressure_levels'] == [1],
        }
    pairs = {}
    for suffix in ('short', 'long'):
        control = results[f'mmap-{suffix}']
        candidate = results[f'advised-{suffix}']
        pairs[suffix] = {length: {metric: candidate['warm'][length][metric] / value
                                  for metric, value in control['warm'][length].items()} for length in PROMPTS}
    eligible = all(arm['measurement_eligible'] for arm in results.values())
    assert (receipt['status'] == 'complete') == eligible
    assert body(archive, 'phase-status.txt') == f'read-ahead {0 if eligible else 1}\n'.encode()
    warm_band = all(row['ttft_seconds'] <= 1.03 and row['decode_tokens_per_second'] >= 0.97
                    for pair in pairs.values() for row in pair.values())
    return {
        'scope': 'Advisory read-ahead with guarded copied MLX storage',
        'receipt_sha256': sha256(raw).hexdigest(), 'terminal_status': receipt['status'],
        'matched_ids_across_arms': 32 * sum(len(arm['cases']) for arm in receipt['arms'].values()),
        'all_measurements_eligible': eligible, 'all_warmed_bands_met': warm_band,
        'eligible_for_runtime_review': eligible and warm_band,
        'arms': results, 'warm_candidate_to_control': pairs,
    }


def trace_hints(log, expected_selection, mode, resident_seconds):
    shards = [[name, int(count)] for name, count in re.findall(
        r'Flash Next resident shard shard=Some\("([^"/]+)"\) tensor_count=(\d+)', log)]
    assert shards == expected_selection
    phase = re.findall(r'Flash Next startup phase phase="resident_tensors" elapsed_us=(\d+)', log)
    assert len(phase) == 1 and int(phase[0]) / 1e6 == resident_seconds
    matches = re.findall(r'Flash Next resident read-ahead lab attempts=(\d+) accepted=(\d+) accepted_bytes=(\d+)', log)
    if mode == 'mmap':
        assert not matches
        return {'attempts': 0, 'accepted': 0, 'accepted_bytes': 0}
    assert len(matches) == 1
    attempts, accepted, count = map(int, matches[0])
    assert 0 < accepted <= attempts <= sum(row[1] for row in shards)
    assert accepted * 256 * 1024 <= count <= accepted * 64 * 1024 * 1024
    return {'attempts': attempts, 'accepted': accepted, 'accepted_bytes': count}


def replay(root=ROOT, previous=PREVIOUS):
    with tarfile.open(previous) as baseline:
        raw = body(baseline, 'raw/startup-phase.json')
    assert sha256(raw).hexdigest() == 'd4646afd9c4414672544843ce3d0a0ff3a885cd12810000e0de316deace09c28'
    prior = json.loads(raw)
    configurations = [[[row['name'], row['tensors']] for row in arm['shards']]
                      for arm in prior['starts'].values()]
    assert len(configurations) == 4 and all(row == configurations[0] for row in configurations)
    with tarfile.open(root / 'evidence.tar.xz') as archive:
        return audit(archive, configurations[0])


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
