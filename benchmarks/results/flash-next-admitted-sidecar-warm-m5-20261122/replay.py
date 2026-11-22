"""Audit timing eligibility and token identity of the exact sidecar candidate."""
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
PROTOTYPE = ROOT.parent / 'flash-next-sidecar-reader-warm-m5-20261122' / 'replay.py'
STARTUP = ROOT.parent / 'flash-next-admitted-sidecar-delivered-m5-20261122' / 'comparison.json'


def replay(receipt, common, build):
    require = common['require']
    arms, widths, vocab = (common[key] for key in ('ARMS', 'WIDTHS', 'VOCAB'))
    require(receipt['build'] == build, 'Warm and delivery candidate builds differ')
    require(receipt['scope'] == 'guarded mmap trunk with isolated MTP sidecar reader comparison'
            and receipt['launch'] == 'ordinary-32k-resident-single-request'
            and receipt['pack_revision'] == 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
            and receipt['inventory_sha256'] == '97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6'
            and receipt['model_manifest_sha256'] == '62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef'
            and receipt['runner_sha256'] == sha256(PROTOTYPE.with_name('run_sidecar_warm.py').read_bytes()).hexdigest()
            and list(receipt['arms']) == list(arms), 'Wrong scope or incomplete four-arm run')
    token_outputs = {}
    per_arm = {}
    ineligible = []
    for label, first_width in arms.items():
        arm = receipt['arms'][label]
        mode = label.split('-')[0]
        require(arm['loader_mode'] == mode and arm['positional_sidecar_reads'] is (mode == 'pread')
                and 'positional_payload_reads' not in arm, 'The guarded trunk reader changed')
        require(arm['model'] == {'id': 'qwen3.8-flash-next', 'context_length': 32768}
                and arm['profile'] == {'effective_sequence_cap': 1, 'expert_residency': 'resident'}
                and arm['pressure_levels'] == [1]
                and arm['health_ready'] == arm['health_after_requests'] == 'ok'
                and arm['context_overflow'] == {'http_status': 400, 'error_code': 'context_length_exceeded'}
                and arm['shard_count'] == 15 and arm['tensor_count'] == 1942,
                f'Unexpected lifecycle or profile for {label}')
        pre = arm['quiet_baseline']['cpu_idle_percent']
        post = arm['post_run_cpu_idle_percent']
        require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100
                    for value in (pre, post)), 'Invalid host readings')
        if min(pre, post) < 90:
            ineligible.append(label)
        rows = arm['cases']
        for row, (length, trial, phase) in zip(rows, common['workload'](first_width), strict=True):
            generator = common['matrix']
            prompt_hash = generator.ids_sha(generator.workload_ids(vocab, length, trial))
            require((row['prompt_tokens'], row['trial'], row['phase'], row['prompt_hash'])
                    == (length, trial, phase, prompt_hash), 'The named warm workload changed')
            tokens = row['output_tokens']
            require(row['route_errors'] == row['verified_steps'] == 0 and len(tokens) == 32
                    and all(type(token) is int and 0 <= token < vocab for token in tokens)
                    and token_outputs.setdefault(prompt_hash, tokens) == tokens,
                    'A direct request changed output or route')
        sidecar, trunk, ready = (arm[key] for key in ('sidecar_load_seconds','load_phase_seconds','ready_seconds'))
        require(all(math.isfinite(value) and value > 0 for value in (sidecar, trunk, ready))
                and sidecar + trunk < ready, 'Invalid startup phase timers')
        per_arm[label] = {'sidecar_load_seconds': sidecar, 'trunk_load_seconds': trunk,
                          'ready_seconds': ready, 'cpu_idle_before_percent': pre,
                          'cpu_idle_after_percent': post, 'matched_output_ids': 32 * len(rows),
                          'warmed_medians': {str(length): common['summarize_times']([
                              row for row in rows if row['phase'] == 'warm' and row['prompt_tokens'] == length
                          ]) for length in widths}}
    require(receipt['status'] == ('failed' if ineligible else 'complete'), 'Terminal eligibility disagrees')
    ratios = {}
    for order in ('short', 'long'):
        control = per_arm[f'mmap-{order}']['warmed_medians']
        candidate = per_arm[f'pread-{order}']['warmed_medians']
        ratios[order] = {length: {name: candidate[length][name] / value for name, value in row.items()}
                         for length, row in control.items()}
    numeric = all(row['ttft_seconds'] <= 1.03 and row['decode_tokens_per_second'] >= 0.97
                  for order in ratios.values() for row in order.values())
    return {'timing_eligible': not ineligible, 'ineligible_arms': ineligible,
            'warm_ratio_limits_met_numerically': numeric, 'warmed_gate_passed': not ineligible and numeric,
            'per_arm': per_arm, 'warm_candidate_to_control_ratios': ratios}


if __name__ == '__main__':
    observed = gzip.decompress((ROOT / 'receipt.json.gz').read_bytes())
    if sha256(observed).hexdigest() != (ROOT / 'original.sha256').read_text().split()[0]:
        raise ValueError('Original warm receipt bytes changed')
    common = runpy.run_path(str(PROTOTYPE))
    build = json.loads(STARTUP.read_bytes())['build']
    comparison = replay(json.loads(observed), common, build)
    code = int((ROOT / 'status.txt').read_text())
    if code != (0 if comparison['timing_eligible'] else 1):
        raise ValueError('Observed process exit disagrees with timing eligibility')
    print(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False))
