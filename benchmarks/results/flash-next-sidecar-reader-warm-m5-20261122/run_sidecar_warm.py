"""Compare warmed serving with isolated Flash Next MTP sidecar readers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import socket
from statistics import median
import subprocess
import tempfile
from threading import Event
import time
from urllib.error import HTTPError

from run_load_order import cache_state, fresh_clone
from scripts import check_flash_next_native_support as native
from scripts import run_flash_next_generate_matrix as matrix
from scripts.watch_flash_next_reference import CRITICAL_PRESSURE, sample_host


REVISION = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
MODEL_ID = 'qwen3.8-flash-next'
CONTEXT = 32768
COMPLETION_TOKENS = 32
TIMED_LENGTHS = (512, 2048, 8192)
ARMS = [
    ('mmap-short', 'mmap', 512), ('pread-short', 'pread', 512),
    ('pread-long', 'pread', 8192), ('mmap-long', 'mmap', 8192),
]
WARM_REPEATS = (1, 2, 3)

def workload(first_length):
    other_length = 8192 if first_length == 512 else 512
    return [
        (first_length, 0, 'first'),
        (512, 100, 'warmup'),
        *((length, run, 'warm') for run in WARM_REPEATS for length in TIMED_LENGTHS),
        (other_length, 0, 'parity'),
        (2048, 0, 'parity'),
        (CONTEXT - COMPLETION_TOKENS, 0, 'boundary'),
    ]
STARTUP_TIMEOUT = 1800
BETWEEN_ARM_COOLDOWN_SECONDS = 20
MEMORY_GAUGES = {
    'mlx_active_bytes': 'ax_engine_memory_mlx_active_bytes',
    'mlx_cache_bytes': 'ax_engine_memory_mlx_cache_bytes',
    'mlx_peak_bytes': 'ax_engine_memory_mlx_peak_bytes',
    'server_rss_bytes': 'ax_engine_memory_host_resident_bytes',
}


def health(base):
    body = json.loads(native.request(base, '/health'))
    if body.get('status') != 'ok':
        raise RuntimeError('Native server did not report healthy')
    return body['status']


def ordinary_model(base):
    models = json.loads(native.request(base, '/v1/models'))['data']
    if len(models) != 1 or models[0]['id'] != MODEL_ID or models[0]['context_length'] != CONTEXT:
        raise RuntimeError('Expected the inferred Flash Next model with a 32768-token context')
    return {'id': models[0]['id'], 'context_length': models[0]['context_length']}


def verify_overflow(base, prompt):
    try:
        native.request(base, '/v1/completions',
                       {'model': MODEL_ID, 'prompt': prompt, 'max_tokens': COMPLETION_TOKENS + 1,
                        'temperature': 0})
    except HTTPError as error:
        code = json.load(error)['error']['code']
        if error.code == 400 and code == 'context_length_exceeded':
            return {'http_status': error.code, 'error_code': code}
        raise RuntimeError(f'Unexpected boundary response {error.code}, {code}') from error
    raise RuntimeError('The ordinary completions route accepted a 32769-token request')


def choose_ordinary_command(server, root, port):
    return [str(server), '--mlx', '--mlx-model-artifacts-dir', str(root),
            '--host', '127.0.0.1', '--port', str(port)]


def footprint_reader(library):
    function = ctypes.CDLL(str(library)).ax_lab_read_footprint
    pointer = ctypes.POINTER(ctypes.c_uint64)
    function.argtypes = (ctypes.c_int, pointer, pointer)
    function.restype = ctypes.c_int

    def sample(pid):
        physical, lifetime_peak = ctypes.c_uint64(), ctypes.c_uint64()
        error = function(pid, ctypes.byref(physical), ctypes.byref(lifetime_peak))
        if error:
            return {'physical_footprint_sample_errno': error}
        return {
            'physical_footprint_bytes': physical.value,
            'kernel_lifetime_peak_footprint_bytes': lifetime_peak.value,
        }

    return sample


def allocator_sample(metrics):
    return {key: int(metrics[name]) if name in metrics else None
            for key, name in MEMORY_GAUGES.items()}


def sample_live_server(child, began, samples, stop, footprint):
    while not stop.is_set():
        observation = sample_host(child.pid) | footprint(child.pid)
        print(json.dumps(observation | {'elapsed_seconds': time.monotonic() - began}),
              file=samples, flush=True)
        if observation['pressure'] in CRITICAL_PRESSURE:
            child.terminate()
            raise RuntimeError('Critical memory pressure during loader or inference control')
        stop.wait(5)


def boot(args, root, label, mode, cases, output, footprint, selection):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = choose_ordinary_command(args.server, root, port)
    log_path = args.output / f'{label}.log'
    with (
        log_path.open('x') as log,
        (args.output / f'{label}-samples.jsonl').open('x') as samples,
        ThreadPoolExecutor(max_workers=1) as collector,
    ):
        began = time.monotonic()
        experiment = {'RUST_LOG': 'info,ax_engine_mlx::flash_next_startup=debug'}
        if mode == 'pread':
            experiment['AX_FLASH_NEXT_MTP_SIDECAR_POSITIONAL'] = '1'
        child = subprocess.Popen(command, env=native.server_env(experiment), stdout=log, stderr=log)
        stop = Event()
        collection = collector.submit(sample_live_server, child, began, samples, stop, footprint)
        try:
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'Owned native loader exited with {child.returncode}')
                try:
                    output['model'] = ordinary_model(base)
                except OSError:
                    if time.monotonic() - began > STARTUP_TIMEOUT:
                        raise TimeoutError('Native load readiness deadline elapsed')
                    time.sleep(2)
                else:
                    break
            output['ready_seconds'] = time.monotonic() - began
            output['selected_pages_at_ready'] = selected_page_state(root, selection)
            output['health_ready'] = health(base)
            output['allocator_at_ready'] = allocator_sample(
                native.metrics(native.request(base, '/metrics').decode())
            )
            output['context_overflow'] = verify_overflow(base, cases[-1][1])
            for geometry, run in cases:
                before = native.metrics(native.request(base, '/metrics').decode())
                result = matrix.generate(base, run, max_tokens=COMPLETION_TOKENS, ignore_eos=True)
                after = native.metrics(native.request(base, '/metrics').decode())
                row = matrix.slim(result, before, after)
                if len(row['output_tokens']) != COMPLETION_TOKENS or result['prompt_tokens'] != run:
                    raise RuntimeError(f'Token geometry changed at {len(run)} prompt tokens')
                native.check_activation(before, after, False, COMPLETION_TOKENS, COMPLETION_TOKENS)
                matrix.check_verifier_schedule(row, 'default', 'canonical')
                output['cases'].append({
                    'prompt_hash': matrix.ids_sha(run), 'trial': geometry[1],
                    'phase': geometry[2], **row,
                    'allocator_after_request': allocator_sample(after),
                })
            output['health_after_requests'] = health(base)
            output['process_before_stop'] = footprint(child.pid)
        finally:
            stop.set()
            try:
                collection.result()
            finally:
                native.stop_owned(child)
    observations = [
        json.loads(line)
        for line in (args.output / f'{label}-samples.jsonl').read_text().splitlines()
    ]
    output['peak_sampled_rss_bytes'] = max(
        row['reference_rss_bytes'] for row in observations if 'reference_rss_bytes' in row
    )
    for metric in ('physical_footprint_bytes', 'kernel_lifetime_peak_footprint_bytes'):
        present = [row[metric] for row in (*observations, output['process_before_stop']) if metric in row]
        output[f'max_observed_{metric}'] = max(present, default=None)
    output['pressure_levels'] = sorted({row['pressure'] for row in observations})
    log = log_path.read_text()
    profile = [line for line in log.splitlines()
               if 'resolved the Flash Next M5 Max 128 GB serving profile' in line]
    plan = [line for line in log.splitlines() if 'memory plan: ' in line]
    if not any('context_tokens=32768' in line and 'effective_sequence_cap=Some(1)' in line for line in profile):
        raise RuntimeError('The ordinary launch did not resolve the one-request 32K profile')
    if not any('decision=resident' in line and 'live-sequences=1 ' in line for line in plan):
        raise RuntimeError('The ordinary launch did not keep Flash Next resident')
    output['profile'] = {'effective_sequence_cap': 1, 'expert_residency': 'resident'}
    attach_loader_trace(log, mode, output)


def attach_loader_trace(log, mode, output):
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    phases = re.findall(r'Flash Next startup phase phase="resident_tensors" elapsed_us=(\d+)', log)
    shards = [[name, int(count)] for name, count in re.findall(
        r'Flash Next resident shard shard=Some\("([^"/]+)"\) tensor_count=(\d+)', log)]
    expected = json.loads(Path(__file__).with_name('expected-resident-selection.json').read_bytes())
    if len(phases) != 1 or int(phases[0]) <= 0 or shards != expected:
        raise ValueError('Unexpected resident selection or missing startup phase')
    output['shard_count'] = len(shards)
    output['tensor_count'] = sum(count for _, count in shards)
    output['load_phase_seconds'] = int(phases[0]) / 1e6
    if 'Flash Next resident positional read lab' in log:
        raise ValueError('The trunk reader changed during the sidecar comparison')
    if log.count('Flash Next MTP sidecar positional read lab') != int(mode == 'pread'):
        raise ValueError('Unexpected sidecar payload route')
    sidecar = re.findall(r'Flash Next startup phase phase="mtp_sidecar" elapsed_us=(\d+)', log)
    if len(sidecar) != 1 or int(sidecar[0]) <= 0:
        raise ValueError('Missing isolated sidecar stage')
    output['sidecar_load_seconds'] = int(sidecar[0]) / 1e6
    output['positional_sidecar_reads'] = mode == 'pread'


def compare(arms, vocab):
    if set(arms) != {label for label, _, _ in ARMS}:
        raise ValueError('Incomplete run')
    baseline = arms[ARMS[0][0]]
    outputs = {}
    for label, mode, first in ARMS:
        arm = arms[label]
        geometry = workload(first)
        rows = arm['cases']
        expected = [
            (length, trial, phase, matrix.ids_sha(matrix.workload_ids(vocab, length, trial)))
            for length, trial, phase in geometry
        ]
        observed = [
            (row['prompt_tokens'], row['trial'], row['phase'], row['prompt_hash'])
            for row in rows
        ]
        if observed != expected or len(set(row[-1] for row in observed)) != len(expected):
            raise ValueError(f'{label}: inconsistent prompt geometry or missing sample')
        if (
            arm['loader_mode'] != mode or arm['pressure_levels'] != [1]
            or not 90 <= arm['quiet_baseline']['cpu_idle_percent'] <= 100
            or not 90 <= arm['post_run_cpu_idle_percent'] <= 100
            or arm['health_ready'] != 'ok' or arm['health_after_requests'] != 'ok'
            or arm['context_overflow'] != {'http_status': 400, 'error_code': 'context_length_exceeded'}
            or any(arm[key] != baseline[key] for key in ('tensor_count', 'shard_count', 'model', 'profile'))
        ):
            raise ValueError(f'{label}: profile, pressure or lifecycle changed')
        for row in rows:
            tokens = row['output_tokens']
            if len(tokens) != COMPLETION_TOKENS or outputs.setdefault(row['prompt_hash'], tokens) != tokens:
                raise ValueError(f'{label}: output identities changed')

    def time_summary(rows):
        return {
            'ttft_seconds': median(row['timing']['ttft_seconds'] for row in rows),
            'decode_tokens_per_second': median(matrix.decode_rate(row['timing']) for row in rows),
        }

    per_arm = {}
    for label, _, _ in ARMS:
        arm = arms[label]
        first = arm['cases'][0]
        per_arm[label] = {
            'ready_seconds': arm['ready_seconds'],
            'load_phase_seconds': arm['load_phase_seconds'],
            'sidecar_load_seconds': arm['sidecar_load_seconds'],
            'first_prompt_tokens': first['prompt_tokens'],
            'first_request': time_summary([first]),
            'ready_seconds_plus_first_ttft': arm['ready_seconds'] + first['timing']['ttft_seconds'],
            'warmed_medians': {
                length: time_summary([row for row in arm['cases']
                                      if row['phase'] == 'warm' and row['prompt_tokens'] == length])
                for length in TIMED_LENGTHS
            },
        }

    warm_ratios = {}
    for order in ('short', 'long'):
        candidate = per_arm[f'pread-{order}']['warmed_medians']
        control = per_arm[f'mmap-{order}']['warmed_medians']
        warm_ratios[order] = {
            length: {metric: candidate[length][metric] / value for metric, value in readings.items()}
            for length, readings in control.items()
        }
    return {
        'matched_output_ids_per_arm': len(baseline['cases']) * COMPLETION_TOKENS,
        'slowest_pread_to_fastest_mmap_ready_ratio': (
            max(per_arm[f'pread-{order}']['ready_seconds'] for order in ('short', 'long'))
            / min(per_arm[f'mmap-{order}']['ready_seconds'] for order in ('short', 'long'))
        ),
        'arms': per_arm,
        'warm_candidate_to_control_ratios': warm_ratios,
        'warmed_timing_band_met': all(
            readings['ttft_seconds'] <= 1.03 and readings['decode_tokens_per_second'] >= 0.97
            for order in warm_ratios.values() for readings in order.values()
        ),
    }

def selected_page_state(root, selection):
    return [
        {'file': row['file'],
         **cache_state({'path': str(root / row['file']), 'names': row['names']})}
        for row in selection['files']
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('server', 'pack', 'inventory', 'scratch-dir', 'build-receipt', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    model_code = subprocess.check_output(['/usr/sbin/sysctl','-n','hw.model'],text=True).strip()
    assert model_code == 'Mac17,6'
    build = json.loads(args.build_receipt.read_text())
    inventory = json.loads(args.inventory.read_text())
    assert build['server_sha256'] == native.sha256(args.server)
    assert inventory['revision'] == REVISION and len(inventory['members']) == 49
    names = subprocess.check_output(['/bin/ps', '-axo', 'comm='], text=True).splitlines()
    if any(Path(name.strip()).name.startswith('ax-engine-server') for name in names):
        raise RuntimeError('The device has an active server')
    fixture_dir = Path(__file__).resolve().parent
    manifest = fixture_dir / 'model-manifest.json'
    page_plan = fixture_dir / 'resident-page-selection.json'
    footprint_library = fixture_dir / 'loader_footprint.dylib'
    footprint = footprint_reader(footprint_library)
    selection = json.loads(page_plan.read_text())
    assert selection['revision'] == REVISION and len(selection['files']) == 3
    args.output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    record = {'status':'checking-publisher', 'build':build, 'pack_revision':REVISION,
              'hardware':'MacBook Pro M5 Max 128 GB, internal SSD',
              'macos_version': platform.mac_ver()[0],
              'runner_sha256': native.sha256(Path(__file__)),
              'model_manifest_sha256': native.sha256(manifest),
              'inventory_sha256': native.sha256(args.inventory),
              'footprint_probe_sha256': native.sha256(footprint_library),
              'footprint_probe_source_sha256': native.sha256(fixture_dir / 'loader_footprint.c'),
              'resident_page_selection_sha256': native.sha256(page_plan),
              'libmlx_sha256': native.sha256(Path(os.environ['DYLD_LIBRARY_PATH']) / 'libmlx.dylib'),
              'started':started, 'launch':'ordinary-32k-resident-single-request',
              'scope':'guarded mmap trunk with isolated MTP sidecar reader comparison',
              'between_arm_cooldown_seconds': BETWEEN_ARM_COOLDOWN_SECONDS,
              'workloads': {label: workload(first) for label, _, first in ARMS}, 'arms':{}}

    def save(status):
        record['status'] = status
        temporary = args.output / 'receipt.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output / 'receipt.json')
        print(status,flush=True)

    try:
        save('checking-publisher')
        members = [entry['name'] for entry in inventory['members']]
        for entry in inventory['members']:
            file = args.pack / entry['name']
            assert file.stat().st_size == entry['size'] and native.sha256(file) == entry['sha256'], entry['name']
        vocab = matrix.pack_vocab_size(args.pack)
        for label, mode, first in ARMS:
            cases = [(case, matrix.workload_ids(vocab, *case[:2])) for case in workload(first)]
            with tempfile.TemporaryDirectory(dir=args.scratch_dir) as directory:
                root = Path(directory)
                for name in members:
                    fresh_clone((args.pack/name).resolve(strict=True), root/name)
                (root/'model-manifest.json').write_bytes(manifest.read_bytes())
                if record['arms']:
                    time.sleep(BETWEEN_ARM_COOLDOWN_SECONDS)
                quiet = native.wait_for_quiet()
                output = record['arms'][label] = {
                    'cases': [], 'loader_mode': mode, 'quiet_baseline': quiet,
                    'initial_selected_page_samples': selected_page_state(root, selection),
                }
                save(label+'-starting')
                boot(args, root, label, mode, cases, output, footprint, selection)
                output['post_run_cpu_idle_percent'] = native.cpu_idle_percent()
                save(label+'-complete')
        for entry in inventory['members']:
            file = args.pack / entry['name']
            if file.stat().st_size != entry['size'] or native.sha256(file) != entry['sha256']:
                raise RuntimeError(f'Publisher member changed during measurement: {file.name}')
        record['diagnostic_comparison'] = compare(record['arms'], vocab)
        save('complete')
    except Exception:
        save('failed')
        raise


if __name__ == '__main__':
    main()
