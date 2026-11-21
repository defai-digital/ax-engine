"""Compare Flash Next sidecar readers in the native HTTP server."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from run_load_order import cache_state, fresh_clone
from scripts import check_flash_next_native_support as native
from scripts import run_flash_next_generate_matrix as matrix
from scripts.watch_flash_next_reference import CRITICAL_PRESSURE, sample_host


REVISION = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
ENV = {'AX_ENGINE_PREFIX_REUSE_DISABLED': '1', 'AX_MLX_PREFIX_CACHE_MAX_BYTES': '0',
       'AX_MLX_PREFIX_CACHE_MAX_ENTRIES': '0', 'AX_MLX_PREFIX_CACHE_DISK_DISABLED': '1'}
ARMS = ['name-order', 'descriptor-lazy']
STARTUP_TIMEOUT = 1800


def boot(args, root, mode, policy, ids, output):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(args.server), '--mlx', '--model-id', 'qwen3.8-flash-next:mxfp4',
               '--mlx-model-artifacts-dir', str(root), '--mlx-mtp-policy', policy,
               '--stream-experts', 'off', '--disable-ngram-acceleration', '--max-concurrent-requests', '1',
               '--total-blocks', '2048', '--host', '127.0.0.1', '--port', str(port)]
    log_path = args.output / f'{policy}-{mode}.log'
    with log_path.open('x') as log, (args.output / f'{policy}-{mode}-samples.jsonl').open('x') as samples:
        began = time.monotonic()
        env = ENV | {'AX_FLASH_NEXT_RESIDENT_LOAD_LAB': 'lazy-cpu',
                     'AX_FLASH_NEXT_SIDECAR_LOAD_LAB': mode}
        child = subprocess.Popen(command, env=native.server_env(env), stdout=log, stderr=log)
        try:
            after_sample = began
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'Owned native loader exited with {child.returncode}')
                sample = sample_host(child.pid)
                now = time.monotonic()
                if now >= after_sample:
                    print(json.dumps(sample | {'elapsed_seconds':now - began}),file=samples,flush=True)
                    after_sample = now + 15
                if sample['pressure'] in CRITICAL_PRESSURE:
                    raise RuntimeError('Critical memory pressure during native load')
                try:
                    native.request(base, '/v1/models')
                except OSError:
                    if now - began > STARTUP_TIMEOUT:
                        raise TimeoutError('Native load readiness deadline elapsed')
                    time.sleep(2)
                else:
                    break
            output['ready_seconds'] = time.monotonic() - began
            for run in ids:
                before = native.metrics(native.request(base, '/metrics').decode())
                result = matrix.generate(base, run, max_tokens=32, ignore_eos=True)
                after = native.metrics(native.request(base, '/metrics').decode())
                row = matrix.slim(result, before, after)
                assert len(row['output_tokens']) == 32 and result['prompt_tokens'] == run
                required = policy == 'required'
                native.check_activation(before, after, required, 32, 32)
                matrix.check_verifier_schedule(row, policy, 'canonical')
                assert row['mtp']['available'] is True, row['mtp']
                assert row['mtp']['requested'] is required, row['mtp']
                assert row['mtp']['active'] is required, row['mtp']
                output['cases'].append({'prompt_hash':matrix.ids_sha(run), **row})
        finally:
            native.stop_owned(child)
    logs = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
    assert 'sidecar failed to attach' not in logs
    assert 'sidecar is present but not attached' not in logs
    for label, expected_mode in (('resident', 'lazy-cpu'), ('sidecar', mode)):
        lines = [line for line in logs.splitlines() if f'flash_next_{label}_loader_lab' in line]
        assert len(lines) == 1 and f'loader_mode="{expected_mode}"' in lines[0], lines
        output[label] = {
            field: float(value)
            for field, value in re.findall(r'(tensor_count|shard_count|elapsed_seconds)=([0-9.]+)', lines[0])
        }
    assert output['resident']['tensor_count'] == 1942 and output['resident']['shard_count'] == 15
    assert output['sidecar']['tensor_count'] == 31


def sidecar_page_state(root, selection):
    state = cache_state({**selection, 'path': str(root / 'mtp.safetensors')})
    if state.get('selected_pages') != 318256:
        raise RuntimeError(f'Unexpected selected sidecar page state: {state}')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('server', 'pack', 'inventory', 'scratch-dir', 'build-receipt', 'output', 'sidecar-selection'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    build = json.loads(args.build_receipt.read_text())
    inventory = json.loads(args.inventory.read_text())
    selection = json.loads(args.sidecar_selection.read_text())
    sidecar_selection, = selection['files']
    assert len(sidecar_selection['names']) == 31 and selection['revision'] == REVISION
    sidecar, = (member for member in inventory['members'] if member['name'] == 'mtp.safetensors')
    assert sidecar['sha256'] == selection['expected_sha256']
    assert build['server_sha256'] == native.sha256(args.server)
    assert inventory['revision'] == REVISION and len(inventory['members']) == 49
    manifest_hash = native.sha256(args.pack / 'model-manifest.json')
    names = subprocess.check_output(['/bin/ps', '-axo', 'comm='], text=True).splitlines()
    if any(Path(name.strip()).name.startswith('ax-engine-server') for name in names):
        raise RuntimeError('The device has an active server')
    args.output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    record = {'status':'checking-publisher', 'build':build, 'pack_revision':REVISION,
              'hardware':'Mac Studio M2 Ultra 192 GB, external SSD', 'started':started, 'manifest_sha256':manifest_hash, 'arms':{}}

    def save(status):
        record['status'] = status
        temporary = args.output / 'receipt.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output / 'receipt.json')
        print(status,flush=True)

    save('checking-publisher')
    members = [entry['name'] for entry in inventory['members']]
    for member in inventory['members']:
        file = args.pack / member['name']
        assert file.stat().st_size == member['size'] and native.sha256(file) == member['sha256'], member['name']
    vocab = matrix.pack_vocab_size(args.pack)
    ids = [matrix.workload_ids(vocab, size, 0) for size in (512, 2048, 8192)]
    expected = ('prompt_hash', 'output_tokens', 'prompt_tokens', 'finish_reason', 'status',
                'mtp', 'verifier_schedule', 'verified_steps', 'route_errors')
    for policy, order in (('disabled', ARMS), ('required', reversed(ARMS))):
        policy_runs = record['arms'][policy] = {}
        for arm in order:
            with tempfile.TemporaryDirectory(dir=args.scratch_dir) as directory:
                root = Path(directory)
                for name in [*members, 'model-manifest.json']:
                    fresh_clone(args.pack / name, root / name)
                assert native.sha256(root / 'model-manifest.json') == manifest_hash
                output = policy_runs[arm] = {
                    'cases': [],
                    'sidecar_initial_page_state': sidecar_page_state(root, sidecar_selection),
                }
                save(f'{policy}-{arm}-starting')
                boot(args, root, arm, policy, ids, output)
                save(f'{policy}-{arm}-complete')
        baseline, lazy = (policy_runs[arm]['cases'] for arm in ARMS)
        for previous, candidate in zip(baseline, lazy, strict=True):
            for key in expected:
                assert previous[key] == candidate[key], (policy, previous['prompt_tokens'], key)
    assert native.sha256(args.pack / 'model-manifest.json') == manifest_hash
    for member in inventory['members']:
        file = args.pack / member['name']
        assert file.stat().st_size == member['size'] and native.sha256(file) == member['sha256'], member['name']
    save('complete')


if __name__ == '__main__':
    main()
