"""Verify completed Flash Next startup traces and account for nested timers."""
from datetime import datetime
from hashlib import sha256
import importlib.util
import json
from math import isfinite
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory

DIRECT = ('disabled-first/disabled', 'disabled-first/default',
          'default-first/default', 'default-first/disabled')
SERIAL_PHASES = ('trunk_admission', 'resident_tensors', 'trunk_weight_contract',
                 'trunk_assembly', 'mtp_sidecar', 'vision_tower')
VM_COUNTERS = ('Pageins', 'Pageouts', 'Swapins', 'Swapouts')
VM_GAUGES = ('Pages free', 'Pages occupied by compressor')
PHASE_SCRIPT = Path(__file__).with_name('capture_phases.py')
SELECTION_SHA256 = '7c897171b372eea008611ec23ccea62c971c43a1b62ad684fb60209b16d433e1'


def vm_snapshot(raw):
    page_size = re.search(r'page size of (\d+) bytes', raw)
    if page_size is None or int(page_size.group(1)) <= 0:
        raise ValueError('Missing kernel VM page size')
    pages = {}
    for name in (*VM_COUNTERS, *VM_GAUGES):
        matches = re.findall(rf'^{re.escape(name)}:\s+(\d+)\.$', raw, re.M)
        if len(matches) != 1:
            raise ValueError(f'Expected one system VM value for {name}')
        pages[name] = int(matches[0])
    return int(page_size.group(1)), pages


def vm_difference(samples):
    before_size, before = vm_snapshot(samples['before_launch'])
    after_size, after = vm_snapshot(samples['at_readiness'])
    if before_size != after_size or any(after[name] < before[name] for name in VM_COUNTERS):
        raise ValueError('System VM observations cross an invalid counter boundary')
    return {name: (after[name] - before[name]) * before_size for name in before}


def post_admission_seconds(log):
    lines = re.sub(r'\x1b\[[0-9;]*m', '', log).splitlines()
    stamps = []
    for phase in ('trunk_admission', 'vision_tower'):
        matches = [line.split()[0] for line in lines if 'Flash Next startup phase' in line
                   and f'phase="{phase}"' in line]
        if len(matches) != 1:
            raise ValueError(f'Expected one startup timestamp for {phase}')
        stamps.append(datetime.fromisoformat(matches[0]))
    return (stamps[1] - stamps[0]).total_seconds()


def account(run, expected_selection, post_admission):
    phases = run['phases_us']
    shards = run['shards']
    selection = len(shards), sum(row['tensors'] for row in shards)
    if (selection != expected_selection or len({row['name'] for row in shards}) != selection[0]
            or any(row['tensors'] <= 0 for row in shards)):
        raise ValueError(f'Incomplete resident shard selection: {selection}')
    plan_to_warm = run['plan_to_kernel_warm_seconds']
    after_weights = run['weights_to_kernel_warm_seconds']
    ready = run['ready_seconds']
    if (not all(isfinite(value) for value in (plan_to_warm, after_weights, ready, post_admission))
            or not 0 <= after_weights <= plan_to_warm <= ready
            or not 0 <= round(post_admission * 1e6) <= round((plan_to_warm - after_weights) * 1e6)):
        raise ValueError('Invalid plan, weights and readiness timeline')

    if (any(type(value) is not int or value < 0 for value in phases.values())
            or any(type(row[key]) is not int or row[key] < 0
                   for row in shards for key in ('read_us', 'eval_us'))):
        raise ValueError('Phase and shard counters must be nonnegative microseconds')

    selection_us = phases['resident_selection']
    read_us = sum(row['read_us'] for row in shards)
    eval_us = sum(row['eval_us'] for row in shards)
    resident_residual = phases['resident_tensors'] - selection_us - read_us - eval_us
    traced_us = sum(phases[name] for name in SERIAL_PHASES if name != 'trunk_admission')
    untraced_us = round(post_admission * 1e6) - traced_us
    if resident_residual < 0 or untraced_us < 0:
        raise ValueError('Nested or serial timers exceed their containing interval')

    return {
        'http_ready_seconds': run['ready_seconds'],
        'plan_to_kernel_warm_seconds': plan_to_warm,
        'weight_log_to_kernel_warm_seconds': after_weights,
        'resident': {
            'scope_us': phases['resident_tensors'],
            'selection_us': selection_us,
            'shard_copy_and_allocation_us': read_us,
            'shard_sync_eval_us': eval_us,
            'unattributed_us': resident_residual,
        },
        'serial_weight_phases_us': {name: phases[name] for name in SERIAL_PHASES},
        'post_admission_to_weight_log_seconds': post_admission,
        'post_admission_to_weight_log_unattributed_us': untraced_us,
        'system_vm_delta_bytes_before_launch_to_readiness': vm_difference(run['system_vm']),
    }


def verify_http_controls(root, build):
    summary = json.loads((root / 'result.json').read_bytes())
    orders = {'disabled-first': ['disabled', 'default'], 'default-first': ['default', 'disabled']}
    counter = 'ax_engine_flash_next_mtp_verified_steps_total'
    total = 0
    for order, modes in orders.items():
        report = json.loads((root / order / 'result.json').read_bytes())
        if (report['functional_controls_passed'] is not True or report['modes'] != modes
                or list(report['results']) != modes
                or report['server_sha256'] != build['server_sha256']
                or report['inventory_sha256'] != summary['inventory_sha256']
                or report['hardware'] != summary['hardware']):
            raise ValueError(f'{order}: startup control identity changed')
        for mode in modes:
            cases = report['results'][mode]
            if len(cases) != 7 or any(row['activation_delta'][counter] != 0 for row in cases):
                raise ValueError(f'{order}/{mode}: missing controls or observed MTP activation')
            total += len(cases)
    return total


def replay(root, evidence):
    spec = importlib.util.spec_from_file_location('capture_phases', PHASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ValueError('Startup capture replay is missing')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    terminal_raw = (root / 'startup-phase.json').read_bytes()
    terminal = json.loads(terminal_raw)
    regenerated = capture.compare_root(root, evidence / 'prior.json')
    if regenerated != terminal or tuple(terminal['starts']) != DIRECT:
        raise ValueError('Terminal receipt does not reproduce all four ordered starts')
    if terminal['build'] != json.loads((evidence / 'expected-build.json').read_bytes()):
        raise ValueError('Startup capture uses a different binary or source')
    controls = verify_http_controls(root, terminal['build'])

    selection_raw = (evidence / 'selection-control.json').read_bytes()
    if sha256(selection_raw).hexdigest() != SELECTION_SHA256:
        raise ValueError('The archived resident-loader selection proof changed')
    selection = json.loads(selection_raw)
    if selection['pack_revision'] != 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3':
        raise ValueError('The selection proof covers a different checkpoint')
    expected, = {(row['shard_count'], row['tensor_count']) for row in selection['arms'].values()}
    starts = {}
    for name, row in terminal['starts'].items():
        order, mode = name.split('/')
        log = (root / order / f'server-{mode}.log').read_text()
        starts[name] = account(row, expected, post_admission_seconds(log))
    return {
        'scope': 'M5 Flash Next startup attribution; diagnostic only',
        'source': terminal['build'],
        'terminal_sha256': sha256(terminal_raw).hexdigest(),
        'matched_http_controls': controls,
        'selection_control_sha256': SELECTION_SHA256,
        'starts': starts,
    }


if __name__ == '__main__':
    with TemporaryDirectory(prefix='ax-flash-startup-replay-') as directory:
        root = Path(directory)
        with tarfile.open(Path(__file__).with_name('evidence.tar.xz'), 'r:xz') as archive:
            archive.extractall(root, filter='data')
        print(json.dumps(replay(root / 'raw', root), indent=2, allow_nan=False))
