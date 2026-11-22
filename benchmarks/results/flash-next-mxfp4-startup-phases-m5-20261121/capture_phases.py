"""Read completed Flash Next shard and assembly timings from the same server."""
from datetime import datetime
import json
from hashlib import sha256
from pathlib import Path
import re
import sys

PHASES = ('trunk_admission', 'resident_selection', 'resident_tensors',
          'trunk_weight_contract', 'trunk_assembly', 'mtp_sidecar', 'vision_tower')
PHASE = re.compile(r'Flash Next startup phase.*\bphase="([a-z_]+)".*\belapsed_us=(\d+)')
SHARD = re.compile(r'Flash Next resident shard.*\bshard=Some\("([^"/]+)"\).*\btensor_count=(\d+)'
                   r'.*\bread_us=(\d+).*\beval_us=(\d+)')


def read_log(payload):
    text = re.sub(r'\x1b\[[0-9;]*m', '', payload)
    phases = PHASE.findall(text)
    if tuple(name for name, _ in phases) != PHASES:
        raise ValueError(f'Unexpected completed phases: {phases}')
    members = SHARD.findall(text)
    if not members or len(members) != len({name for name, *_ in members}):
        raise ValueError('The completed shard list is empty or repeats a name')
    return {name: int(value) for name, value in phases}, [
        {'name': name, 'tensors': int(tensors), 'read_us': int(read), 'eval_us': int(evaluated)}
        for name, tensors, read, evaluated in members
    ]


def compare_root(root, prior_path):
    repeat = json.loads((root / 'result.json').read_text())
    build = json.loads((root / 'build.json').read_text())
    assert repeat['control_count'] == 28 and repeat['same_output_and_inactive_drafter'] is True
    assert repeat['server_sha256'] == build['server_sha256']
    prior_raw = prior_path.read_bytes()
    assert sha256(prior_raw).hexdigest() == '76e61486f0cb8ee8024e4888b02929f0da731d9e6560246014a2ef465f5d98b0'
    prior = json.loads(prior_raw)
    assert repeat['inventory_sha256'] == prior['inventory_sha256']
    starts = {}
    baseline = None
    for order, modes in repeat['readiness_seconds'].items():
        report = json.loads((root / order / 'result.json').read_text())
        for mode, ready_seconds in modes.items():
            assert [row['response']['choices'] for row in report['results'][mode]] == [
                row['response']['choices'] for row in prior['results'][mode]]
            log = (root / order / f'server-{mode}.log').read_text()
            phases, shards = read_log(log)
            stamps = [datetime.fromisoformat(next(line.split()[0] for line in log.splitlines() if part in line))
                      for part in ('memory plan:', 'phase="vision_tower"', 'warmed linear-attention Metal kernels')]
            assert stamps[0] <= stamps[1] <= stamps[2]
            vm = json.loads((root / order / f'server-{mode}-vm.json').read_text())
            assert all('page size' in sample.lower() for sample in vm.values())
            selected = [(shard['name'], shard['tensors']) for shard in shards]
            if baseline is None:
                baseline = selected
            if selected != baseline:
                raise ValueError('Startup arms loaded different shard selections')
            assert report['readiness_seconds'][mode] == ready_seconds
            starts[f'{order}/{mode}'] = {
                'ready_seconds': ready_seconds,
                'phases_us': phases,
                'plan_to_kernel_warm_seconds': (stamps[2] - stamps[0]).total_seconds(),
                'weights_to_kernel_warm_seconds': (stamps[2] - stamps[1]).total_seconds(),
                'system_vm': vm,
                'shard_read_us': sum(row['read_us'] for row in shards),
                'shard_eval_us': sum(row['eval_us'] for row in shards),
                'shards': shards,
            }
    assert len(starts) == 4
    return {'scope': 'Flash Next startup phase diagnostic on one delivered snapshot',
            'qualification': False, 'release_ready': False, 'build': build,
            'prior_receipt_sha256': sha256(prior_raw).hexdigest(), 'starts': starts}


if __name__ == '__main__':
    print(json.dumps(compare_root(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
