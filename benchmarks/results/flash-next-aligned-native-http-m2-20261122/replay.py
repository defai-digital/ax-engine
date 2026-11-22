"""Recheck functional parity of the resident aligned reader on the native HTTP path."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
SCRIPTS = ROOT.parents[2] / 'scripts'
PIN = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
MANIFEST = '62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef'
INVENTORY = RESULTS / 'qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json'
SELECTION = RESULTS / 'flash-next-sidecar-reader-warm-m5-20261122/expected-resident-selection.json'
ARM_ORDER = ('default-mapped', 'default-aligned', 'required-aligned', 'required-mapped')
sys.path.insert(0, str(SCRIPTS))
import check_flash_next_native_support as native


def require(passed, message):
    if not passed:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def stable(row):
    response = row['response']
    usage = response['usage']
    return (row['name'], row['request'], response['choices'],
            tuple(usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')),
            row['activation_delta'])


def replay(root=ROOT):
    build, result, trace, software = (read(root / name) for name in
        ('build.json', 'native-result.json', 'resident-traces.json', 'software.json'))
    manifest = {name: value for value, name in (row.split() for row in
                (root / 'inputs.sha256').read_text().splitlines())}
    for staged, path in (('build.json', root / 'build.json'), ('run_smoke.py', root / 'run_smoke.py'),
                         ('inventory.json', INVENTORY), ('qualify_qwen38_flash_next.py', SCRIPTS / 'qualify_qwen38_flash_next.py'),
                         ('check_flash_next_native_support.py', Path(native.__file__))):
        require(manifest[staged] == digest(path), f'Staged input changed: {staged}')
    require(manifest['ax-engine-server'] == build['server_sha256'] and build['mlx'] == build['mlx_metal'] == '0.32.4'
            and build['commit'].startswith('31e2b6e1f') and build['inventory_sha256'] == digest(INVENTORY)
            and build['rustc'].startswith('rustc 1.99.1 ') and build['profile'] == 'release-server',
            'Tested Rust, MLX, pack or binary identity changed')
    require(result['build'] == build and result['completed'] is True and result['qualification'] is False
            and result['pack_revision'] == PIN and result['manifest'] == {'model-manifest.json': MANIFEST}
            and result['scope'] == 'M2 Ultra functional native HTTP'
            and (root / 'status.txt').read_text() == '0\n', 'The full native smoke did not complete')
    require(software['rust_workspace'] == 4790 and software['rust_failed'] == 0
            and software['rust_ignored'] == 107 and software['python_passed'] == 345
            and software['python_skipped'] == 7 and software['software_passed'] is True,
            'Software acceptance did not run on the candidate')
    require(tuple(result['runs']) == ARM_ORDER == tuple(trace), 'Native reader process order changed')
    baseline, arms = {}, {}
    for label in ARM_ORDER:
        arm, observed = result['runs'][label], trace[label]
        policy, reader = label.split('-')
        aligned = reader == 'aligned'
        require(arm['mode'] == policy and arm['aligned_nocache'] is aligned
                and arm['pressure'] == ['1', '1'] and arm['native_shards'] == 15 and arm['native_tensors'] == 1942,
                'Functional arm did not preserve ownership or resident coverage')
        require(observed['aligned_nocache'] is aligned and observed['shards'] == read(SELECTION)
                and observed['resident'] is True,
                'A startup log did not show the requested resident reader and selected tensors')
        require(arm['matches'] == len(arm['rows']) == 7 and
                [row['name'] for row in arm['rows']] == [name for name, _, _ in native.CASES],
                'Native HTTP control count or name changed')
        for row in arm['rows']:
            native.check_answer(row['name'], row['response'])
            steps = native.check_activation(row['metrics_before'], row['metrics_after'], policy == 'required',
                row['response']['usage']['completion_tokens'], row['request']['max_tokens'])
            require(steps == row['activation_delta'], 'Replayed verifier counters changed')
        controls = list(map(stable, arm['rows']))
        require(controls == baseline.setdefault(policy, controls), 'HTTP content, token counts, stops or counters diverged')
        verified = sum(row['activation_delta'][native.VERIFIED] for row in arm['rows'])
        require(arm['verified_steps'] == verified == (107 if policy == 'required' else 0),
                'Unexpected MTP verification activity')
        require(0 < arm['ready_seconds'] and isfinite(arm['ready_seconds']), 'Missing readiness observation')
        arms[label] = {field: arm[field] for field in ('matches', 'verified_steps', 'ready_seconds')}
    return {'scope': result['scope'], 'qualification': False, 'commit': build['commit'],
            'software': software, 'native_arms': arms}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
