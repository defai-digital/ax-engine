"""Replay the observed M5 resident-reader crossover without a GPU or checkpoint."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
VOCAB_SIZE = 248_320
PIN = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
BUILD = '31e2b6e1f10beb42ddb2a7f5ef3ec8a6e96d4356'
MANIFEST = '62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def extract(archive_path, root):
    with tarfile.open(archive_path) as archive:
        for member in archive:
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts,
                    'Unexpected archive member')
            target = root / path
            require(not target.exists(), 'Duplicate archive member')
            target.parent.mkdir(exist_ok=True, parents=True)
            with archive.extractfile(member) as data:
                target.write_bytes(data.read())


def check_inputs(root):
    digests = {name: checksum for checksum, name in
               (line.split() for line in (root / 'inputs.sha256').read_text().splitlines())}
    elided = {'ax-engine-server', 'loader_footprint.dylib'}
    for name, checksum in digests.items():
        if name not in elided:
            require(digest(root / name) == checksum, f'Input digest changed for {name}')
    build = read(root / 'build.json')
    require(build['commit'] == BUILD and build['profile'] == 'release-server'
            and build['rustc'].startswith('rustc 1.99.1 ')
            and build['mlx'] == build['mlx_metal'] == '0.32.4'
            and build['inventory_sha256'] == digests['inventory.json']
            and build['server_sha256'] == digests['ax-engine-server'],
            'Candidate provenance changed')
    inventory = read(root / 'inventory.json')
    require(inventory['revision'] == PIN and len(inventory['members']) == 49
            and digests['model-manifest.json'] == MANIFEST,
            'Frozen publisher inputs changed')
    return build, digests


def replay_policy(root, policy, build, digests, runner):
    receipt = read(root / policy / 'receipt.json')
    require((root / f'{policy}-status.txt').read_text() == '0\n'
            and receipt['status'] == 'complete' and receipt['policy'] == policy
            and receipt['launch'] == f'{policy}-32k-resident-single-request'
            and receipt['hardware'] == 'MacBook Pro M5 Max 128 GB, internal SSD'
            and receipt['pack_revision'] == PIN and receipt['build'] == build
            and receipt['libmlx_sha256'] == build['libmlx_sha256'],
            f'{policy}: model, profile or process identity changed')
    dispatch = read(root / f'dispatch-capacity-{policy}.json')['capacity']
    require(dispatch['pressure'] == 1 and dispatch['eligible_for_attempt'] is False
            and dispatch['reason'] == 'background-rss-exceeds-signed-margin',
            f'{policy}: ordinary-reference dispatch conditions changed')
    for field, name in (
        ('runner_sha256', 'run_aligned.py'), ('model_manifest_sha256', 'model-manifest.json'),
        ('inventory_sha256', 'inventory.json'), ('resident_page_selection_sha256', 'resident-page-selection.json'),
        ('footprint_probe_source_sha256', 'loader_footprint.c'), ('footprint_probe_sha256', 'loader_footprint.dylib'),
    ):
        require(receipt[field] == digests[name], f'{policy}: input receipt changed for {name}')

    require(list(receipt['arms']) == [label for label, _, _ in runner.ARMS],
            f'{policy}: counterbalanced order changed')
    for label, mode, _ in runner.ARMS:
        arm = receipt['arms'][label]
        require(sum(row['verified_steps'] for row in arm['cases']) == 277,
                f'{policy}/{label}: recorded verification total changed')
        trace = {}
        measured = root / policy
        runner.attach_loader_trace((measured / f'{label}.txt').read_text(), mode, policy, trace)
        require(all(arm[key] == value for key, value in trace.items()),
                f'{policy}/{label}: reader trace changed')
        samples = [json.loads(line) for line in (measured / f'{label}-samples.jsonl').read_text().splitlines()]
        require(sorted({row['pressure'] for row in samples}) == arm['pressure_levels'] == [1]
                and len(samples) >= 2 and all(
                    later['elapsed_seconds'] > earlier['elapsed_seconds']
                    for earlier, later in zip(samples, samples[1:])
                ), f'{policy}/{label}: missing or invalid resource observations')
        require(arm['peak_sampled_rss_bytes'] == max(row['reference_rss_bytes'] for row in samples
                    if 'reference_rss_bytes' in row), f'{policy}/{label}: RSS summary changed')
        require(all(isfinite(arm[name]) and arm[name] > 0
                    for name in ('ready_seconds', 'load_phase_seconds', 'sidecar_load_seconds'))
                and all(isfinite(case['timing']['ttft_seconds']) and case['timing']['ttft_seconds'] > 0
                        and case['timing']['decode_tokens'] == runner.COMPLETION_TOKENS - 1
                        and isfinite(case['timing']['decode_seconds']) and case['timing']['decode_seconds'] > 0
                        for case in arm['cases']),
                f'{policy}/{label}: missing or invalid startup or client timing')

    result = runner.compare(receipt['arms'], VOCAB_SIZE, policy)
    normalized = json.loads(json.dumps(result, allow_nan=False))
    require(normalized == receipt['diagnostic_comparison'], f'{policy}: measured cells do not match comparison')
    return normalized


def replay(archive_path):
    with TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        extract(archive_path, root)
        build, digests = check_inputs(root)
        sys.path.insert(0, str(root))
        import run_aligned as runner
        require(Path(runner.__file__).resolve() == root / 'run_aligned.py', 'Unexpected replay runner')
        try:
            phases = [policy for policy in ('default', 'required')
                      if (root / f'{policy}-status.txt').is_file()]
            require(phases == ['required'], 'Unexpected terminal policy set')
            return {'commit': build['commit'], 'qualification': False,
                    'phases': {policy: replay_policy(root, policy, build, digests, runner)
                               for policy in phases}}
        finally:
            sys.path.remove(str(root))


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'evidence.tar.xz'
    print(json.dumps(replay(path), indent=2, allow_nan=False, sort_keys=True))
