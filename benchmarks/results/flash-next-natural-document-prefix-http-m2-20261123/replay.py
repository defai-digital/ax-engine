"""Replay the cold and learned Flash Next no-seed HTTP receipts."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / 'flash-next-mxfp4-mixed-serving-m5-20261121/evidence.tar.xz'
ORDER = ('first', 'unseeded', 'replacement', 'aligned', 'cancel', 'boundary', 'outside')
LEARNED_HITS = {'aligned', 'cancel', 'boundary'}
SEPARATE_FILES = {'build.json', 'inputs.sha256', 'run_natural_fork.py', 'source.patch', 'status.txt'}
OBSERVED_FILES = {'receipt.json', 'partial.json', 'natural-samples.jsonl', 'natural.log'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_member(archive, filename):
    member = archive.getmember(filename)
    require(member.isfile(), f'{filename} must be a regular file')
    with archive.extractfile(member) as stream:
        return stream.read()


def stage_arm(archive, stage, common, name):
    directory = stage / name
    directory.mkdir()
    files = dict(common)
    files.update({filename: read_member(archive, f'{name}/{filename}') for filename in SEPARATE_FILES})
    manifest_rows = [row.split() for row in files['inputs.sha256'].decode().splitlines()]
    manifest = {filename: digest for digest, filename in manifest_rows}
    drivers = set(common) | {'run_natural_fork.py', 'build.json'}
    require(set(manifest) == drivers | {'ax-engine-server'} and len(manifest_rows) == len(manifest),
            f'{name} staged driver list changed')
    require(all(manifest[filename] == sha256(files[filename]).hexdigest() for filename in drivers),
            f'{name} staged driver contents changed')
    build = json.loads(files['build.json'])
    require(build['server_sha256'] == manifest['ax-engine-server']
            and build['patch_sha256'] == sha256(files['source.patch']).hexdigest()
            and build['codesign_verified'] is True and build['mlx'] == build['mlx_metal'] == '0.32.4'
            and files['status.txt'].strip() == b'0', f'{name} build or exit status changed')
    for filename in drivers:
        (directory / filename).write_bytes(files[filename])
    for filename in OBSERVED_FILES:
        (directory / filename).write_bytes(read_member(archive, f'{name}/observed/{filename}'))
    return directory


def check_arm(directory, learn):
    script = """
from pathlib import Path
import json
import sys
from run_fixed_extension import CASES, MANIFEST, check_case, read

root = Path(sys.argv[1])
learn = sys.argv[2] == '1'
hits = set(sys.argv[3:])
receipt = read(root / 'receipt.json')
if receipt['qualification'] is not False or receipt['explicit_seed_requests'] != 0:
    raise ValueError('This must be an unseeded experiment')
if receipt.get('learning_enabled', False) != learn:
    raise ValueError('The learning switch changed')
if receipt['manifest'] != MANIFEST or receipt['build'] != read(root / 'build.json'):
    raise ValueError('The publisher data or signed executable changed')
if receipt['cases'] != read(root / 'partial.json'):
    raise ValueError('The case ledger changed')
outputs = {name: check_case(receipt['cases'][name], name in hits, stop, tail)
           for name, _, _, stop, _, tail in CASES}
if outputs != receipt['outputs']:
    raise ValueError('The recorded token IDs changed')
samples = [json.loads(row) for row in (root / 'natural-samples.jsonl').read_text().splitlines()]
if not samples or len(samples) != receipt['normal_pressure_samples'] or any(row['pressure'] != 1 for row in samples):
    raise ValueError('The recorded host pressure changed')
"""
    subprocess.run(
        [sys.executable, '-B', '-c', script, str(directory), str(int(learn)),
         *sorted(LEARNED_HITS if learn else set())],
        cwd=directory, check=True,
    )
    return json.loads((directory / 'receipt.json').read_bytes())


def replay(path):
    with (tarfile.open(path) as archive, tarfile.open(CORPUS) as corpus,
          TemporaryDirectory(prefix='flash-next-natural-prefix-') as temp):
        names = archive.getnames()
        shared = {name for name in names if name.startswith('inputs/')}
        require(all('/' not in name.removeprefix('inputs/') for name in shared),
                'Unexpected shared input path')
        required = shared | {
            f'{arm}/{filename}' for arm in ('cold', 'learned')
            for filename in SEPARATE_FILES | {f'observed/{name}' for name in OBSERVED_FILES}
        }
        require(set(names) == required and len(names) == len(required), 'The archive file list changed')
        common = {name.removeprefix('inputs/'): read_member(archive, name) for name in shared}
        common['mixed-corpus.jsonl'] = read_member(corpus, 'mixed-corpus.jsonl')
        result = {}
        for arm in ('cold', 'learned'):
            directory = stage_arm(archive, Path(temp), common, arm)
            result[arm] = check_arm(directory, arm == 'learned')
        cold, learned = result['cold'], result['learned']
        require(cold['outputs'] == learned['outputs'] and cold['manifest'] == learned['manifest']
                and tuple(cold['order']) == tuple(learned['order']) == ORDER,
                'Prompt order, publisher data or token parity changed')
        count = sum(len(ids) for row in cold['outputs'].values() for ids in row.values())
        require(count == 1921, 'The number of compared token IDs changed')
        return {
            'arms': {
                arm: {
                    'normal_pressure_samples': receipt['normal_pressure_samples'],
                    'patch_sha256': receipt['build']['patch_sha256'],
                    'physical_reuse': {
                        name: dict(zip(('tokens', 'hits'), (
                            after - before for before, after in zip(
                                row['reuse_before'], row['reuse_after'], strict=True
                            )
                        ), strict=True))
                        for name, row in receipt['cases'].items()
                    },
                    'server_sha256': receipt['build']['server_sha256'],
                } for arm, receipt in result.items()
            },
            'compared_token_ids': count,
            'explicit_seed_requests': 0,
            'qualification': False,
        }


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence.tar.xz')
    print(json.dumps(replay(parser.parse_args().archive), indent=2, sort_keys=True))
