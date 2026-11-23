"""Replay the signed Flash Next L1-capacity experiment from its raw receipts."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
COLD_ARCHIVE = ROOT.parent / 'flash-next-natural-document-prefix-http-m2-20261123/evidence.tar.xz'
CORPUS_ARCHIVE = ROOT.parent / 'flash-next-mxfp4-mixed-serving-m5-20261121/evidence.tar.xz'
CORPUS = 'mixed-corpus.jsonl'
ORACLE = 'no-seed-cold-oracle.json'
DRIVERS = {
    'bench_ax_serving.py', 'build.json', 'capture_mixed_steps.py', 'check_flash_next_native_support.py',
    'mxfp4-inventory.json', 'qualify_qwen38_flash_next.py', 'run_fixed_extension.py',
    'run_flash_next_lifecycle.py', 'run_flash_next_prefill_split_probe.py', 'run_natural_eviction.py',
    'run_natural_fork.py', 'run_qwen38_reference_serving.py', 'run_selective_mixed.py',
    'run_warm_prefix.py', 'split_step_route.py', 'suffix-cases.json',
}
ARM_FILES = {
    f'observed/entries-{cap}/{name}'
    for cap in (2, 1)
    for name in ('partial.json', f'entries-{cap}.log', f'entries-{cap}-samples.jsonl')
}
RECEIPT = 'observed/receipt.json'
AUX = {'inputs.sha256', 'source.patch', 'status.txt', RECEIPT} | ARM_FILES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_member(archive, filename):
    member = archive.getmember(filename)
    require(member.isfile(), f'{filename} must be a regular file')
    with archive.extractfile(member) as stream:
        return stream.read()


def arm_summary(receipt, files):
    result = {}
    for limit in (2, 1):
        name = str(limit)
        arm = receipt['arms'][name]
        samples = [
            json.loads(row) for row in files[f'observed/entries-{limit}/entries-{limit}-samples.jsonl'].splitlines()
        ]
        ledger = json.loads(files[f'observed/entries-{limit}/partial.json'])
        require(ledger == arm['cases'], f'{name} entry ledger changed')
        require(samples and len(samples) == arm['normal_pressure_samples']
                and all(row['pressure'] == 1 for row in samples), f'{name} entry pressure changed')
        cases = {}
        for label, row in ledger.items():
            cases[label] = {
                'physical_reuse': dict(zip(('tokens', 'hits'), (
                    after - before for before, after in zip(row['reuse_before'], row['reuse_after'], strict=True)
                ), strict=True)),
            }
        result[name] = {'normal_pressure_samples': len(samples), 'cases': cases}
    return result


def replay(path):
    with (tarfile.open(path) as archive, tarfile.open(COLD_ARCHIVE) as cold_archive,
          tarfile.open(CORPUS_ARCHIVE) as corpus_archive, TemporaryDirectory(prefix='flash-next-eviction-') as temp):
        names = archive.getnames()
        require(set(names) == DRIVERS | AUX and len(names) == len(DRIVERS | AUX), 'Unexpected archive file list')
        files = {name: read_member(archive, name) for name in names}
        files[CORPUS] = read_member(corpus_archive, CORPUS)
        files[ORACLE] = read_member(cold_archive, 'cold/observed/receipt.json')
        pairs = [line.split() for line in files['inputs.sha256'].decode().splitlines()]
        manifest = {name: digest for digest, name in pairs}
        required = DRIVERS | {CORPUS, ORACLE}
        require(set(manifest) == required | {'ax-engine-server'} and len(pairs) == len(manifest)
                and all(manifest[name] == sha256(files[name]).hexdigest() for name in required),
                'Driver hashes changed')
        build = json.loads(files['build.json'])
        require(build['server_sha256'] == manifest['ax-engine-server']
                and build['patch_sha256'] == sha256(files['source.patch']).hexdigest()
                and build['codesign_verified'] is True and files['status.txt'].strip() == b'0',
                'Signed runtime or status changed')
        stage = Path(temp)
        for name, body in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        subprocess.check_output([sys.executable, '-B', str(stage / 'run_natural_eviction.py'),
                                 '--replay', str(stage / RECEIPT)])
        receipt = json.loads(files[RECEIPT])
        oracle = json.loads(files[ORACLE])
        labels = ('first', 'replacement', 'aligned')
        count = sum(len(ids) for label in labels for ids in oracle['outputs'][label].values())
        require(count == 864 and receipt['qualification'] is False, 'Comparison scope changed')
        return {'comparison_arms': arm_summary(receipt, files), 'compared_token_ids_per_arm': count,
                'server_sha256': build['server_sha256'], 'qualification': False}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence.tar.xz')
    print(json.dumps(replay(parser.parse_args().archive), indent=2, sort_keys=True))
