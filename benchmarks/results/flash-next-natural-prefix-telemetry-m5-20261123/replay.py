"""Replay the signed no-seed Flash Next crossover on MacBook Pro M5 Max 128 GB."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
CORPUS_ARCHIVE = ROOT.parent / 'flash-next-mxfp4-mixed-serving-m5-20261121/evidence.tar.xz'
ORACLE_ARCHIVE = ROOT.parent / 'flash-next-variable-prefix-suffix-http-m5-20261123/evidence.tar.xz'
EVICTION_ARCHIVE = ROOT.parent / 'flash-next-natural-prefix-eviction-m2-20261123/evidence.tar.xz'
CORPUS = 'mixed-corpus.jsonl'
DRIVERS = {
    'bench_ax_serving.py', 'build.json', 'capture_mixed_steps.py', 'check_flash_next_native_support.py',
    'mxfp4-inventory.json', 'qualify_qwen38_flash_next.py', 'run_fixed_extension.py',
    'run_fixed_extension_crossover.py', 'run_flash_next_lifecycle.py', 'run_flash_next_prefill_split_probe.py',
    'run_natural_crossover.py', 'run_natural_fork.py', 'run_natural_m5.sh', 'run_qwen38_reference_serving.py',
    'run_selective_mixed.py', 'run_warm_prefix.py', 'split_step_route.py', 'suffix-cases.json',
}
ARMS = ('off-1', 'on-1', 'on-2', 'off-2')
OBSERVED = {'observed/receipt.json'} | {
    f'observed/{name}/{filename}' for name in ARMS
    for filename in ('arm.json', 'partial.json', 'natural-samples.jsonl', 'natural.log')
}
AUX = {'host.json', 'inputs.sha256', 'software-gate.json', 'source.patch', 'status.txt'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_member(archive, name):
    member = archive.getmember(name)
    require(member.isfile(), f'{name} must be a regular file')
    with archive.extractfile(member) as stream:
        return stream.read()


def verify_oracle(files):
    receipt = json.loads(files['observed/receipt.json'])
    with tarfile.open(ORACLE_ARCHIVE) as archive:
        platform_anchor = json.loads(read_member(archive, 'host.json'))
        staged = json.loads(read_member(archive, 'observed/receipt.json'))
        cold = json.loads(read_member(archive, 'observed/off-1-arm.json'))
    with tarfile.open(EVICTION_ARCHIVE) as archive:
        revised_m2 = json.loads(read_member(archive, 'observed/receipt.json'))
    require(platform_anchor == json.loads(files['host.json'])
            and staged['manifest'] == revised_m2['manifest'] == receipt['manifest']
            and receipt['build'] == revised_m2['build'], 'Host, supported-tier runtime, or publisher data changed')
    for name in ARMS:
        data = json.loads(files[f'observed/{name}/arm.json'])
        require(receipt['arms'][name] == data and data['outputs'] == cold['outputs'],
                f'{name} differs from the raw arm or same-host cold oracle')


def replay(path):
    with tarfile.open(path) as archive, tarfile.open(CORPUS_ARCHIVE) as corpus_archive, TemporaryDirectory(
        prefix='flash-next-natural-crossover-'
    ) as temp:
        names = archive.getnames()
        require(set(names) == DRIVERS | OBSERVED | AUX and len(names) == len(DRIVERS | OBSERVED | AUX),
                'The exact archive file list changed')
        files = {name: read_member(archive, name) for name in names}
        files[CORPUS] = read_member(corpus_archive, CORPUS)
        pairs = [row.split() for row in files['inputs.sha256'].decode().splitlines()]
        manifest = {name: digest for digest, name in pairs}
        inputs = DRIVERS | {CORPUS}
        require(set(manifest) == inputs | {'ax-engine-server'} and len(pairs) == len(manifest)
                and all(manifest[name] == sha256(files[name]).hexdigest() for name in inputs),
                'The staged driver hashes changed')
        build = json.loads(files['build.json'])
        require(build['server_sha256'] == manifest['ax-engine-server']
                and build['patch_sha256'] == sha256(files['source.patch']).hexdigest()
                and build['codesign_verified'] is True and files['status.txt'].strip() == b'0',
                'Signed build, patch or terminal status changed')
        software = json.loads(files['software-gate.json'])
        require(software['runtime_patch_sha256'] == build['patch_sha256']
                and software['status'] == 0 and software['rust_failed'] == 0
                and software['rust_passed'] > 0 and software['python_passed'] > 0,
                'The software receipt does not match the signed runtime')
        host = json.loads(files['host.json'])
        require(host['hardware'] == 'MacBook Pro M5 Max 128 GB' and host['storage'] == 'internal SSD'
                and host['mlx'] == build['mlx'] == build['mlx_metal'] == '0.32.4',
                'The platform or MLX version changed')
        verify_oracle(files)
        stage = Path(temp)
        for name, body in files.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
        return subprocess.check_output([sys.executable, '-B', stage / 'run_natural_crossover.py',
                                        '--replay', stage / 'observed/receipt.json'])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence.tar.xz')
    sys.stdout.buffer.write(replay(parser.parse_args().archive))
