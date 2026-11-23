"""Replay the signed Flash Next prefix extension from a plain-weight model directory."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
SERVER = 'ax-engine-server'
CORPUS = 'mixed-corpus.jsonl'
CORPUS_ARCHIVE = ROOT.parent / 'flash-next-mxfp4-mixed-serving-m5-20261121/evidence.tar.xz'
DRIVERS = {
    'bench_ax_serving.py', 'build.json', 'capture_mixed_steps.py', 'check_flash_next_native_support.py',
    'mxfp4-inventory.json', 'qualify_qwen38_flash_next.py', 'run_fixed_extension.py',
    'run_flash_next_lifecycle.py', 'run_flash_next_prefill_split_probe.py', 'run_qwen38_reference_serving.py',
    'run_selective_mixed.py', 'run_warm_prefix.py', 'split_step_route.py',
}
OBSERVED = {'observed/receipt.json'} | {
    f'observed/{arm}{suffix}' for arm in ('off', 'on') for suffix in
    ('-partial.json', '-samples.jsonl', '-seed.json', '.log')
}
AUXILIARY = {'artifact-address.json', 'inputs.sha256', 'replay_extension.py',
             'run_fixed_extension_crossover.py', 'source.patch', 'status.txt'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contents(archive, filename):
    member = archive.getmember(filename)
    require(member.isfile(), f'{filename} must be a regular file')
    with archive.extractfile(member) as stream:
        return stream.read()


def replay(path):
    with tarfile.open(path) as archive, TemporaryDirectory(prefix='flash-next-fixed-prefix-m2-') as temp:
        required = DRIVERS | OBSERVED | AUXILIARY
        names = archive.getnames()
        require(set(names) == required and len(names) == len(required), 'The exact archive file list changed')
        files = {name: contents(archive, name) for name in names}
        with tarfile.open(CORPUS_ARCHIVE) as corpus_archive:
            files[CORPUS] = contents(corpus_archive, CORPUS)
        pairs = [line.split() for line in files['inputs.sha256'].decode().splitlines()]
        manifest = {name: digest for digest, name in pairs}
        require(set(manifest) == DRIVERS | {SERVER, CORPUS} and len(pairs) == len(manifest)
                and all(manifest[name] == sha256(files[name]).hexdigest() for name in DRIVERS | {CORPUS}),
                'The staged inputs no longer match the original checksum table')
        build = json.loads(files['build.json'])
        require(build['server_sha256'] == manifest[SERVER] and build['codesign_verified'] is True,
                'The signed binary is not bound to the staged input checksums')
        address = json.loads(files['artifact-address.json'])
        require(address['tensor_files'] == 26 and address['missing_content_addresses'] == 26
                and address['prefix_fingerprint'] is None
                and address['model_manifest_sha256'] == json.loads(files['observed/receipt.json'])['manifest']['model-manifest.json'],
                'The run no longer describes all 26 ordinary tensor files without a durable address')
        stage = Path(temp)
        for name, body in files.items():
            output = stage / name
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(body)
        return subprocess.check_output([
            sys.executable, '-B', str(stage / 'replay_extension.py'), 'm2', str(stage),
            '--expected-build', str(stage / 'build.json'), '--patch', str(stage / 'source.patch'),
        ])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence.tar.xz')
    sys.stdout.buffer.write(replay(parser.parse_args().archive))
