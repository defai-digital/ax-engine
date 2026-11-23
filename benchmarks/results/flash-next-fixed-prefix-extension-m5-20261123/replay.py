"""Replay the signed Flash Next prefix extension on the primary M5 Max."""

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
M2_ARCHIVE = ROOT.parent / 'flash-next-fixed-prefix-plain-layout-m2-20261123/evidence.tar.xz'
DRIVERS = {
    'bench_ax_serving.py', 'build.json', 'capture_mixed_steps.py', 'check_flash_next_native_support.py',
    'mxfp4-inventory.json', 'qualify_qwen38_flash_next.py', 'run_fixed_extension.py',
    'run_flash_next_lifecycle.py', 'run_flash_next_prefill_split_probe.py', 'run_qwen38_reference_serving.py',
    'run_selective_mixed.py', 'run_warm_prefix.py', 'split_step_route.py',
    'run_fixed_extension_crossover.py', 'run_fixed_extension_m5.sh',
}
OBSERVED = {'observed/receipt.json'} | {
    f'observed/{arm}{suffix}' for arm in ('off-1', 'on-1', 'on-2', 'off-2') for suffix in
    ('-partial.json', '-samples.jsonl', '-seed.json', '-arm.json', '.log')
}
AUXILIARY = {'m2-admission.json', 'host.json', 'inputs.sha256', 'replay_extension.py',
             'source.patch', 'status.txt'}


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
        admission = json.loads(files['m2-admission.json'])
        host = json.loads(files['host.json'])
        require(host['hardware'] == 'MacBook Pro M5 Max 128 GB' and host['storage'] == 'internal SSD'
                and host['macos'] == '27.2.1', 'The comparison ran outside the bound primary host')
        with tarfile.open(M2_ARCHIVE) as m2_archive:
            m2_receipt = contents(m2_archive, 'observed/receipt.json')
        require(admission['comparison_receipt_sha256'] == sha256(m2_receipt).hexdigest()
                and json.loads(m2_receipt)['build'] == build
                and all(admission[key] == build[key] for key in ('server_sha256','patch_sha256')),
                'The primary executable and M2 functional admission no longer match')
        stage = Path(temp)
        for name, body in files.items():
            output = stage / name
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(body)
        return subprocess.check_output([
            sys.executable, '-B', str(stage / 'replay_extension.py'), 'm5', str(stage),
            '--expected-build', str(stage / 'build.json'), '--patch', str(stage / 'source.patch'),
        ])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence.tar.xz')
    sys.stdout.buffer.write(replay(parser.parse_args().archive))
