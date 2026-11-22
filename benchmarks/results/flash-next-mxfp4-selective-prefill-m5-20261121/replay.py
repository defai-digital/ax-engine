"""Validate and replay the six-arm Flash Next selective-prefill archive."""
import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / 'flash-next-mxfp4-mixed-serving-m5-20261121' / 'replay.py'
ARMS = ('off-1', 'all-1', 'long-1', 'long-2', 'all-2', 'off-2')


def load_module(path, name):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Missing replay source {path.name}')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_holdouts(root, binary):
    inventory = sha256((root / 'mxfp4-inventory.json').read_bytes()).hexdigest()
    for length in (2048, 8192):
        folder = root / f'holdout-{length}'
        raw = (folder / 'result.json').read_bytes()
        body = json.loads(raw)
        if body['inventory_sha256'] != inventory:
            raise ValueError('Holdout inventory differs')
        computed = subprocess.check_output([
            sys.executable, '-B', str(root / 'verify_flash_next_prefill_split.py'),
            str(folder / 'result.json'), '--server-sha256', binary,
        ])
        if computed != (folder / 'audit.json').read_bytes():
            raise ValueError(f'Holdout replay differs at {length} tokens')


def replay(bundle, common):
    unpacker = load_module(common, 'mixed_archive_unpack')
    with TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = unpacker.unpack(bundle, root)
        if metadata['partial'] is not False or tuple(metadata['arm_order']) != ARMS:
            raise ValueError('A partial selective campaign cannot be replayed as complete')
        build = json.loads((root / 'build.json').read_bytes())
        if metadata['binary_sha256'] != build['server_sha256']:
            raise ValueError('Archive binary provenance differs from its build')
        validate_holdouts(root, build['server_sha256'])
        prior = metadata['original_client_sha256']['solo-reference.json']
        if 'base_url' in json.loads((root/'solo-reference.json').read_bytes())['target']:
            raise ValueError('The archived solo client should already be redacted')
        program = (
            'import sys; sys.path.insert(0, sys.argv[1]); '
            'import audit_selective_mixed; '
            'audit_selective_mixed.main(original_solo_sha256=sys.argv[2])'
        )
        completed = subprocess.run(
            [sys.executable, '-I', '-B', '-c', program, str(root), prior],
            capture_output=True, text=True,
        )
        if completed.returncode:
            details = completed.stderr.splitlines()
            raise ValueError(details[-1] if details else 'The selective audit failed')
        text = completed.stdout
        if text.encode() != (root / 'final-audit.json').read_bytes():
            raise ValueError('Replay differs from the terminal original metrics')
        return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT)
    parser.add_argument('--common-replay', type=Path, default=COMMON)
    args = parser.parse_args()
    print(replay(args.bundle, args.common_replay), end='')
