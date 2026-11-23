"""Replay the pinned M2 Flash Next restored-prefix suffix matrix."""

from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
SUFFIX_LENGTHS = [1, 16, 63, 64, 65, 512, 1024, 1025, 2047, 2048]
MEMBERS = {
    'build.json', 'inputs.sha256', 'pressure.jsonl', 'result.json', 'source.patch', 'status.txt',
    'suffix-cases.json', 'test.log',
}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def read(archive, name):
    member = archive.getmember(name)
    require(member.isfile(), f'Expected a regular member: {name}')
    with archive.extractfile(member) as stream:
        return stream.read()


def replay(path):
    with tarfile.open(path) as archive:
        require(set(archive.getnames()) == MEMBERS and len(archive.getnames()) == len(MEMBERS),
                'The suffix evidence file list changed')
        require(read(archive, 'status.txt') == b'0\n', 'The signed probe did not finish')
        report = json.loads(read(archive, 'result.json'))
        build = json.loads(read(archive, 'build.json'))
        fixture = read(archive, 'suffix-cases.json')
        checksums = {f: digest for digest,f in map(str.split,read(archive,'inputs.sha256').decode().splitlines())}
        require(report['build'] == build and build['codesign_verified'] is True
                and build['binary_sha256'] == checksums['ax-flash-prefix-suffix-test']
                and build['inventory_sha256'] == checksums['mxfp4-inventory.json']
                and sha256(fixture).hexdigest() == build['fixture_sha256'] == checksums['suffix-cases.json']
                and sha256(read(archive,'source.patch')).hexdigest() == build['source_patch_sha256'],
                'The signed image, source, pack inventory or fixture changed')
        prefix, suffixes = json.loads(fixture)
        require(len(prefix) == 2048 and list(map(len,suffixes)) == SUFFIX_LENGTHS
                and len({r[0] for r in suffixes}) == len(SUFFIX_LENGTHS),
                'The prefix or suffix fixtures lost a boundary')
        log = read(archive,'test.log').decode()
        encoded, = re.findall(r'FLASH_NEXT_PREFIX_SUFFIX_MATRIX (\{[^\n]+\})', log)
        probe = json.loads(encoded)
        require('test result: ok. 1 passed; 0 failed; 0 ignored' in log
                and probe == report['probe'] and report['qualification'] is False
                and probe['native_gdn_dispatches'] == report['native_gdn_dispatches'] > 0,
                'The native dispatch witness or test result changed')
        matrix = probe['observations']
        rows = matrix['rows']
        require(matrix['prefix_tokens'] == len(prefix) and matrix['prefix_width'] == 512
                and [r['suffix_tokens'] for r in rows] == SUFFIX_LENGTHS
                and [r['suffix_width'] for r in rows] == [1024]*9 + [512]
                and all(r['cold'] == r['warm'] and len(r['warm']['output_ids']) == 5 for r in rows)
                and len({r['warm']['post_prefill_sha256'] for r in rows}) == len(rows),
                'Cold and rehydrated states disagree or miss a suffix boundary')
        pressure = [json.loads(v) for v in read(archive,'pressure.jsonl').splitlines()]
        require(len(pressure) == report['normal_pressure_samples'] > 0
                and all(row['pressure'] == 1 for row in pressure), 'Pressure trace changed')
        return report


if __name__ == '__main__':
    print(json.dumps(replay(ROOT/'evidence.tar.xz'), indent=2, sort_keys=True, allow_nan=False))
