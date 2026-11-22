"""Measure publisher checksum cache residency on fresh APFS clones."""

import argparse
import ctypes
from contextlib import closing, nullcontext
import fcntl
from hashlib import sha256
import json
import mmap
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from time import monotonic


INVENTORY_SHA256 = '97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6'
REVISION = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
SHARD = 'model-00001-of-00025.safetensors'
ORDER = ('buffered', 'no-cache', 'no-cache', 'buffered')
CHUNK = 4 * 1024 * 1024
PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')


def darwin_calls():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    libc.clonefile.restype = ctypes.c_int
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    libc.mincore.restype = ctypes.c_int
    return libc


def clone(libc, source, target):
    if libc.clonefile(os.fsencode(source), os.fsencode(target), 0) != 0:
        raise OSError(ctypes.get_errno(), 'APFS clone failed')


def resident_pages(libc, file):
    with file.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as data:
        pages = (len(data) + PAGE_SIZE - 1) // PAGE_SIZE
        flags = ctypes.create_string_buffer(pages)
        start = ctypes.c_char.from_buffer(data)
        result = libc.mincore(ctypes.addressof(start), len(data), flags)
        del start
        if result != 0:
            raise OSError(ctypes.get_errno(), 'mincore failed')
        return sum(flag & 1 for flag in flags.raw), pages


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def pressure():
    return int(subprocess.check_output(
        ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))


def read_chunks(stream, buffer):
    if buffer is None:
        while chunk := stream.read(CHUNK):
            yield chunk
        return
    with memoryview(buffer) as view:
        while count := stream.readinto(view):
            with view[:count] as chunk:
                yield chunk


def checksum(file, mode, expected_size, expected_sha256, sample=pressure, *, aligned=False):
    if mode not in ORDER:
        raise ValueError('Unknown checksum mode')
    samples = [sample()]
    digest = sha256()
    total = 0
    next_sample = 1 << 30
    alignment = mmap.mmap(-1, CHUNK) if aligned else nullcontext(None)
    with alignment as buffer, file.open('rb', buffering=0 if aligned else -1) as stream:
        original = identity(os.fstat(stream.fileno()))
        if mode == 'no-cache':
            fcntl.fcntl(stream.fileno(), fcntl.F_NOCACHE, 1)
        if samples[-1] != 1:
            raise ValueError('Memory pressure is above normal before the checksum')
        started = monotonic()
        with closing(read_chunks(stream, buffer)) as chunks:
            for chunk in chunks:
                digest.update(chunk)
                total += len(chunk)
                if total >= next_sample:
                    samples.append(sample())
                    next_sample += 1 << 30
                    if samples[-1] != 1:
                        raise ValueError('Memory pressure interrupted the checksum')
        elapsed = monotonic() - started
        final = identity(os.fstat(stream.fileno()))
    if original != final or final != identity(file.stat()):
        raise ValueError('Checksum input changed during the read')
    if total != expected_size or digest.hexdigest() != expected_sha256:
        raise ValueError('Publisher checksum mismatch')
    samples.append(sample())
    if samples[-1] != 1:
        raise ValueError('Memory pressure is above normal after the checksum')
    return {'bytes': total, 'sha256': digest.hexdigest(), 'checksum_seconds': elapsed,
            'sampled_pressure': samples}


def pinned_member(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != INVENTORY_SHA256:
        raise ValueError('Pinned inventory checksum mismatch')
    inventory = json.loads(raw)
    if inventory['revision'] != REVISION or len(inventory['members']) != 49:
        raise ValueError('Pinned inventory identity mismatch')
    member, = [row for row in inventory['members'] if row['name'] == SHARD]
    if member['sha256'] != member['publisher_sha256']:
        raise ValueError('Missing publisher checksum binding')
    return member


def run(source, inventory, *, aligned=False):
    member = pinned_member(inventory)
    backing = source / SHARD
    before = identity(backing.stat())
    if before[2] != member['size']:
        raise ValueError('Publisher member size mismatch')
    libc = darwin_calls()
    trials = []
    with tempfile.TemporaryDirectory(prefix='ax-flash-checksum-clones-', dir='/private/tmp') as directory:
        for number, mode in enumerate(ORDER):
            path = Path(directory) / f'proof-{number}.safetensors'
            clone(libc, backing, path)
            try:
                cold, pages = resident_pages(libc, path)
                measured = checksum(path, mode, member['size'], member['sha256'], aligned=aligned)
                warm, after_pages = resident_pages(libc, path)
                if pages != after_pages or path.stat().st_ino == before[1]:
                    raise ValueError('Unexpected clone geometry')
                trials.append({'number': number, 'mode': mode, 'file_pages': pages,
                               'resident_pages_before': cold, 'resident_pages_after': warm, **measured})
            finally:
                path.unlink()
    if identity(backing.stat()) != before:
        raise ValueError('Source identity changed during clone checks')
    return {'scope': 'Publisher SHA-256 and file-page residency only', 'pack_revision': REVISION,
            'inventory_sha256': INVENTORY_SHA256, 'member': SHARD, 'page_size': PAGE_SIZE,
            'read_chunk_bytes': CHUNK, 'aligned_user_buffer': aligned, 'hardware': 'Mac Studio M2 Ultra 192 GB, internal SSD',
            'macos': platform.mac_ver()[0], 'trials': trials}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pack', 'inventory', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--aligned-buffer', action='store_true')
    args = parser.parse_args()
    record = run(args.pack, args.inventory, aligned=args.aligned_buffer)
    with args.output.open('x') as file:
        print(json.dumps(record, indent=2, allow_nan=False), file=file)
