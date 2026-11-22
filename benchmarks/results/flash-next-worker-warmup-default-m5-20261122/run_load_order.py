"""Record per-file selected-byte identity, load time and cache state."""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import mmap
import os
from pathlib import Path
import platform
import subprocess
import tempfile


MODES = ('name-order', 'offset-order', 'lazy-cpu')
PAGE = os.sysconf('SC_PAGESIZE')
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
LIBC.mincore.restype = ctypes.c_int
LIBC.clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
LIBC.clonefile.restype = ctypes.c_int


def fresh_clone(source, scratch):
    result = LIBC.clonefile(os.fsencode(source), os.fsencode(scratch), 0)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def cache_state(row):
    path = Path(row['path'])
    with path.open('rb') as source:
        raw_size = int.from_bytes(source.read(8), 'little')
        header = json.loads(source.read(raw_size))
        start = 8 + raw_size
        pages = set()
        for name in row['names']:
            low, high = header[name]['data_offsets']
            if low < high:
                pages.update(range((start + low) // PAGE, (start + high - 1) // PAGE + 1))
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY) as pages_map:
            vector = ctypes.create_string_buffer((len(pages_map) + PAGE - 1) // PAGE)
            address = ctypes.c_char.from_buffer(pages_map)
            result = LIBC.mincore(ctypes.addressof(address), len(pages_map), vector)
            del address
            if result != 0:
                return {'sample_error': ctypes.get_errno()}
            flags = vector.raw
    return {'selected_pages': len(pages), 'resident_pages': sum(flags[page] & 1 for page in pages)}


def hardware():
    return {'soc': subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            'memory_bytes': int(subprocess.check_output(['/usr/sbin/sysctl','-n','hw.memsize'],text=True)),
            'os': platform.mac_ver()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('workload', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--storage', required=True)
    parser.add_argument('--mlx-lib-dir', type=Path)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--fresh-clones', action='store_true')
    parser.add_argument('--scratch-dir', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.workload.read_bytes())['files']
    assert plan and args.passes > 0
    trial_env = dict(os.environ)
    if args.mlx_lib_dir is not None:
        trial_env['DYLD_LIBRARY_PATH'] = str(args.mlx_lib_dir)
    timestamps = lambda: datetime.now(timezone.utc).isoformat(timespec='seconds')
    with tempfile.TemporaryDirectory(dir=args.scratch_dir) as directory, args.output.open('x') as evidence:
        print(json.dumps({'kind':'host','hardware':hardware(),'storage':args.storage,'start':timestamps(),
                          'page_size': PAGE, 'pass_count':args.passes,
                          'fresh_clones':args.fresh_clones}),file=evidence,flush=True)
        for pass_index in range(args.passes):
            order = range(len(plan)) if pass_index % 2 == 0 else reversed(range(len(plan)))
            for file_index in order:
                row = plan[file_index]
                work = Path(directory) / 'workload.json'
                start = (file_index + pass_index) % len(MODES)
                modes = MODES[start:] + MODES[:start]
                expected_digest = None
                for mode in modes:
                    sample = dict(row)
                    clone = Path(directory) / 'sample.safetensors'
                    if args.fresh_clones:
                        fresh_clone(row['path'], clone)
                        sample['path'] = str(clone)
                    try:
                        work.write_text(json.dumps({'files':[sample]}))
                        before = cache_state(sample)
                        result = subprocess.run([args.binary, work, mode], env=trial_env, timeout=240, capture_output=True, text=True)
                    finally:
                        if args.fresh_clones:
                            clone.unlink()
                    if result.returncode:
                        raise RuntimeError(f'load {mode} for file {file_index} failed: {result.returncode}')
                    data, = json.loads(result.stdout)['files']
                    data['file'] = Path(row['path']).name
                    if expected_digest is None:
                        expected_digest = data['sha256']
                    if data['sha256'] != expected_digest or data['selected_bytes'] != row['selected_bytes']:
                        raise RuntimeError('Loader path materialized different tensor bytes.')
                    print(json.dumps({'kind':'load', 'pass':pass_index,'file_index':file_index,'mode':mode,
                                      'before':before,'name_order_backward_jumps':row['backward_jumps'],
                                      **data}),file=evidence,flush=True)
        print(json.dumps({'kind':'complete','finish':timestamps()}),file=evidence,flush=True)


if __name__ == '__main__':
    main()
