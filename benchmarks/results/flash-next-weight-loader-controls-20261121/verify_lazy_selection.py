"""Check selected-byte parity and excluded payload residency through the native probe."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from run_load_order import cache_state, fresh_clone

MODES = ('name-order', 'offset-order', 'lazy-cpu')
MIB = 1024**2


def read_headers(path):
    with path.open('rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
    return {name: spec for name, spec in header.items() if name != '__metadata__'}


def run_probe(binary, directory, mode, rows):
    plan = directory / 'control-plan.json'
    plan.write_text(json.dumps({'files': rows}))
    subprocess_result = subprocess.run([binary, plan, mode], check=True, text=True,
                                       capture_output=True, timeout=45)
    result = json.loads(subprocess_result.stdout)
    assert result['mode'] == mode
    return result['files']


def compare_pack(binary, directory, root):
    minimum = {}
    for path in sorted(root.glob('*.safetensors')):
        for name, spec in read_headers(path).items():
            low, high = spec['data_offsets']
            width = high - low
            dtype = spec['dtype']
            if width and (dtype not in minimum or width < minimum[dtype][0]):
                minimum[dtype] = (width, path, name)
    assert minimum
    assert sum(row[0] for row in minimum.values()) < 16 * MIB
    selection = defaultdict(list)
    for width, path, name in minimum.values():
        selection[str(path)].append((name, width))
    rows = [{'path':path,'names':sorted(name for name, _ in items),
             'selected_bytes':sum(width for _,width in items)} for path,items in sorted(selection.items())]
    observations = {}
    for mode in MODES:
        observed = run_probe(binary, directory, mode, rows)
        for obs, target in zip(observed, rows, strict=True):
            assert obs['selected_bytes'] == target['selected_bytes'] and obs['tensors'] == len(target['names'])
        observations[mode] = {row['file']: row['sha256'] for row in observed}
    assert observations['name-order'] == observations['offset-order'] == observations['lazy-cpu']
    return {'selected':{dtype:{'file':path.name,'name':name,'bytes':width}
                        for dtype,(width,path,name) in sorted(minimum.items())},
            'file_sha256_by_mode':observations,'total_selected_bytes':sum(row[0] for row in minimum.values())}


def excluded_fixture(directory):
    path = directory / 'original.safetensors'
    selected_bytes = 4 * 4096
    gap_bytes = 16 * MIB
    expert_bytes = 512 * MIB
    sentinel_bytes = 16 * MIB
    expert_begin = selected_bytes + gap_bytes
    sentinel_begin = expert_begin + expert_bytes
    header = {
        'a.packed.weight':{'dtype':'U32','shape':[4096],'data_offsets':[0,selected_bytes]},
        'b.gap':{'dtype':'U8','shape':[gap_bytes],'data_offsets':[selected_bytes,expert_begin]},
        'c.offloaded_experts.weight':{'dtype':'U32','shape':[256,1024,512],
                                     'data_offsets':[expert_begin,sentinel_begin]},
        'd.sentinel':{'dtype':'U8','shape':[sentinel_bytes],
                      'data_offsets':[sentinel_begin,sentinel_begin+sentinel_bytes]},
    }
    encoded = json.dumps(header,sort_keys=True).encode()
    data_base = 8 + len(encoded)
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(encoded)))
        f.write(encoded)
        f.write(bytes(range(256)) * (selected_bytes // 256))
        for offset in (64 * MIB, 128 * MIB, 256 * MIB):
            f.seek(data_base + expert_begin + offset)
            f.write(bytes([0x5A]) * MIB)
        f.seek(data_base + sentinel_begin)
        f.write(bytes([0xAC]) * sentinel_bytes)
    return path


def compare_exclusion(binary, directory):
    source = excluded_fixture(directory)
    results = []
    digests = set()
    for mode in MODES:
        clone = directory / f'{mode}.safetensors'
        fresh_clone(source, clone)
        selected = {'path':str(clone),'names':['a.packed.weight']}
        excluded = {'path':str(clone),'names':['c.offloaded_experts.weight']}
        before = cache_state(excluded)
        raw, = run_probe(binary,directory,mode,[selected])
        after = cache_state(excluded)
        assert raw['selected_bytes'] == 4 * 4096
        assert before['selected_pages'] == after['selected_pages'] > 0
        assert before['resident_pages'] == after['resident_pages'] == 0
        digests.add(raw['sha256'])
        results.append({'mode':mode,'excluded_before':before,'excluded_after':after,'sha256':raw['sha256']})
    assert len(digests) == 1
    positive = directory / 'positive.safetensors'
    fresh_clone(source, positive)
    selected = {'path':str(positive),'names':['d.sentinel']}
    before = cache_state(selected)
    raw, = run_probe(binary,directory,'lazy-cpu',[selected])
    after = cache_state(selected)
    assert raw['selected_bytes'] == 16 * MIB and before['resident_pages'] <= 1
    assert after['resident_pages'] > 0
    return {'excluded_expert_bytes':512*MIB,'controls':results,
            'positive_sentinel_before':before,'positive_sentinel_after':after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary','mxfp4','mxfp8','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        record = {'formats':{label:compare_pack(args.binary,root,path)
                             for label,path in [('mxfp4',args.mxfp4),('mxfp8',args.mxfp8)]},
                  'excluded_fixture':compare_exclusion(args.binary,root)}
    args.output.write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
