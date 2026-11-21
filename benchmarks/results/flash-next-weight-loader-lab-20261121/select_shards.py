"""Choose fixed complete-tensor samples from actual safetensors shards."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--resident-source', type=Path)
parser.add_argument('--shards', type=int, default=3)
parser.add_argument('--max-bytes', type=int, default=1536 * 1024**2)
args = parser.parse_args()
selected_by_file = defaultdict(set)
if args.resident_source:
    code = args.resident_source.read_text().split('fn is_trunk_resident_role', 1)[1].split('\n}', 1)[0]
    def snake(name):
        return re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)).lower()
    roles = {snake(name) for name in re.findall(r'NativeTensorRole::(\w+)', code)}
    specs = json.loads((args.root / 'model-manifest.json').read_bytes())['tensors']
    weight_map = json.loads((args.root / 'model.safetensors.index.json').read_bytes())['weight_map']
    assert {x['role'] for x in specs} - roles == {'ngram_embedding', 'other'}
    for spec in specs:
        if spec['role'] not in roles:
            continue
        name, file = spec['name'], spec['file']
        selected_by_file[file].add(name)
        if name.endswith('.weight'):
            for suffix in ('.scales', '.biases', '.bias'):
                companion = name.removesuffix('.weight') + suffix
                selected_by_file[weight_map.get(companion, file)].add(companion)

shards = []
for path in sorted(args.root.glob('model-*.safetensors')):
    with path.open('rb') as file:
        size = int.from_bytes(file.read(8), 'little')
        header = json.loads(file.read(size))
    header.pop('__metadata__', None)
    eligible = set(header) & selected_by_file[path.name] if args.resident_source else set(header)
    # Stable dispersion across rows and across large and small projections.
    candidates = sorted(eligible)
    random.Random(f'flash-load-order-v1:{path.name}').shuffle(candidates)
    selected = []
    size = 0
    for name in candidates:
        start, end = header[name]['data_offsets']
        if end > start and size + end - start <= args.max_bytes:
            selected.append(name)
            size += end - start
    selected.sort()
    ranges = [header[name]['data_offsets'] for name in selected]
    backwards = sum(b[0] < a[1] for a, b in zip(ranges, ranges[1:]))
    if backwards and size >= args.max_bytes // 2:
        shards.append({'path': str(path.resolve()), 'names': selected, 'selected_bytes': size,
                       'backward_jumps': backwards})
shards.sort(key=lambda row: (-row['backward_jumps'], row['path']))
assert len(shards) >= args.shards, 'too few actual interleaved shards'
args.output.write_text(json.dumps({'files': shards[:args.shards]}, indent=2) + '\n')
