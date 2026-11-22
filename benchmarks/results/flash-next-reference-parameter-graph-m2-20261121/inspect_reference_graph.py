"""Inventory the pinned reference's lazy parameter graph before weight evaluation."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path


REVISION = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
REFERENCE_COMMIT = '6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb'
LANGUAGE_SHA256 = '30f059d6289451bde0f5ef816090ff946ebc857d1b557f83415e2ba867d76199'
LOADER_SHA256 = '9a914b1709e2e7113bed408987bf8557cfccff1284646de5fa16731220435564'
INVENTORY_SHA256 = '97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6'
MEMBERS = {
    'config.json': 'fb3c637ce4a240453c454fbc1211fa1e893d0af12df54a9df19c9c017fa59237',
    'model.safetensors.index.json': '37344288777e7e78b35901cb1d572dad0dc640cb9cedb90024860b932a657696',
}
VERSIONS = {'mlx': '0.32.2', 'mlx-metal': '0.32.2', 'mlx-vlm': '0.7.0rc0',
            'transformers': '5.18.0.dev0', 'tokenizers': '0.23.2'}
BATCH = 32
EXPECTED_ARRAYS = 2326


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def summarize(rows):
    names = [row['name'] for row in rows]
    if len(set(names)) != len(names) or any(row['logical_bytes'] < 0 for row in rows):
        raise ValueError('Duplicate parameter or invalid size')
    batches = []
    logical_total = 0
    for start in range(0, len(rows), BATCH):
        group = rows[start:start + BATCH]
        size = sum(row['logical_bytes'] for row in group)
        ple = sum(row['logical_bytes'] for row in group if '.ngram_embedding.' in row['name'])
        logical_total += size
        batches.append({'index': start // BATCH, 'arrays': len(group), 'logical_bytes': size,
                        'ple_ngram_logical_bytes': ple, 'cumulative_logical_bytes': logical_total})
    ple_arrays = [row for row in rows if '.ngram_embedding.' in row['name']]
    ple_bytes = sum(row['logical_bytes'] for row in ple_arrays)
    return {'array_count': len(rows), 'logical_bytes': logical_total, 'ple_ngram_arrays': len(ple_arrays),
            'ple_ngram_logical_bytes': ple_bytes, 'other_logical_bytes': logical_total - ple_bytes,
            'batches': batches}


def inspect(root, inventory_file):
    observed_versions = {name: version(name) for name in VERSIONS}
    if observed_versions != VERSIONS:
        raise ValueError('Pinned runtime packages changed')
    if digest(inventory_file) != INVENTORY_SHA256:
        raise ValueError('Pinned inventory bytes changed')
    inventory = json.loads(inventory_file.read_bytes())
    if inventory['revision'] != REVISION or len(inventory['members']) != 49:
        raise ValueError('Pinned inventory identity changed')
    if {name: digest(root / name) for name in MEMBERS} != MEMBERS:
        raise ValueError('Pinned pack metadata changed')
    for member in inventory['members']:
        if (root / member['name']).stat().st_size != member['size']:
            raise ValueError(f"Pinned member size changed: {member['name']}")

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import language
    from mlx_vlm import utils

    if digest(language.__file__) != LANGUAGE_SHA256 or digest(utils.__file__) != LOADER_SHA256:
        raise ValueError('Pinned reference language or loader implementation changed')
    before = mx.get_active_memory()
    model = utils.load_model(root, lazy=True, strict=True).language_model
    rows = [{'name': name, 'logical_bytes': array.nbytes, 'shape': list(array.shape),
             'dtype': str(array.dtype)} for name, array in tree_flatten(model.parameters())]
    after = mx.get_active_memory()
    summary = summarize(rows)
    if summary['array_count'] != EXPECTED_ARRAYS or summary['ple_ngram_arrays'] == 0:
        raise ValueError('Unexpected pinned graph shape')
    if max(before, after, mx.get_peak_memory()) >= 64 << 20:
        raise ValueError('Lazy graph inventory exceeded its allocation envelope')
    return {'schema': 'ax-engine.flash-next.reference-parameter-graph.v1', 'pack_revision': REVISION,
            'pack_metadata_sha256': MEMBERS, 'inventory_sha256': INVENTORY_SHA256,
            'reference_commit': REFERENCE_COMMIT, 'loader_sha256': LOADER_SHA256,
            'versions': observed_versions, 'language_sha256': LANGUAGE_SHA256,
            'scope': 'Logical parameter bytes before weight evaluation; diagnostic only',
            'mlx_active_before_bytes': before, 'mlx_active_after_bytes': after,
            'mlx_peak_bytes': mx.get_peak_memory(), 'parameters': rows, 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    receipt = inspect(args.snapshot, args.inventory)
    with args.output.open('x') as file:
        json.dump(receipt, file, indent=2, allow_nan=False)
        file.write('\n')
