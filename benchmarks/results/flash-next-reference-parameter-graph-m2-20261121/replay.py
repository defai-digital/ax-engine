"""Recalculate the pinned reference parameter inventory without MLX or weights."""

from collections import Counter
import gzip
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from inspect_reference_graph import (EXPECTED_ARRAYS, INVENTORY_SHA256, LANGUAGE_SHA256,
                                     LOADER_SHA256, MEMBERS, REFERENCE_COMMIT,
                                     REVISION, VERSIONS, summarize)


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT.parent / 'qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json'
SIGNED = ROOT.parent / 'qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json'
DTYPE_BYTES = {'mlx.core.bfloat16': 2, 'mlx.core.uint32': 4, 'mlx.core.uint8': 1, 'mlx.core.int64': 8}


def check_parameters(rows):
    if len(rows) != EXPECTED_ARRAYS:
        raise ValueError('Unexpected parameter count')
    for row in rows:
        shape = row['shape']
        if (not shape or any(type(dim) is not int or dim <= 0 for dim in shape)
                or row['logical_bytes'] != prod(shape) * DTYPE_BYTES[row['dtype']]):
            raise ValueError(f"Invalid parameter geometry for {row['name']}")
    suffixes = Counter(row['name'].rsplit('.', 1)[1] for row in rows if '.ngram_embedding.' in row['name'])
    if suffixes != {'weight': 128, 'scales': 128, 'biases': 128}:
        raise ValueError('Unexpected n-gram parameter coverage')


def replay(root=ROOT):
    raw = gzip.decompress((root / 'graph.json.gz').read_bytes())
    graph = json.loads(raw)
    inventory = INVENTORY.read_bytes()
    signed = json.loads(SIGNED.read_bytes())
    assert graph['reference_commit'] == signed['reference']['commit'] == REFERENCE_COMMIT
    assert graph['language_sha256'] == signed['language_sha256'] == LANGUAGE_SHA256
    assert graph['versions'] == signed['versions'] == VERSIONS
    assert graph['loader_sha256'] == LOADER_SHA256
    assert graph['pack_revision'] == signed['revision'] == REVISION
    assert graph['inventory_sha256'] == sha256(inventory).hexdigest() == INVENTORY_SHA256
    hashes = {row['name']: row['sha256'] for row in json.loads(inventory)['members']}
    assert graph['pack_metadata_sha256'] == MEMBERS == {name: hashes[name] for name in MEMBERS}
    assert 0 <= graph['mlx_active_before_bytes'] <= graph['mlx_active_after_bytes'] <= graph['mlx_peak_bytes'] < 64 << 20
    check_parameters(graph['parameters'])
    summary = summarize(graph['parameters'])
    assert summary == graph['summary']
    assert summary['array_count'] == signed['weights_materialized']
    batches = summary.pop('batches')
    with_ple = [batch['index'] for batch in batches if batch['ple_ngram_logical_bytes'] > 0]
    return {
        'graph_sha256': sha256(raw).hexdigest(), **summary,
        'materialization_batches': len(batches),
        'batches_with_ple_ngram': with_ple,
        'largest_logical_batch_bytes': max(batch['logical_bytes'] for batch in batches),
        'metadata_probe_mlx_peak_bytes': graph['mlx_peak_bytes'],
        'signed_ordinary_run_peak_mlx_bytes': signed['peak_mlx_memory_bytes'],
    }


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
