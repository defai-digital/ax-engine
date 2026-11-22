"""Replay the matched fused-gather QSA state and timing check."""

import gzip
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def replay(root=ROOT):
    receipt = json.loads((root / 'receipt.json').read_bytes())
    result = receipt['result']
    assert receipt['status'] == 'passed' and receipt['test_exit'] == 0
    assert result['final_logits_exact'] is True and result['final_state_exact'] is True
    assert (result['prompt_tokens'], result['chunk_size']) == (8192, 512)

    stdout = gzip.decompress((root / 'test-stdout.txt.gz').read_bytes()).decode().splitlines()
    assert [json.loads(line) for line in stdout if line.startswith('{')] == [result]
    assert any(line.startswith('test result: ok. 1 passed; 0 failed;') for line in stdout)
    pressure = [json.loads(line) for line in gzip.decompress(
        (root / 'state-samples.jsonl.gz').read_bytes()).splitlines()]
    assert len(pressure) == receipt['memory_samples'] == 105
    assert all(sample['pressure'] == 1 for sample in pressure)
    assert max(sample['footprint_bytes'] for sample in pressure) == receipt['peak_bytes']

    build_bytes = (root / 'build.json').read_bytes()
    build = receipt['build']
    assert build == json.loads(build_bytes)
    bound = {filename: digest for digest, filename in (
        line.split() for line in (root / 'inputs.sha256').read_text().splitlines())}
    assert sha256(build_bytes).hexdigest() == bound['build.json']
    assert build['test_binary_sha256'] == bound['flash-next-qsa-state-test']
    assert receipt['inventory_sha256'] == bound['mxfp4-inventory.json']

    samples = result['samples']
    assert [row['index_mode'] for row in samples] == ['separate', 'shared', 'shared', 'separate']
    assert all(row['fused_takes'] == 864 for row in samples)
    assert [row['index_tiles'] for row in samples] == [864, 432, 432, 864]
    assert all(math.isfinite(row['seconds']) and row['seconds'] > 0 for row in samples)
    assert all(row['peak_bytes'] >= row['active_bytes_before'] > 0 for row in samples)

    return {
        'final_logits_and_state_exact': True,
        'shared_index_tile_reduction_percent': 50,
        'normal_pressure_samples': len(pressure),
        'pairs': [{
            'separate_seconds': separate['seconds'],
            'shared_seconds': shared['seconds'],
            'shared_duration_change_percent': round(100 * (shared['seconds'] / separate['seconds'] - 1), 2),
            'shared_peak_change_bytes': shared['peak_bytes'] - separate['peak_bytes'],
        } for separate, shared in ((samples[0], samples[1]), (samples[3], samples[2]))],
        'runtime_promotion_supported': False,
    }


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
