"""Recheck the bounded Flash Next runner MTP controls from the recorded decisions."""

import gzip
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(root, name):
    return json.loads((root / name).read_bytes())


def check_runs(record):
    assert record['schema'] == 'ax-engine.flash-next.runner-mtp-greedy-audit.v1'
    assert record['qualification'] is False
    accepted = rejected = steps = tokens = 0
    runs = record['runs']
    assert [(len(run['prompt']), run['budget']) for run in runs] == [(16, 64), (512, 64), (16, 2)]
    for run in runs:
        prompt_size = len(run['prompt'])
        assert run['prompt'] == list(range(1, prompt_size + 1))
        direct, mtp, budget = (run[key] for key in ('direct_tokens', 'mtp_tokens', 'budget'))
        assert direct == mtp and len(mtp) == budget
        cursor = 1
        for row in run['decisions']:
            assert row['position'] - prompt_size + 1 == cursor
            assert row['remaining'] == budget - cursor
            assert 0 < cursor < budget and mtp[cursor - 1] == row['primary']
            accept = row['remaining'] > 1 and row['draft'] == row['target']
            assert type(row['accepted']) is int and row['accepted'] == int(accept)
            emitted = [row['target'], row['next']] if accept else [row['target']]
            if not accept:
                assert row['next'] == row['target']
            assert emitted == row['emitted'] == mtp[cursor:cursor + len(emitted)]
            cursor += len(emitted)
            accepted += int(accept)
            rejected += int(row['draft'] is not None and not accept)
            steps += 1
        assert cursor == budget
        tokens += len(mtp)
    assert accepted == record['accepted'] == 55 and rejected == record['rejected'] == 15
    assert steps == 72 and tokens == 130
    return {'controls': len(runs), 'matched_direct_output_ids': tokens,
            'verified_steps': steps, 'accepted_drafts': accepted, 'rejected_drafts': rejected}


def replay(root=ROOT):
    audit_bytes = gzip.decompress((root / 'audit.json.gz').read_bytes())
    provenance = read(root, 'provenance.json')
    assert sha256(audit_bytes).hexdigest() == provenance['typed_result_sha256']
    pack = (root / 'pack.json').read_bytes()
    assert sha256(pack).hexdigest() == provenance['pack_before_sha256'] == provenance['pack_after_sha256']
    assert provenance['test_exit'] == 0 and provenance['pack_bindings_stable'] is True
    assert json.loads(pack)['members'] == 49
    build, launcher = read(root, 'build.json'), read(root, 'launcher-repair.json')
    assert build['test_sha256'] == launcher['binary_sha256']
    assert build['mlx'] == launcher['pinned_mlx'] == read(root, 'hardware.json')['mlx'] == '0.32.4'
    return {'scope': 'Bounded canonical-singleton greedy runner evidence', 'qualification': False,
            **check_runs(json.loads(audit_bytes))}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
