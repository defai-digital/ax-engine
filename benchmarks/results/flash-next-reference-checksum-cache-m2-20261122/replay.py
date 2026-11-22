"""Verify the pinned shard checksums and APFS page-residency comparisons."""

import json
from math import isfinite
from pathlib import Path
from statistics import median

from hash_cache_probe import CHUNK, INVENTORY_SHA256, ORDER, REVISION, SHARD, pinned_member


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT.parent / 'qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json'


def audit(data, member, *, aligned):
    assert data['pack_revision'] == REVISION and data['inventory_sha256'] == INVENTORY_SHA256
    assert data['member'] == SHARD and data['read_chunk_bytes'] == CHUNK
    assert data.get('aligned_user_buffer', False) is aligned
    assert data['page_size'] == 16 * 1024
    assert [trial['mode'] for trial in data['trials']] == list(ORDER)
    pages = (member['size'] + data['page_size'] - 1) // data['page_size']
    for i, trial in enumerate(data['trials']):
        assert trial['number'] == i and trial['file_pages'] == pages
        assert trial['sha256'] == member['sha256'] and trial['bytes'] == member['size']
        assert trial['resident_pages_before'] == 0
        assert 0 <= trial['resident_pages_after'] <= pages
        assert isfinite(trial['checksum_seconds']) and trial['checksum_seconds'] > 0
        assert len(trial['sampled_pressure']) == 6 and set(trial['sampled_pressure']) == {1}
    return {mode: {
        'resident_fractions': [t['resident_pages_after'] / pages for t in data['trials'] if t['mode'] == mode],
        'checksum_seconds_median': median(t['checksum_seconds'] for t in data['trials'] if t['mode'] == mode),
    } for mode in ('buffered', 'no-cache')}


def replay(root=ROOT):
    member = pinned_member(INVENTORY)
    grouped = {}
    for name, aligned in (('buffered-reader', False), ('page-aligned', True)):
        raw = json.loads((root / f'{name}.json').read_bytes())
        grouped[name] = audit(raw, member, aligned=aligned)
    return {'member_bytes': member['size'], 'publisher_sha256': member['publisher_sha256'],
            'matched_trials': 8, 'observations': grouped}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
