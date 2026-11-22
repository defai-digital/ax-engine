"""Validate the paired M5 publisher checksum and materialization receipts."""
from hashlib import sha256
import gzip
import json
from math import isfinite
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT.parent / 'qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json'
GRAPH = ROOT.parent / 'flash-next-reference-parameter-graph-m2-20261121' / 'graph.json.gz'
INV = '97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6'
GRAPH_SHA = 'bbc459b8456395d817dd0f4a3a9f1b8adaa2763b10897e7f261f6483623c5d33'
FROZEN_SHA = 'f92c9e04a94b0a0a681a7820b780847cba433bc5bbcadf483a516cbecc3a3221'
NATIVE_MANIFEST_SHA = '06cc592252dfe740b078519091995e706f2a3ae4b2e8b863720bea179d7d0e76'
INPUT_MANIFEST_SHA = 'b50899fa9b4ac95a73846c13374d30479d9173d3f3cd6e95553cbf0300716253'
REFERENCE_COMMIT = '6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb'
REVISION = 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
DIAGNOSTIC_INPUTS = ('checksum_ledger.py', 'clean_state.py', 'frozen_progress.py',
                     'hash_cache_probe.py', 'measure_checksum.py', 'ple_batches.py',
                     'process_footprint.py', 'run_pair.py')
ORDER = (('no-cache-first', 'no-cache'), ('buffered-first', 'buffered'),
         ('buffered-second', 'buffered'), ('no-cache-second', 'no-cache'))
VM = ('Pages free', 'Pages inactive', 'File-backed pages', 'Pages occupied by compressor')
PAGE_SIZE = 16 * 1024
PREFIX = ('lazy_load_start', 'lazy_load_end', 'materialize_start')


def body(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f'No raw evidence for {name}')
    return stream.read()


def check_vm(gauges):
    assert set(gauges) == set(VM)
    assert all(type(value) is int and value >= 0 and value % PAGE_SIZE == 0 for value in gauges.values())


def audit(archive, label, mode, packaged, publisher, batches):
    def file(name):
        return body(archive, f'{label}/{name}')

    done = json.loads(file('status.json'))
    prelaunch = json.loads(file('prelaunch.json'))
    partial = json.loads(file('partial-result.json'))
    hashes = [json.loads(line) for line in file('checksum-events.jsonl').splitlines()]
    events = [json.loads(line) for line in file('phase-events.jsonl').splitlines()]
    samples = [json.loads(line) for line in file('samples.jsonl').splitlines()]
    assert done == packaged['receipt'] and done['checksum_mode'] == packaged['mode'] == mode
    assert done['graph_sha256'] == GRAPH_SHA and done['frozen_sha256'] == FROZEN_SHA
    assert done['ordinary_qualification'] is False and done['wired_override'] is False
    assert prelaunch['sample']['pressure'] == 1 and prelaunch['quiet']['cpu_idle_percent'] >= 90
    assert partial['revision'] == REVISION and partial['inventory_sha256'] == INV
    assert partial['reference']['commit'] == REFERENCE_COMMIT
    assert partial['hardware']['soc'] == 'Apple M5 Max' and partial['hardware']['memory_bytes'] == 128 * 2**30
    assert partial['hardware']['storage']['internal'] and partial['hardware']['storage']['solid_state']
    assert partial['qualification'] is False and partial['completed'] is False and partial['phase'] == 'loading'
    assert len(hashes) == packaged['completed_checksums'] == len(publisher)
    assert sum(row['file_bytes'] for row in hashes) == packaged['checksummed_bytes']
    assert [row['member'] for row in hashes] == list(publisher)[:len(hashes)]
    for row in hashes:
        expected = publisher[row['member']]
        assert row['mode'] == mode and expected['sha256'] == row['sha256']
        assert expected['size'] == row['file_bytes']
        assert row['file_pages'] == (row['file_bytes'] + PAGE_SIZE - 1) // PAGE_SIZE
        assert row['cached_before'] == 0
        assert row['cached_after'] == (0 if mode == 'no-cache' else row['file_pages'])
        assert row['checksum_seconds'] > 0 and isfinite(row['checksum_seconds'])
        check_vm(row['vm_before'])
        check_vm(row['vm_after'])
    timing = [row['monotonic_seconds'] for row in (*hashes, *events)]
    assert all(isfinite(x) for x in timing) and all(p < q for p, q in zip(timing, timing[1:]))
    assert tuple(row['phase'] for row in events[:3]) == PREFIX[:len(events[:3])]
    assert len(events) >= len(PREFIX)
    if len(events) > 3:
        sequence = [(part, b['index']) for b in batches for part in ('batch_start', 'batch_end')]
        sequence.append(('materialization_boundary', len(batches) - 1))
        checks = [(row['phase'], row['batch']) for row in events[3:]]
        assert checks == sequence[:len(checks)]
        for row in events[3:]:
            if row['phase'] in ('batch_start', 'batch_end'):
                planned = batches[row['batch']]
                assert (row['logical_bytes'], row['ple_bytes']) == (planned['logical_bytes'], planned['ple_ngram_logical_bytes'])
    pressure = [row['pressure'] for row in samples]
    assert len(pressure) >= 2 and pressure[-1] == 2 and all(value == 1 for value in pressure[:-1])
    sample_times = [row['monotonic_seconds'] for row in samples[1:]]
    assert all(isfinite(x) for x in sample_times) and all(p < q for p, q in zip(sample_times, sample_times[1:]))
    assert events[-1]['monotonic_seconds'] <= sample_times[-1]
    for row in samples[1:]:
        check_vm(row['vm_memory_bytes'])
    completed = sum(row['phase'] == 'batch_end' for row in events)
    assert done['stop_reason'] == 'memory pressure above normal' and done['exit_code'] == -15
    assert events[-1]['phase'] == 'batch_start' and 0 <= completed < len(batches)
    footprint = samples[-1]['physical_footprint_bytes']
    assert type(footprint) is int and footprint > 0
    return {'completed_payload_checksums': len(hashes), 'verified_payload_bytes': packaged['checksummed_bytes'],
            'per_file_initial_resident_pages_sum': sum(row['cached_before'] for row in hashes),
            'per_file_post_checksum_resident_pages_sum': sum(row['cached_after'] for row in hashes),
            'system_file_backed_bytes_after_last_checksum': hashes[-1]['vm_after']['File-backed pages'] if hashes else None,
            'completed_materialization_batches': completed,
            'last_materialization_event': ({k: events[-1][k] for k in ('phase', 'batch') if k in events[-1]}
                                           if events else None),
            'terminal_pressure': pressure[-1], 'pressure_samples': len(pressure),
            'terminal_vm_bytes': samples[-1].get('vm_memory_bytes'),
            'terminal_process_footprint_bytes': footprint}


def replay(root=ROOT):
    raw = INVENTORY.read_bytes()
    assert sha256(raw).hexdigest() == INV
    publisher = {row['name']: row for row in json.loads(raw)['members'] if row['name'].endswith('.safetensors')}
    assert len(publisher) == 26
    raw = gzip.decompress(GRAPH.read_bytes())
    assert sha256(raw).hexdigest() == GRAPH_SHA
    batches = json.loads(raw)['summary']['batches']
    with tarfile.open(root / 'evidence.tar.xz') as archive:
        inputs = body(archive, 'inputs.sha256')
        assert sha256(inputs).hexdigest() == INPUT_MANIFEST_SHA
        entries = (line.split() for line in inputs.decode().splitlines())
        manifest = {name: digest for digest, name in entries}
        for source in DIAGNOSTIC_INPUTS:
            assert sha256(body(archive, f'inputs/{source}')).hexdigest() == manifest[source]
        raw = body(archive, 'result.json')
        report = json.loads(raw)
        assert report['native_manifest_sha256'] == NATIVE_MANIFEST_SHA
        assert report['order'] == [list(pair) for pair in ORDER]
        assert body(archive, 'phase-status.txt') == b'checksum-cache-pair 0\n'
        trials = {label: audit(archive, label, mode, report['arms'][label], publisher, batches)
                  for label, mode in ORDER}
    return {'scope': report['scope'], 'result_sha256': sha256(raw).hexdigest(), 'trials': trials}


if __name__ == '__main__':
    result = json.dumps(replay(), indent=2, sort_keys=True) + '\n'
    if result != (ROOT / 'comparison.json').read_text():
        raise ValueError('Replayed result differs from the published comparison')
    print(result, end='')
