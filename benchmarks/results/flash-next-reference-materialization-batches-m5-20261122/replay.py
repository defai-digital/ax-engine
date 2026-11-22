"""Replay the M5 bounded and full reference materialization diagnostics."""
from hashlib import sha256
import gzip
import json
from math import isfinite
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent
GRAPH = ROOT.parent / 'flash-next-reference-parameter-graph-m2-20261121' / 'graph.json.gz'
GRAPH_SHA = 'bbc459b8456395d817dd0f4a3a9f1b8adaa2763b10897e7f261f6483623c5d33'
SOURCE_SHA = 'f92c9e04a94b0a0a681a7820b780847cba433bc5bbcadf483a516cbecc3a3221'
INITIAL = ('lazy_load_start', 'lazy_load_end', 'materialize_start')


def contents(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f'Missing archived file: {name}')
    return stream.read()


def audit_case(archive, case, groups):
    def raw(name):
        return contents(archive, f'{case}/{name}')

    receipt = json.loads(raw('status.json'))
    prelaunch = json.loads(raw('prelaunch.json'))
    partial = json.loads(raw('partial-result.json'))
    events = [json.loads(line) for line in raw('phase-events.jsonl').splitlines()]
    samples = [json.loads(line) for line in raw('samples.jsonl').splitlines()]
    expected_count = 15 if case == 'ple' else 73
    terminal_event = 'ple_boundary' if case == 'ple' else 'materialization_boundary'
    assert receipt['graph_sha256'] == GRAPH_SHA and receipt['frozen_sha256'] == SOURCE_SHA
    assert receipt['wired_override'] is False and receipt['ordinary_qualification'] is False
    assert prelaunch['sample']['pressure'] == 1 and prelaunch['quiet']['cpu_idle_percent'] >= 90
    assert partial['completed'] is False and partial['phase'] == 'loading' and 'qa' not in partial
    assert tuple(e['phase'] for e in events[:3]) == INITIAL
    intervals = [e['monotonic_seconds'] for e in events]
    assert all(isfinite(x) for x in intervals) and all(a < b for a, b in zip(intervals, intervals[1:]))
    assert all(0 <= e['mlx_active_bytes'] <= e['mlx_peak_bytes'] and
               0 < e['physical_footprint_bytes'] <= e['kernel_lifetime_peak_footprint_bytes'] for e in events)
    sequence = [(part, index) for index in range(expected_count) for part in ('batch_start', 'batch_end')]
    sequence.append((terminal_event, expected_count - 1))
    assert [(e['phase'], e['batch']) for e in events[3:]] == sequence[:len(events) - 3]
    completed = []
    for event in events[3:]:
        if event['phase'] == terminal_event:
            continue
        group = groups[event['batch']]
        assert event['logical_bytes'] == group['logical_bytes']
        assert event['ple_bytes'] == group['ple_ngram_logical_bytes']
        if event['phase'] == 'batch_end':
            completed.append(group['index'])
    pressure = [s['pressure'] for s in samples]
    assert pressure and pressure[0] == 1
    stopped = receipt['stop_reason'] is not None
    exit_prefix = 'ple-batches' if case == 'ple' else 'full-batches'
    assert raw('phase-status.txt') == f'{exit_prefix} {1 if stopped else 0}\n'.encode()
    if stopped:
        assert receipt['stop_reason'] == 'memory pressure above normal' and receipt['exit_code'] == -15
        assert pressure[-1] != 1 and all(value == 1 for value in pressure[:-1])
        assert events[-1]['phase'] != terminal_event
    else:
        assert receipt['exit_code'] == 0 and all(value == 1 for value in pressure)
        assert events[-1]['phase'] == terminal_event and len(completed) == expected_count
    pending = events[-1]['batch'] if events[-1]['phase'] == 'batch_start' else None
    warning = None
    if stopped:
        sample = samples[-1]
        lag = sample['monotonic_seconds'] - intervals[-1]
        assert isfinite(lag) and lag >= 0
        warning = {'seconds_after_last_phase_event': round(lag, 6),
                   'process_physical_footprint_bytes': sample['physical_footprint_bytes'],
                   'system_compressor_bytes': sample['compressor_bytes']}
    return {
        'completed_batches': len(completed), 'completed_logical_bytes': sum(groups[b]['logical_bytes'] for b in completed),
        'last_started_unfinished_batch': pending, 'terminal_pressure': pressure[-1],
        'pressure_samples': len(samples), 'stopped_for_pressure': stopped,
        'elapsed_observed_materialization_seconds': round(intervals[-1] - intervals[2], 6),
        'phase_peak_mlx_bytes': max(e['mlx_peak_bytes'] for e in events),
        'phase_peak_physical_footprint_bytes': max(e['physical_footprint_bytes'] for e in events),
        'host_peak_compressor_bytes': max(s['compressor_bytes'] for s in samples),
        'pressure_warning_sample': warning,
    }


def replay(root=ROOT):
    data = gzip.decompress(GRAPH.read_bytes())
    assert sha256(data).hexdigest() == GRAPH_SHA
    groups = json.loads(data)['summary']['batches']
    with tarfile.open(root / 'evidence.tar.xz') as archive:
        return {'scope': 'Reference materialization only; no prefill or QA',
                'graph_sha256': GRAPH_SHA,
                'ranges': {name: audit_case(archive, name, groups) for name in ('ple', 'full')}}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
