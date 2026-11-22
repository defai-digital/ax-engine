"""Check current M5 bounded-reference parity and report scoped timing observations."""

from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[2] / 'scripts'
SIGNED = ROOT.parent / 'qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json'
FROZEN_QA = 'benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa'
INVENTORY = 'benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json'
QA_FIELDS = ('id', 'prompt_tokens', 'generated_ids', 'text', 'normal_stop',
             'terminal_eos_id', 'generated_tokens')
MATRIX_FIELDS = ('length', 'run', 'warmup', 'generated_ids', 'normal_stop',
                 'terminal_eos_id', 'generated_tokens')


def digest(blob):
    return sha256(blob).hexdigest()


def unpack(path):
    return gzip.decompress(path.read_bytes())


def identical_rows(rows, expected, fields, size):
    assert len(rows) == len(expected) == size
    for index, (row, prior) in enumerate(zip(rows, expected, strict=True)):
        for key in fields:
            assert row[key] == prior[key], (index, key)


def replay(artifact=ROOT):
    sys.path.insert(0, str(SCRIPTS))
    import check_flash_next_release_gates as evaluator

    sources = json.loads((artifact / 'source.json').read_bytes())
    raw_native = unpack(artifact / 'native-matrix.json.gz')
    raw_bounded = unpack(artifact / 'bounded-result.json.gz')
    raw_samples = unpack(artifact / 'memory-samples.jsonl.gz')
    raw_signed = SIGNED.read_bytes()
    for raw, path in (
        (raw_native, 'ax-matrix.json'),
        (raw_signed, 'signed-m5-reference.json'),
        (unpack(artifact / 'run_flash_next_affine8_reference.py.gz'),
         'scripts/run_flash_next_affine8_reference.py'),
    ):
        assert digest(raw) == sources[path], path
    native, bounded, signed = map(json.loads, (raw_native, raw_bounded, raw_signed))
    assert bounded['schema'] == 'ax-engine.flash-next.affine8-ple-diagnostic.v1'
    assert bounded['mode'] == 'bounded' and bounded['backend'] == 'metal'
    assert bounded['completed'] is True and bounded['phase'] == 'complete'
    assert bounded['qualification'] is False and bounded['release_ready'] is False
    assert bounded['reference']['runtime'] == 'mlx-vlm-affine8-ple-diagnostic'
    assert bounded['reference']['commit'] == signed['reference']['commit']
    for key in ('versions', 'language_sha256', 'runtime_files_bound', 'prefill_chunk'):
        assert bounded[key] == signed[key], key
    assert bounded['pack_revision'] == signed['revision'] == native['revision']
    assert bounded['inventory_sha256'] == signed['inventory_sha256'] == native['inventory_sha256'] == sources[INVENTORY]
    assert bounded['hardware']['os'] == native['hardware']['os']
    for key in ('soc', 'memory_bytes'):
        assert bounded['hardware'][key] == native['hardware'][key] == signed['hardware'][key], key
    assert bounded['workload_hashes']['ax_matrix_sha256'] == sources['ax-matrix.json']
    for path, key in (
        ('full-qa-items.json', 'items_sha256'), ('full-qa-inputs.json', 'inputs_sha256'),
        ('closed_checks.py', 'closed_checks_sha256'), ('qa_checks.py', 'qa_checks_sha256'),
        ('checkers.py', 'checkers_sha256'), ('prompt_def.py', 'prompt_def_sha256'),
    ):
        assert bounded['workload_hashes'][key] == sources[f'{FROZEN_QA}/{path}'], key
    assert native['repeats_per_arm'] == 2
    qa, signed_qa = bounded['qa'], signed['qa']
    identical_rows(qa['rows'], signed_qa['rows'], QA_FIELDS, 105)
    assert qa['grades'] == signed_qa['grades'] and qa['failures'] == signed_qa['failures']
    matrix, signed_matrix = bounded['matrix'], signed['matrix']
    assert matrix['prompt_id_hashes'] == signed_matrix['prompt_id_hashes']
    identical_rows(matrix['cells'], signed_matrix['cells'], MATRIX_FIELDS, 15)
    assert all(len(cell['generated_ids']) == 128 for cell in matrix['cells'])

    thresholds = json.loads(evaluator.THRESHOLDS.read_bytes())
    floor = thresholds['timing_host']['min_cpu_idle_percent']
    idle = (bounded['host_quiet']['cpu_idle_percent'], bounded['host_idle_after'])
    pressure = Counter(json.loads(line)['pressure'] for line in raw_samples.splitlines())
    assert pressure == {1: 25}
    numeric = evaluator.gate_reference_relative(native, bounded, thresholds)[0]
    assert len(numeric['cells']) == 6
    return {
        'scope': 'Independent bounded PLE diagnostic',
        'qualification': False,
        'release_ready': False,
        'native_server_sha256': native['server_sha256'],
        'native_matrix_sha256': digest(raw_native),
        'bounded_result_sha256': digest(raw_bounded),
        'resource_samples_sha256': digest(raw_samples),
        'signed_reference_sha256': digest(raw_signed),
        'matched_qa_rows_and_grades': len(qa['rows']),
        'matched_fixed_matrix_cells': len(matrix['cells']),
        'peak_mlx_memory_bytes': bounded['peak_mlx_memory_bytes'],
        'peak_process_rss_bytes': bounded['peak_process_rss_bytes'],
        'normal_pressure_samples': pressure[1],
        'reference_cpu_idle_before_percent': idle[0],
        'reference_cpu_idle_after_percent': idle[1],
        'timing_host_floor_met': all(floor <= sample <= 100 for sample in idle),
        'numeric_bounds_met': numeric['passed'],
        'cells': numeric['cells'],
    }


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
