"""Adjudicate the fixed-cut state control from the original M5 evidence."""

import gzip
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / 'flash-next-mxfp4-prefix-state-m2-20261121'
INVENTORY = ROOT.parent / 'qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json'
CHECKS = ('state_bits_match_at_cut', 'serialized_restore_exact', 'suffix_state_match',
          'decode_state_match', 'greedy_ids_match')


def digest(raw):
    return sha256(raw).hexdigest()


def unpack(path):
    return gzip.decompress(path.read_bytes())


def serving_cuts(control, prompts, cases):
    cold = control['arms']['cold']['outputs']
    warm = control['arms']['l1']
    assert warm['both_seed_stores_reported'] is True
    cuts = {}
    for index, (name, seed, arm, output) in enumerate((
        ('target', 'seed-a', 'paired', 'output_tokens'),
        ('cancelled', 'seed-b', 'disconnect', 'first_output_tokens'),
    )):
        prompt = control['prompts'][name]['input_tokens']
        assert prompt == prompts[index]
        assert control['prompts'][seed]['input_tokens'] == prompt[:2048]
        assert warm[seed]['output_tokens'][0] == cases[index]['ignored_seed_first_token']
        pair = warm[arm]
        assert pair['checks']['target_tokens_match'] is True
        assert pair['checks']['peer_tokens_match'] is True
        assert pair['checks']['peer_no_restored_tokens'] is True
        assert pair['checks']['l1_restore_2048_proven'] is False
        for sample in (cold[name], pair['target']):
            assert [(row['cursor'], row['advanced_prompt_tokens'], row['delta_token_count'])
                    for row in sample['prefill_progress']] == [(2048, 2048, 0), (2112, 64, 1)]
        steps = pair['target']['prefill_progress']
        assert [row['step_id'] for row in steps] == pair['checks']['shared_prefill_steps_with_ongoing_decode']
        assert all(row['scheduled_requests'] == 2 for row in steps)
        expected = cold[name]['output_tokens']
        assert expected[:32] == cases[index]['cold_tokens']
        observed = pair['target'][output]
        assert observed and observed == expected[:len(observed)]
        cuts[name] = {'logical_widths': [2048, 64], 'observed_output_ids': len(observed),
                      'overlapping_prefill_steps': len(steps)}
    return cuts


def replay(artifact=ROOT):
    raw = (artifact / 'original-state-receipt.json').read_bytes()
    receipt = json.loads(raw)
    source = receipt['source']
    inputs = {
        'source_patch_sha256': unpack(M2 / 'state-probe.patch.gz'),
        'prompts_sha256': (M2 / 'prompts.json').read_bytes(),
        'inventory_sha256': INVENTORY.read_bytes(),
        'control_sha256': unpack(artifact / 'http-control.json.gz'),
    }
    for key, payload in inputs.items():
        assert source[key] == digest(payload), key
    assert source['pack_revision'] == json.loads(inputs['inventory_sha256'])['revision']
    control = json.loads(inputs['control_sha256'])
    assert control['pack_revision'] == source['pack_revision']
    assert control['inventory_sha256'] == source['inventory_sha256']
    assert control['manifest_sha256'] == source['manifest_sha256']
    assert control['exact_prompt_reuses_and_matches'] is True
    assert control['strict_extensions_cold_prefill_and_match'] is True
    assert control['warm_overlap_proven'] is False
    assert receipt['test_exit'] == 0

    samples = [json.loads(line) for line in unpack(artifact / 'probe-samples.jsonl.gz').splitlines()]
    assert len(samples) == receipt['sample_count'] > 0
    assert all(sample['pressure'] == 1 for sample in samples)
    assert max(sample['footprint_bytes'] for sample in samples) == receipt['peak_footprint_bytes']
    cases = receipt['cases']
    assert [case['case'] for case in cases] == [0, 1]
    for case in cases:
        assert (case['prefix_tokens'], case['suffix_tokens'], case['generated_tokens']) == (2048, 64, 32)
        assert all(case[key] is True for key in CHECKS)
        assert case['cold_cut_sha256'] == case['standalone_sha256']
        assert case['alternate_layout_sha256'] != case['standalone_sha256']
        assert len(case['cold_tokens']) == len(case['warm_tokens']) == 32
        assert case['cold_tokens'] == case['warm_tokens']
        assert case['first_differing_output'] is None
    seed_witnesses = [case['case'] for case in cases
                      if case['cold_tokens'][0] != case['ignored_seed_first_token']]
    assert seed_witnesses, 'No case distinguishes the suffix output from the standalone seed output'
    assert receipt['status'] == ('passed' if len(seed_witnesses) == len(cases) else 'failed')
    cuts = serving_cuts(control, json.loads(inputs['prompts_sha256']), cases)
    return {
        'scope': 'Isolated fixed-cut state and token controls',
        'input_receipt_sha256': digest(raw),
        'original_harness_status': receipt['status'],
        'matched_state_and_output_cases': len(cases),
        'matched_generated_ids': sum(len(case['cold_tokens']) for case in cases),
        'cases_distinguishing_seed_and_suffix': seed_witnesses,
        'observed_cold_serving_cuts': cuts,
        'normal_pressure_samples': len(samples),
        'peak_footprint_bytes': receipt['peak_footprint_bytes'],
        'strict_extension_serving_warm_reuse_proven': False,
    }


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True))
