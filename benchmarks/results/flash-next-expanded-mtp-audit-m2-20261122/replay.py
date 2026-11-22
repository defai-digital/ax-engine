"""Verify the verifier's captured decisions and immutable prompt sources."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def replay(record, inputs, provenance):
    require(record['schema'] == 'ax-engine.flash-next.runner-mtp-greedy-long-audit.v1'
            and record['qualification'] is False and record['prefill_quantum'] == 2048
            and record['ignore_eos'] is True, 'Wrong exercised runner scope')
    require([case['name'] for case in inputs] == [case['name'] for case in provenance['cases']],
            'The frozen workload inventory changed')
    result = []
    vocab = provenance['matrix_vocab_size']
    for case, target, plan in zip(record['cases'], inputs, provenance['cases'], strict=True):
        n = len(target['prompt'])
        budget = target['budget']
        require(case['name'] == target['name'] == plan['name']
                and case['input_tokens'] == plan['input_tokens'] == n
                and case['budget'] == plan['budget'] == budget and n + budget <= 32768,
                'Audit row does not match its bound workload')
        direct, mtp = (case[key] for key in ('direct_tokens', 'mtp_tokens'))
        require(len(direct) == len(mtp) == budget
                and all(type(token) is int and 0 <= token < vocab for token in direct + mtp),
                'Missing output IDs or invalid vocabulary tokens')
        changed = [i for i, (x, y) in enumerate(zip(direct, mtp, strict=True)) if x != y]
        require(changed == case['changed_from_direct'], 'The direct comparison report changed')
        steps, accepted, rejected, positions, end = 0, 0, 0, 0, 1
        for observation in case['decisions']:
            pos = observation['position'] - n + 1
            require(end <= pos < budget and mtp[pos - 1] == observation['primary']
                    and observation['remaining'] == budget - pos, 'Decision position disagrees with the output')
            accept = observation['remaining'] > 1 and observation['draft'] == observation['target']
            require(observation['accepted'] == int(accept), 'Accepted a token outside the verifier decision')
            if accept:
                expected = [observation['target'], observation['next']]
            else:
                require(observation['next'] == observation['target'], 'Rejected draft changed the correction')
                expected = [observation['target']]
            end = pos + len(expected)
            require(end <= budget and expected == observation['emitted'] == mtp[pos:end],
                    'Verified outputs were missing or changed')
            steps += 1
            accepted += int(accept)
            rejected += int(observation['draft'] is not None and not accept)
            positions += len(expected)
        require(steps == case['verified_steps'] > 0 and positions == case['output_positions_audited']
                and accepted == case['accepted'] and rejected == case['rejected'],
                'The raw audit coverage and counters disagree')
        result.append({'name': case['name'], 'input_tokens': n, 'budget': budget, 'verified_steps': steps,
                       'accepted_drafts': accepted, 'rejected_drafts': rejected,
                       'verified_output_positions': positions, 'changed_from_direct': changed})
    return result


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
QA_DATA = (REPO / 'benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa'
           / 'full-qa-inputs.json')
INVENTORY = (REPO / 'benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support'
             / 'pack-inventory.json')


def bound_prompts(provenance):
    sys.path.insert(0, str(REPO / 'scripts'))
    from run_flash_next_generate_matrix import workload_ids

    source = QA_DATA.read_bytes()
    require(sha256(source).hexdigest() == provenance['qa_source_sha256'], 'Original QA source changed')
    frozen = {case['id']: case['input_ids'] for case in json.loads(source)['prepared']}
    vocab = provenance['matrix_vocab_size']
    require(vocab == 248320, 'Wrong model vocabulary')
    cases = [{'name': 'math_apples_store', 'prompt': frozen['math_apples_store'], 'budget': 64},
             {'name': 'matrix-8192-1', 'prompt': workload_ids(vocab, 8192, 1), 'budget': 64},
             {'name': 'long_context_record_mass', 'prompt': frozen['long_context_record_mass'], 'budget': 8}]
    return (json.dumps(cases, separators=(',', ':')) + '\n').encode()


if __name__ == '__main__':
    raw = gzip.decompress((ROOT / 'report.json.gz').read_bytes())
    run = json.loads((ROOT / 'receipt.json').read_bytes())
    config = json.loads((ROOT / 'cases-provenance.json').read_bytes())
    prompts = bound_prompts(config)
    require(run['status'] == 'complete' and run['test_exit'] == 0 and run['qualification'] is False
            and (ROOT / 'status.txt').read_text() == '0\n'
            and run['mtp_cost_policy_disabled'] is True, 'The owned real-pack process did not pass')
    require(run['build']['commit'] == '9df17f26a5d135357b8556a6881e2f1306a5bd29'
            and run['test_sha256'] == run['build']['test_sha256']
            == 'ba5845f8bfe9e4ef38254e3f8ca913a04c310e963ffabf5154b1224d4b2768ea'
            and run['hardware']['soc'] == 'Apple M2 Ultra' and run['hardware']['memory_bytes'] == 192 << 30
            and run['build']['mlx'] == run['build']['mlx_metal'] == '0.32.4', 'Platform or binary changed')
    require(run['pack_revision'] == 'ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3'
            and run['inventory_sha256'] == sha256(INVENTORY.read_bytes()).hexdigest()
            and run['native_manifest'] == {
                'model-manifest.json': '62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef'},
            'The verified publisher pack changed')
    require(sha256(raw).hexdigest() == run['report_sha256'] and sha256(prompts).hexdigest()
            == run['cases_sha256'] == config['cases_sha256'], 'Workload or Rust result bytes changed')
    verdict = replay(json.loads(raw), json.loads(prompts), config)
    require(verdict == json.loads((ROOT / 'comparison.json').read_bytes()), 'Replay and preserved verdict differ')
    print(json.dumps(verdict, indent=2, allow_nan=False))
