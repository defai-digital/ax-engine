"""Verify the pinned Flash Next explicit MTP load bypass controls."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'scripts'))
import check_flash_next_native_support as native  # noqa: E402


ATTACH_FAILURES = 'ax_engine_flash_next_mtp_attach_failed_total'
USAGE_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def require(assertion, what):
    if not assertion:
        raise ValueError(what)


def signature(row):
    response = row['response']
    return (row['name'], row['request'], response['choices'],
            tuple(response['usage'][field] for field in USAGE_FIELDS), row['activation_delta'])


def replay(receipt):
    require(receipt['completed'] is True and receipt['qualification'] is False
            and receipt['revision'] == native.PACK_REVISION, 'Unexpected functional scope')
    require(receipt['server_sha256'] == '36676d61d3d4407e8f8955e7353927bf92e1e8e507837a143963975238bbee28'
            and receipt['inventory_sha256'] == '97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6'
            and receipt['model_manifest_sha256'] == '62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef',
            'Measured executable or publisher input changed')
    require(receipt['original_comparison_exit_code'] != 0, 'Missing original timing comparison outcome')
    arms = receipt['arms']
    require(list(arms) == ['ordinary', 'skip', 'required'], 'Incomplete sequence of actual controls')
    counts = {}
    for arm, policy in (('ordinary', 'default'), ('skip', 'default'), ('required', 'required')):
        item = arms[arm]
        require(item['policy'] == policy, 'Wrong MTP policy')
        counts[arm] = 0
        for row, (name, prompt, max_tokens) in zip(item['cases'], native.CASES, strict=True):
            request = row['request']
            require(row['name'] == name and request['messages'] == [{'role': 'user', 'content': prompt}]
                    and request['max_tokens'] == max_tokens and request['temperature'] == 0,
                    'Functional prompt changed')
            native.check_answer(name, row['response'])
            activation = native.check_activation(row['metrics_before'], row['metrics_after'],
                                                 policy == 'required', row['response']['usage']['completion_tokens'],
                                                 max_tokens)
            require(activation == row['activation_delta'] and row['metrics_after'][ATTACH_FAILURES] == 0,
                    'Unexpected activation or attachment-error telemetry')
            counts[arm] += activation[native.VERIFIED]
    ordinary, bypass = (arms[name]['cases'] for name in ('ordinary', 'skip'))
    require([signature(row) for row in ordinary] == [signature(row) for row in bypass],
            'Stable direct output or usage changed')
    require(counts == {'ordinary': 0, 'skip': 0, 'required': 107}
            and receipt['required_verified_steps'] == counts['required'], 'Wrong verifier counts')
    negative = receipt['required_skip']
    require(negative['exit_code'] != 0 and negative['activation_error_observed'] is True,
            'Required MTP accepted a skipped drafter')
    require(receipt['matched_base_output_controls'] == len(native.CASES), 'Incomplete direct comparison')
    phase = {label: arm['sidecar_us'] for label, arm in arms.items()}
    require(0 <= phase['skip'] < phase['ordinary'] and 0 <= negative['sidecar_us'] < phase['required'],
            'The bypass did not skip the sidecar phase')
    cached = {arm: {row['name']:
                    row['response']['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
                    for row in arms[arm]['cases']} for arm in ('ordinary', 'skip')}
    return {'matched_direct_response_choices_and_token_totals': len(native.CASES),
            'verifier_steps': counts, 'required_skip_exit_code': negative['exit_code'],
            'sidecar_us': phase | {'required_skip': negative['sidecar_us']},
            'request_cache_counters': cached}


if __name__ == '__main__':
    receipt = gzip.decompress((ROOT / 'receipt.json.gz').read_bytes())
    expected = (ROOT / 'raw.sha256').read_text().split()[0]
    require(sha256(receipt).hexdigest() == expected, 'Source receipt checksum mismatch')
    print(json.dumps(replay(json.loads(receipt)), indent=2, sort_keys=True, allow_nan=False))
