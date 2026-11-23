"""Extract charged shared-cache values from request-owned SSE store reports."""

from argparse import ArgumentParser
import json
from pathlib import Path
import tarfile

ENTRY = 'ax_mlx_prefix_cache_entries'
CHARGED_KIB = 'ax_mlx_prefix_cache_bytes_kib'
STORES = 'ax_mlx_prefix_cache_stores'
EVICTIONS = 'ax_mlx_prefix_cache_evictions'
REUSED = 'ax_mlx_prefix_cache_reused_tokens'
ORDER = ('off-1', 'on-1', 'on-2', 'off-2')
CASE_LABELS = ('first', 'unseeded', 'replacement', 'aligned', 'boundary', 'outside')


def store_reports(reply):
    reports = []
    seen_steps = set()
    for event in reply['cache_events']:
        decisions = event['decisions']
        if (event['event'] != 'step' or event['request_id'] != reply['request_id']
                or not decisions.get(STORES, 0)):
            continue
        step_id = event['step_id']
        if type(step_id) is not int or step_id in seen_steps:
            raise ValueError('Duplicate or invalid step for a cache store report')
        seen_steps.add(step_id)
        if not all(key in decisions for key in (ENTRY, CHARGED_KIB)):
            continue
        entry_count, charged_kib = decisions[ENTRY], decisions[CHARGED_KIB]
        if type(entry_count) is not int or not 0 < entry_count <= 8:
            raise ValueError('Unexpected L1 entry count for the bound cache settings')
        if type(charged_kib) is not int or not 0 < charged_kib <= 2 * 1024**2:
            raise ValueError('Unexpected L1 byte charge for the bound cache settings')
        reports.append({
            'step_id': step_id,
            'entries': entry_count,
            'charged_kib': charged_kib,
            'stores': decisions[STORES],
            'evictions': decisions.get(EVICTIONS, 0),
            'reused_tokens': decisions.get(REUSED, 0),
        })
    return reports


def summarize_arm(arm):
    result = {}
    for name in CASE_LABELS:
        row = arm['cases'][name]
        result[name] = {
            'peer_store_reports': store_reports(row['peer']),
            'target_store_reports': store_reports(row['target']),
        }
    return result


def replay(path):
    with tarfile.open(path) as archive:
        member = archive.getmember('observed/receipt.json')
        if not member.isfile():
            raise ValueError('The signed receipt must be a regular file')
        with archive.extractfile(member) as source:
            receipt = json.load(source)
    arms = {name: summarize_arm(receipt['arms'][name]) for name in ORDER}
    return {
        'scope': 'Reported shared L1 values in request-owned SSE store reports',
        'build': receipt['build']['server_sha256'],
        'arms': arms,
        'second_question_target_values': {
            name: arms[name]['replacement']['target_store_reports'] for name in ORDER
        },
    }


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    print(json.dumps(replay(cli.parse_args().archive), indent=2, sort_keys=True, allow_nan=False))
