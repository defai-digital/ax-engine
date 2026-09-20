"""Replay all original QA checks; report quality separately from MTP certification."""
import gzip
import hashlib
import json
from pathlib import Path

import qa_checks

ROOT = Path(__file__).resolve().parent
GATES = {gate: 'not_assessed' for gate in ('MTP-S', 'MTP-P', 'MTP-D')}
require = qa_checks.require


def load():
    summary = json.loads((ROOT / 'summary.json').read_text())
    expected = {'qa.json.gz', 'qa_checks.py', 'closed_checks.py', 'checkers.py',
                'prompt_def.py', 'full-qa-items.json', 'full-qa-inputs.json', 'reader-inputs.json'}
    require(set(summary['artifacts']) == expected, 'Missing or unexpected artifact')
    for name, row in summary['artifacts'].items():
        raw = (ROOT / name).read_bytes()
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                'Changed artifact: ' + name)
    raw = (ROOT / 'qa.json.gz').read_bytes()
    require(raw[4:8] == bytes(4), 'Nondeterministic gzip timestamp')
    return summary, json.loads(gzip.decompress(raw))


def verify(summary, evidence):
    require(summary['source_commit'] == evidence['source_commit'] == qa_checks.SOURCE,
            'Wrong source identity')
    require(summary['wheel_sha256'] == evidence['wheel_sha256'] == qa_checks.WHEEL,
            'Wrong wheel identity')
    for item in (summary, evidence, evidence['root_review']):
        require(item['qualification'] is False and item['release_ready'] is False,
                'QA evidence cannot promote a release')
    require(summary['mtp_certification'] == GATES, 'Unjustified MTP certification')
    rows = evidence['records']
    computed = qa_checks.terminal(rows['output/result.json'], rows['contract.json'],
                                 rows['launch.json'], rows['transfer-receipt.json'])
    require(computed == evidence['root_review'], 'Published verdict differs from raw reconstruction')
    require(summary['original_qa_passed'] is computed['passed']
            and summary['quality_failures'] == computed['quality_failures']
            and summary['matching_text_pairs'] == computed['matching_text_pairs']
            and summary['normal_stops'] is computed['normal_stops']
            and summary['requests'] == computed['requests'] == 210,
            'Changed QA result or scope')
    return dict(collection_valid=True, original_qa_passed=computed['passed'],
                quality_failures=computed['quality_failures'],
                matching_text_pairs=computed['matching_text_pairs'],
                normal_stops=computed['normal_stops'], requests=210,
                mtp_certification=GATES, qualification=False, release_ready=False)


if __name__ == '__main__':
    print(json.dumps(verify(*load()), indent=2))
