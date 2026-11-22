"""Verify the M5 two-user scheduler timing receipts against the prior solo run."""

from hashlib import sha256
import json
from pathlib import Path
import tarfile

from step_cost import summarize


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'flash-next-mxfp4-selective-prefill-m5-20261121' / 'evidence.tar.xz'
PRIOR_SOLO_SHA256 = '612675166c33c197b0389795fa9db1109998454a877543114bda06663cfd8f9b'


def payload(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f'Missing archived file {name}')
    return stream.read()


def replay(root=ROOT, prior=PRIOR):
    with tarfile.open(prior) as reference, tarfile.open(root / 'evidence.tar.xz') as bundle:
        solo = payload(reference, 'solo-reference.json')
        assert sha256(solo).hexdigest() == PRIOR_SOLO_SHA256
        baseline = {row['prompt_id']: row['output_token_ids'] for row in json.loads(solo)['observations']
                    if row['phase'] == 'measured'}
        build = json.loads(payload(reference, 'build.json'))
        report = json.loads(payload(bundle, 'result.json'))
        assert report['server_sha256'] == build['server_sha256']
        assert report['complete_step_costs'] is True

        inputs = json.loads(payload(bundle, 'burst-inputs.json'))
        sources = json.loads(payload(bundle, 'client-sources.json'))
        assert inputs.keys() == sources.keys() == report['results'].keys() == {'early-decoder', 'ending-decoder'}
        combined = []
        for name, expected in report['results'].items():
            corpus = payload(bundle, f'{name}.jsonl')
            assert sha256(corpus).hexdigest() == inputs[name]['corpus_sha256'] == expected['corpus_sha256']
            cases = {row['id']: row for row in map(json.loads, corpus.splitlines())}
            client = payload(bundle, f'{name}.json')
            assert sha256(client).hexdigest() == sources[name]['portable_sha256']
            assert sources[name]['original_sha256'] == expected['client_sha256']
            data = json.loads(client)
            seen = set()
            assert len(data['observations']) == len(cases) == 2
            for row in data['observations']:
                ident = row['prompt_id']
                assert row['ok'] is True and ident not in seen
                seen.add(ident)
                assert row['prompt_token_ids'] == cases[ident]['input_tokens']
                assert row['output_token_ids'] == baseline[ident]
                assert len(row['output_token_ids']) == 96
            assert sorted(row['input_tokens'] for row in data['observations']) == [512, 8192]
            assert sorted(row['scheduled_at_s'] for row in data['observations']) == [
                inputs[name]['short_arrival_seconds'], inputs[name]['long_arrival_seconds']]
            assert expected['matched_output_ids'] == 192
            assert len(expected['drained']) == 5 and all(value == 0 for value in expected['drained'].values())
            step_trace = payload(bundle, f'{name}-steps.json')
            assert sha256(step_trace).hexdigest() == expected['trace_sha256']
            steps = json.loads(step_trace)
            assert {row['request_id'] for row in data['observations']} == {row['request_id'] for row in steps}
            assert json.loads(json.dumps(summarize(steps, require_complete=False))) == expected['step_costs']
            combined.extend(steps)
        samples = payload(bundle, 'burst-samples.jsonl').splitlines()
        assert len(samples) == report['sample_count'] and samples
        assert all(json.loads(line)['pressure'] == 1 for line in samples)
        assert json.loads(json.dumps(summarize(combined))) == report['combined_step_costs']
        return report


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
