"""Root terminal recomputation; no model execution and no release promotion."""
import hashlib
import json
import math
from pathlib import Path

COUNTERS = ('ax_engine_jobs_in_flight', 'ax_engine_generation_jobs_pending',
            'ax_engine_generation_commands_queued', 'ax_engine_generation_active_streams',
            'ax_engine_generation_buffered_stream_events')
SOURCE = '5f583018032c406cd3016c6c1c016612dd8b2fbd'
WHEEL = '06c65e8069431258e77490be0eefd9067adcb262cdea156e0c126c0525f0cd14'


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usage(value):
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    require(isinstance(value, dict) and set(keys) <= set(value)
            and set(value) <= set(keys) | {'prompt_tokens_details'}, 'usage schema')
    counts = tuple(value[k] for k in keys)
    require(all(type(n) is int and 0 <= n < 2**32 for n in counts), 'usage integer')
    require(counts[0] == 46 and counts[2] == sum(counts[:2]), 'usage counts')
    if 'prompt_tokens_details' in value:
        detail = value['prompt_tokens_details']
        require(isinstance(detail, dict) and set(detail) == {'cached_tokens'}, 'cache schema')
        n = detail['cached_tokens']
        require(type(n) is int and 0 <= n <= counts[0], 'cache count')
    return counts


def completion(body):
    require(isinstance(body.get('choices'), list) and len(body['choices']) == 1, 'choice count')
    choice = body['choices'][0]
    require(isinstance(choice['text'], str) and choice['finish_reason'] in ('length', 'stop'), 'choice')
    return choice['text'], choice['finish_reason'], usage(body['usage'])


def stream(value, disconnected=False):
    require(isinstance(value['events'], list) and value['events'], 'missing stream events')
    chunks, finishes, usages = [], [], []
    for event in value['events']:
        require('error' not in event, 'stream error')
        choices = event['choices']
        require(isinstance(choices, list) and len(choices) <= 1, 'stream choice count')
        for choice in choices:
            require(type(choice['index']) is int and choice['index'] == 0, 'stream choice index')
            require(isinstance(choice.get('text'), str), 'stream text type')
            chunks.append(choice['text'])
            if choice.get('finish_reason') is not None:
                finishes.append(choice['finish_reason'])
        if event.get('usage') is not None:
            usages.append(event['usage'])
    require(''.join(chunks) == value['text'], 'stream reconstruction')
    if disconnected:
        require(value['closed_after_output'] is True and value['done'] is False
                and value['text'] and not finishes and not usages
                and value['finish_reason'] is None and value['usage'] is None,
                'disconnect did not interrupt unfinished output')
        require(value['producer_active_before_close'] is True, 'producer inactive at disconnect')
        parsed = metrics(value['before_close_metrics'])
        require(parsed['ax_engine_generation_active_streams'] > 0, 'no active producer metric')
        require(value['before_close_quiescence'] == {k: parsed[k] for k in COUNTERS},
                'disconnect metric projection changed')
        return None
    require(value['done'] is True and value['closed_after_output'] is False,
            'stream not completed')
    require(finishes == [value['finish_reason']] and len(usages) == 1
            and usages[0] == value['usage'], 'stream terminal reconstruction')
    return value['text'], value['finish_reason'], usage(value['usage'])


def metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith('#') or '{' in line:
            continue
        parts = line.split()
        if parts[0] not in COUNTERS + ('ax_engine_mlx_mtp_certified_default_on',
                'ax_engine_mlx_flash_next_selected_expert_gathers_total', 'ax_engine_mtp_draft_tokens_total'):
            continue
        require(len(parts) == 2 and parts[0] not in values, 'metric duplicate or shape')
        value = float(parts[1])
        require(math.isfinite(value) and value >= 0, 'metric nonfinite or negative')
        values[parts[0]] = value
    return values


def requests(record):
    require(set(record['modes']) == {'default', 'required'}, 'lifecycle modes')
    identities = {}
    for mode, row in record['modes'].items():
        baseline = completion(row['baseline'])
        require(baseline[2] == (46, 32, 78), 'baseline budget')
        require(stream(row['stream']) == baseline, 'stream generation mismatch')
        for n in (1, 2):
            text, finish, counts = completion(row['budget_' + str(n)])
            require(counts[1] == n and finish == 'length' and baseline[0].startswith(text),
                    'budget mismatch')
        stopped = completion(row['stop'])
        require('\n\n' in baseline[0] and stopped[0] == baseline[0].split('\n\n', 1)[0]
                and stopped[1] == 'stop' and 0 < stopped[2][1] <= 32, 'stop mismatch')
        stream(row['disconnect'], disconnected=True)
        require(completion(row['recovered']) == baseline, 'recovery mismatch')
        for key in ('quiescence_before_disconnect', 'quiescence_after_disconnect', 'quiescence_after_recovery'):
            samples = row[key]
            require(isinstance(samples, list) and samples, 'missing quiescence samples')
            for sample in samples:
                require(set(sample) == set(COUNTERS)
                        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                for v in sample.values()), 'invalid quiescence metric')
            require(all(v == 0 for v in samples[-1].values()), 'request did not drain')
        counters = metrics(row['metrics'])
        require(counters['ax_engine_mlx_mtp_certified_default_on'] == 0
                and counters['ax_engine_mlx_flash_next_selected_expert_gathers_total'] == 0,
                'default promotion or selected route')
        drafted = counters['ax_engine_mtp_draft_tokens_total']
        require(drafted > 0 if mode == 'required' else drafted == 0, 'MTP route mismatch')
        identities[mode] = {key: completion(row[key]) for key in ('baseline','budget_1','budget_2','stop')}
    require(identities['default'] == identities['required'], 'cross-mode generation mismatch')
    return 14


def terminal(record, contract, launch, transfer):
    require(record.get('completed') is True and record.get('passed') is True
            and record.get('phase') == 'complete', 'failed or incomplete lifecycle')
    require(not any(k in record for k in ('active_pid','active_case','error','outstanding_pid','outstanding_group')),
            'outstanding ownership or error')
    require(record['source_commit'] == contract['final_source_commit'] == SOURCE
            and record['wheel_sha256'] == WHEEL
            and record['installation_sha256'] == contract['installation']['sha256']
            and record['native_controls_sha256'] == contract['files']['native-root-review.json']
            and record['execution_contract_sha256'] == launch['contract_sha256'], 'source binding')
    require(record['qualification'] is False and record['release_ready'] is False, 'unjustified promotion')
    require(record['pack_layout_before'] == record['pack_layout_after']
            and len(record['pack_layout_before']) == 48, 'payload pre/post layout')
    require(record['native_vs_installed_library_hashes_equal'] ==
            {name: True for name in ('libmlx','libjaccl')}, 'runtime libraries')
    require(record['pid'] == launch['pid'] and launch['process_group'] == launch['pid'], 'worker ownership')
    require(record['environment_overrides']=={'AX_ENGINE_FLASH_NEXT_EXPERIMENTAL':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}, 'Changed installed-default environment')
    owned = {record['pid']}
    for mode, row in record['modes'].items():
        index = ['default','required'].index(mode)
        expected = ['<private-path:536bbd6fa29de21bd2f6>',
                    'serve','--host','<private-address>','--port',str(31769+index),contract['pack'],'--',
                    '--api-key','','--model-id','flash-next-candidate']
        if mode=='required':expected+=['--mlx-mtp-policy','required']
        require(row['command']==expected,'Changed installed-default command')
        clean = row['cleanup']
        require(type(clean['pid']) is int and clean['pid'] > 1 and clean['pgid'] == clean['pid'], 'server ownership')
        require(all(clean.get(k) is True for k in ('complete','reaped','group_gone'))
                and clean['forced_kill'] is False and clean['errors'] == []
                and type(clean['exit_code']) is int and clean['exit_code'] == 0
                and type(row['exit_code']) is int and row['exit_code'] == 0 and row['forced_kill'] is False, 'unclean server exit')
        owned.add(clean['pid'])
        model = row['health']['runtime']['mlx_model']
        require(model['model_family'] == 'qwen4_exp' and model['runtime_status']['ready'] is True
                and not model['runtime_status'].get('blockers'), 'unready model')
    require(set(transfer['absent_pids']) == owned and set(transfer['absent_groups']) == owned,
            'live ownership not independently checked')
    count = requests(record)
    return dict(schema='ax.flash_next.canonical_lifecycle_root_review.v1',collection_valid=True,
                lifecycle_passed=True,source_commit=SOURCE,installation_sha256=record['installation_sha256'],
                requests_validated=count,payload_prepost_files=47,owned_processes_gone=True,
                qualification=False,release_ready=False)
