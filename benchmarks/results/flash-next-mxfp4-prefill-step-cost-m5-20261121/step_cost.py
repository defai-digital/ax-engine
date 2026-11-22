"""Summarize measured pure decode turns and shared Flash Next prefill slices."""
import json
from pathlib import Path
from statistics import median
import sys


def summarize(traces, *, require_complete=True):
    by_step = {}
    requests = set()
    for request in traces:
        ident = request['request_id']
        if ident in requests:
            raise ValueError('Duplicate request trace')
        requests.add(ident)
        for row in request['step_costs']:
            step = row['step_id']
            shared = by_step.setdefault(step, {})
            if ident in shared:
                raise ValueError('Duplicate step within request')
            shared[ident] = row
    decode_only = []
    mixed_by_width = {511: [], 512: []}
    for participants in by_step.values():
        records = list(participants.values())
        signatures = {(row['runner_time_us'], row['cpu_time_us'], row['scheduled_requests']) for row in records}
        if len(signatures) != 1:
            raise ValueError('Participants disagree about shared-step timing')
        runner_us, _, scheduled = signatures.pop()
        if len(records) != scheduled:
            continue
        if len(records) == 1 and records[0]['emitted_tokens'] == 1 and records[0]['prefill_tokens'] == 0:
            decode_only.append(runner_us)
        if len(records) == 2:
            prefill = [row['prefill_tokens'] for row in records if row['split_end'] is not None]
            decoding = [row for row in records if row['prefill_tokens'] == 0 and row['emitted_tokens'] == 1]
            if len(prefill) == len(decoding) == 1 and prefill[0] in mixed_by_width:
                mixed_by_width[prefill[0]].append(runner_us)
    if require_complete and (not decode_only or not all(mixed_by_width.values())):
        raise ValueError('The capture lacks a decode-only turn or one of the shared slice widths')
    summaries = {
        'decode_only': {'count': len(decode_only), 'runner_median_us': median(decode_only) if decode_only else None},
        'mixed_prefill_by_width': {
            width: {'count': len(times), 'runner_median_us': median(times) if times else None}
            for width, times in mixed_by_width.items()
        },
    }
    return summaries


if __name__ == '__main__':
    rows = json.loads(Path(sys.argv[1]).read_bytes())
    print(json.dumps(summarize(rows), indent=2, allow_nan=False))
