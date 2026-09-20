"""M4 admission probe, excluded from all performance measurements."""
import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
model = Path(args.model)
import mlx.core as mx
from _ax_engine import Session

rows = []
os.environ['AX_STREAM_EXPERTS'] = 'off'
for explicit in (None, 'auto', 'off'):
    mx.clear_cache()
    start = time.perf_counter()
    session = Session(mlx=True, mlx_model_artifacts_dir=str(model),
                      mlx_stream_experts=explicit)
    mx.synchronize()
    rows.append(dict(explicit=explicit, environment='off',
                     active_bytes=mx.get_active_memory(), load_s=time.perf_counter() - start))
    session.close()
    del session
    gc.collect()
    mx.synchronize()
assert rows[0]['active_bytes'] > 2 * rows[1]['active_bytes'], rows
assert abs(rows[0]['active_bytes'] - rows[2]['active_bytes']) < 128 * 2**20, rows

# Link immutable payloads into a disposable test fixture; mutate only the copied
# expert-admission metadata. The original model directory is never modified.
with tempfile.TemporaryDirectory(prefix='tiel-required-') as temp:
    fixture = Path(temp)
    for source in model.iterdir():
        if source.is_file() and source.name != 'ax_expert_stream.json':
            os.link(source.resolve(), fixture / source.name)
    metadata = json.loads((model / 'ax_expert_stream.json').read_text())
    metadata['required'] = True
    (fixture / 'ax_expert_stream.json').write_text(json.dumps(metadata))
    try:
        session = Session(mlx=True, mlx_model_artifacts_dir=str(fixture),
                          mlx_stream_experts='off')
    except (RuntimeError, ValueError) as error:
        message = str(error)
        assert 'REQUIRED' in message or 'required=true' in message, message
    else:
        session.close()
        raise AssertionError('required expert streaming was bypassed')
Path(args.output).write_text(json.dumps(dict(rows=rows, required_off_rejected=True), indent=2) + '\n')
print('PASS: environment Off loads resident; explicit Auto overrides env; explicit Off loads resident; required Off fails closed.')
