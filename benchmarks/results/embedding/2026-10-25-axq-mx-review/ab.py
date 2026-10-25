"""Reproduce same-binary AX embedding dispatch comparisons on cached pinned packs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pins', default='hub-pins.json')
p.add_argument('--output', required=True)
p.add_argument('--worker', type=int)
p.add_argument('--flag')
p.add_argument('--only', default='')
a = p.parse_args()
pins = json.loads(Path(a.pins).read_text())
if a.worker is not None:
    import numpy as np
    import ax_engine
    from huggingface_hub import snapshot_download
    spec = pins[a.worker]
    path = snapshot_download(spec['repo_id'], revision=spec['revision'], local_files_only=True)
    session = ax_engine.Session(mlx=True, mlx_model_artifacts_dir=path)
    cases = [(1,16),(8,16),(1,256),(8,256),(1,1024),(8,128)]
    result = []
    for b, n in cases:
        ids = [([10,20,30,40] * ((n+3)//4))[:n] for _ in range(b)]
        for _ in range(5): session.embed_batch_array(ids)
        times = []
        for _ in range(9):
            start = time.perf_counter()
            out = session.embed_batch_array(ids)
            times.append(1000 * (time.perf_counter()-start))
        assert np.isfinite(out).all()
        result.append({'batch':b, 'tokens':n, 'times_ms':times, 'median_ms':statistics.median(times), 'vector':out[0].tolist()})
    session.close()
    Path(a.output).write_text(json.dumps(result))
    raise SystemExit(0)

import numpy as np
reports = []
for idx, spec in enumerate(pins):
    if a.only and a.only not in spec['repo_id']: continue
    if spec['model_type'] not in ('qwen3', 'embedding_gemma2'): continue
    flag = a.flag or ('AX_MLX_EMBED_NATIVE_SDPA' if spec['model_type']=='qwen3' else 'AX_MLX_EMBED_GEMMA2_PACKED')
    trials = []
    for value in ['0','1','1','0']:
        tmp = Path(a.output + '.worker.json')
        env = dict(os.environ, **{flag:value})
        if flag == 'gemma2_optimized':
            env.pop(flag)
            env['AX_MLX_EMBED_GEMMA2_PACKED'] = value
            if value == '0': env['AX_EMBED_NO_COMPILE'] = '1'
            else: env.pop('AX_EMBED_NO_COMPILE', None)
        subprocess.run([sys.executable,__file__,'--pins',a.pins,'--worker',str(idx),'--output',str(tmp)],env=env,check=True,stdout=subprocess.DEVNULL)
        trials.append({'value':value, 'cases':json.loads(tmp.read_text())})
        tmp.unlink()
    summary = []
    for c in range(len(trials[0]['cases'])):
        off = [t['cases'][c]['median_ms'] for t in trials if t['value']=='0']
        on = [t['cases'][c]['median_ms'] for t in trials if t['value']=='1']
        vectors = np.array([t['cases'][c]['vector'] for t in trials])
        cosines = vectors @ vectors[0] / (np.linalg.norm(vectors,axis=1)*np.linalg.norm(vectors[0]))
        assert min(cosines) >= .999
        summary.append({'batch':trials[0]['cases'][c]['batch'],'tokens':trials[0]['cases'][c]['tokens'],'off_median_ms':statistics.median(off),'on_median_ms':statistics.median(on),'speedup':statistics.median(off)/statistics.median(on),'min_cosine':float(min(cosines))})
    for t in trials:
        for c in t['cases']: del c['vector']
    reports.append({**spec,'flag':flag,'trials':trials,'summary':summary})
    Path(a.output).write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports[-1]['summary']),flush=True)
