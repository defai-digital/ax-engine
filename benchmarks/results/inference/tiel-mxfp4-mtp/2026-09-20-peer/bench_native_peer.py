"""Matched native-library completion-boundary benchmark, without HTTP timing."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def phases(start, events, budget):
    assert events and sum(n for _, n in events) == budget
    first, first_n = events[0]
    last = events[-1][0]
    assert start < first < last and first_n < budget
    return dict(ttft_s=first-start, completion_s=last-start,
                completion_tok_s=budget/(last-start), first_batch_tokens=first_n,
                decode_tokens=budget-first_n, decode_s=last-first,
                decode_tok_s=(budget-first_n)/(last-first))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',choices=['ax','mtplx'],required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--cases',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--profile',default='sustained')
    p.add_argument('--conservative',type=int,default=0)
    p.add_argument('--stream-experts',choices=['auto','on','off'],default=None)
    p.add_argument('--reps',type=int,default=5)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--cooldown',type=float,default=3)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    cases=json.loads(Path(a.cases).read_text())
    if a.smoke: cases=[dict(cases[1],generation_tokens=32)]
    doc=dict(engine=a.engine,profile=a.profile if a.engine=='mtplx' else 'throughput',
             conservative=a.conservative,cases=[],repetitions=a.reps,warmups=a.warmups,
             timing='perf_counter before native API call to final committed token callback; decode excludes first emitted batch',
             prefix_cache=False,greedy=True,depth_cap=3,model=a.model,cooldown_s=a.cooldown,stream_experts=a.stream_experts)
    import mlx.core as memory_mx
    doc['python_version']=sys.version
    doc['mlx_version']=__import__('importlib.metadata',fromlist=['version']).version('mlx')
    start=time.perf_counter()
    if a.engine=='ax':
        env={'AX_MLX_NATIVE_CONFIRM':'1','AX_MLX_MTP_FORCE_REQUESTED':'1',
             'AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP':'1',
             'AX_MLX_MTP_CONSERVATIVE_DEPTH':str(a.conservative),
             'AX_ENGINE_PREFIX_REUSE_DISABLED':'1','AX_MLX_PREFIX_CACHE_MAX_BYTES':'0',
             'AX_MLX_PREFIX_CACHE_MAX_ENTRIES':'0','AX_MLX_PREFIX_CACHE_DISK_DISABLED':'1'}
        os.environ.update(env)
        import _ax_engine
        rt=_ax_engine.Session(model_id='tiel-peer',mlx=True,mlx_model_artifacts_dir=a.model,
                              mlx_stream_experts=a.stream_experts)
        doc['runtime']=rt.runtime()
        doc['extension_sha256']=hashlib.sha256(Path(_ax_engine.__file__).read_bytes()).hexdigest()
        doc['environment']={k:v for k,v in os.environ.items() if k.startswith('AX_')}
    else:
        import mlx.core as mx
        from mtplx.profiles import apply_profile_env,get_profile
        from mtplx.runtime import load
        from mtplx.generation import generate_mtpk
        from mtplx.sampling import SamplerConfig
        from mtplx.draft_lm_head import _install_draft_lm_head
        apply_profile_env(a.profile)
        os.environ['MTPLX_CONTEXT_COPY']='0'
        profile=get_profile(a.profile)
        rt=load(a.model,mtp=True)
        spec=profile.draft_lm_head
        doc['draft_head']=_install_draft_lm_head(rt,bits=spec.bits,group_size=spec.group_size,mode=spec.mode)
        doc['environment']={k:v for k,v in os.environ.items() if k.startswith('MTPLX_')}
        sampler=SamplerConfig(temperature=0,top_p=1,top_k=0)
        mx.synchronize()
    doc['load_s']=time.perf_counter()-start
    out=Path(a.output)
    out.write_text(json.dumps(doc,indent=2,default=str)+'\n')
    for case in cases:
        tokens=case['token_ids']; budget=case['generation_tokens']
        assert token_hash(tokens)==case['sha256']
        result=dict(case=case['case'],prompt_tokens=len(tokens),generation_tokens=budget,
                    prompt_sha256=token_hash(tokens),trials=[],warmup_trials=[])
        doc['cases'].append(result)
        for rep in range(a.warmups+a.reps):
            events=[]; emitted=[]
            def record(batch):
                if batch:
                    now=time.perf_counter()
                    events.append((now,len(batch)))
                    emitted.extend(batch)
            memory_mx.synchronize()
            time.sleep(a.cooldown)
            memory_mx.reset_peak_memory()
            memory_before = dict(active=memory_mx.get_active_memory(), cache=memory_mx.get_cache_memory())
            start=time.perf_counter()
            if a.engine=='ax':
                terminal=None
                for event in rt.stream_generate(input_tokens=tokens,max_output_tokens=budget,
                       temperature=0,top_p=1,top_k=0,seed=0,ignore_eos=True):
                    if event['event']=='step':record(event.get('delta_tokens',[]))
                    if event['event']=='response':terminal=event['response']
                assert terminal is not None and terminal['output_tokens']==emitted
                native=terminal
            else:
                response=generate_mtpk(rt,tokens,max_tokens=budget,sampler=sampler,
                    draft_sampler=sampler,speculative_depth=3,min_speculative_depth=1,
                    seed=0,stop_token_ids=set(),mtp_hidden_variant='post_norm',
                    mtp_cache_policy='persistent',mtp_history_policy='committed',
                    verify_strategy='capture_commit',verify_core='linear-gdn-from-conv-tape',
                    token_callback=record,session_bank=None,repetition_stop=False,loop_guard=False)
                assert response.tokens==emitted
                native=dataclasses.asdict(response.stats)
            returned=time.perf_counter()
            row=dict(rep=rep,measured=rep>=a.warmups,**phases(start,events,budget),
                     api_return_s=returned-start,output_tokens=emitted,
                     memory_before=memory_before,
                     memory_after=dict(active=memory_mx.get_active_memory(),cache=memory_mx.get_cache_memory(),peak=memory_mx.get_peak_memory()),
                     emissions=[(t-start,n) for t,n in events],native=native)
            target=result['trials'] if row['measured'] else result['warmup_trials']
            target.append(row)
            out.write_text(json.dumps(doc,indent=2,default=str)+'\n')
            print(a.engine,a.profile,case['case'],rep,'completion',round(row['completion_tok_s'],2),
                  'decode',round(row['decode_tok_s'],2),'first_batch',row['first_batch_tokens'],flush=True)
    if a.engine=='ax':rt.close()
    doc['complete']=True
    out.write_text(json.dumps(doc,indent=2,default=str)+'\n')


if __name__=='__main__':main()
