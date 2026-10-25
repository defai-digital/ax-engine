import argparse, json, subprocess, time, statistics
from pathlib import Path
import numpy as np
import mlx.core as mx
from huggingface_hub import snapshot_download
import ax_engine
p=argparse.ArgumentParser();p.add_argument('--pins',default='hub-pins.json');p.add_argument('--output',required=True);p.add_argument('--only',default='');a=p.parse_args()
failed=False
rows=json.load(open(a.pins))
for r in rows:
 if a.only and a.only not in r['repo_id']: continue
 name=r['repo_id'].split('/')[-1]; out=Path(a.output)/name;out.mkdir(parents=True,exist_ok=True)
 path=Path(snapshot_download(r['repo_id'],revision=r['revision'],local_files_only=True))
 started=time.perf_counter()
 try:
  result=subprocess.run(['target/release/generate-manifest',str(path)],capture_output=True,text=True)
  (out/'manifest.log').write_text(result.stdout+result.stderr)
  result.check_returncode()
  session=ax_engine.Session(mlx=True,mlx_model_artifacts_dir=str(path))
  loaded=time.perf_counter()-started
  from tokenizers import Tokenizer
  tok=Tokenizer.from_file(str(path/'tokenizer.json'))
  texts=['The capital of France is Paris.','A short query','AX Engine provides local text embeddings on Apple Silicon.','The cat sits on the mat.','Vector databases store semantic representations.']
  ids=[tok.encode(t).ids for t in texts]
  if r['model_type']=='qwen3': ids=[v if v[-1]==151643 else v+[151643] for v in ids]
  singles=np.array([session.embed(v,normalize=True) for v in ids]); batch=session.embed_batch_array(ids,normalize=True)
  np.savez(out/'vectors.npz',single=singles,batch=batch)
  bench=[]
  for b,n in [(1,16),(8,16),(1,256),(8,256),(1,1024),(8,128)]:
   toks=[([10,20,30,40]*(n//4+1))[:n] for _ in range(b)]
   for _ in range(2): session.embed_batch_array(toks,normalize=True)
   times=[]
   for _ in range(5):
    t=time.perf_counter();v=session.embed_batch_array(toks,normalize=True);times.append((time.perf_counter()-t)*1000)
   bench.append({'batch':b,'tokens':n,'times_ms':times,'median_ms':statistics.median(times),'finite':bool(np.isfinite(v).all())})
  session.close();del session
  mx.clear_cache()
  if r['model_type']=='qwen3':
   from mlx_lm import load
   model,_=load(str(path));ref=[]
   for v in ids:
    h=model.model(mx.array([v]))[0,-1].astype(mx.float32); h=h/(mx.linalg.norm(h)+1e-12);mx.eval(h);ref.append(np.array(h))
  elif r['model_type']=='embedding_gemma2':
   from mlx_vlm.embedding_loader import load_embedding_model
   model=load_embedding_model(path);ref=[]
   for v in ids: ref.append(np.array(model(mx.array([v])).text_embeds[0]))
  else:
   import mlx_embeddings
   model,_=mlx_embeddings.load(str(path));ref=[]
   for v in ids: ref.append(np.array(model(mx.array([v]),attention_mask=mx.ones((1,len(v)),dtype=mx.int32)).text_embeds[0]))
  ref=np.array(ref);np.save(out/'reference.npy',ref)
  def cosine(x,y): return np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1))
  report={**r,'load_seconds':loaded,'shape':list(batch.shape),'reference_cosines':cosine(singles,ref).tolist(),'batch_cosines':cosine(singles,batch).tolist(),'bench':bench}
  del model;mx.clear_cache()
 except Exception as e:
  failed=True
  report={**r,'error':str(e)}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

raise SystemExit(2 if failed else 0)
