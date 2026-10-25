import argparse,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from huggingface_hub import snapshot_download
import ax_engine
from tokenizers import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--pins',default='hub-pins.json');p.add_argument('--only',default='');p.add_argument('--output',required=True);a=p.parse_args()
# BF16 single/batch kernels can differ by an ULP; compare raw direction and scale.
# Fixed-shape repeats and input permutations must still match exactly.
reports=[]
for spec in json.load(open(a.pins)):
 if a.only and a.only not in spec['repo_id']: continue
 path=Path(snapshot_download(spec['repo_id'],revision=spec['revision'],local_files_only=True))
 tok=Tokenizer.from_file(str(path/'tokenizer.json'))
 base=tok.encode('Retrieval systems compare normalized vectors to find relevant documents. Stable local inference enables private search on Apple Silicon.').ids
 lengths=[1,7,16,17,31,32,33,127,128,129,255,256,257,511,512,513,768,1024,2048]
 ids=[(base*((n+len(base)-1)//len(base)))[:n] for n in lengths]
 s=ax_engine.Session(mlx=True,mlx_model_artifacts_dir=str(path))
 singles=np.array([s.embed(row) for row in ids],dtype=np.float32)
 batch=s.embed_batch_array(ids)
 repeat=s.embed_batch_array(ids)
 order=list(range(len(ids)))[::-1]
 reversed_rows=s.embed_batch_array([ids[i] for i in order])[::-1]
 raw_single=np.array(s.embed(ids[3],normalize=False));raw_batch=s.embed_batch_array([ids[3]],normalize=False)[0]
 rejected=[]
 for bad in [[],[json.load(open(path/'config.json')).get('vocab_size',262144)],[2**32-1]]:
  for api in [lambda x:s.embed(x),lambda x:s.embed_batch_array([x])]:
   try: api(bad);rejected.append(False)
   except (ValueError,RuntimeError): rejected.append(True)
 if spec['model_type']=='embedding_gemma2':
  for media in [258880,258881,258884]:
   try:s.embed([2,media,1]);rejected.append(False)
   except (ValueError,RuntimeError):rejected.append(True)
 after=np.array(s.embed(ids[3]));s.close();del s;mx.clear_cache()
 ref=[]
 if spec['model_type']=='qwen3':
  from mlx_lm import load
  model,_=load(str(path))
  for row in ids:
   h=model.model(mx.array([row]))[0,-1].astype(mx.float32); h=h/(mx.linalg.norm(h)+1e-12); ref.append(np.array(h))
 elif spec['model_type']=='embedding_gemma2':
  from mlx_vlm.embedding_loader import load_embedding_model
  model=load_embedding_model(path)
  for row in ids: ref.append(np.array(model(mx.array([row])).text_embeds[0]))
 else:
  import mlx_embeddings
  model,_=mlx_embeddings.load(str(path))
  for row in ids: ref.append(np.array(model(mx.array([row]),attention_mask=mx.ones((1,len(row)),dtype=mx.int32)).text_embeds[0]))
 ref=np.array(ref);del model;mx.clear_cache()
 def cosine(x,y): return np.sum(x*y,axis=-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))
 c=cosine(singles,ref);bc=cosine(singles,batch)
 r={**spec,'lengths':lengths,'shape':list(batch.shape),'reference_cosines':c.tolist(),'batch_cosines':bc.tolist(),'finite':bool(np.isfinite(batch).all() and np.isfinite(singles).all()),'repeat_exact':bool(np.array_equal(batch,repeat)),'permutation_exact':bool(np.array_equal(batch,reversed_rows)),'unnormalized_batch_one_max_abs':float(np.max(np.abs(raw_single-raw_batch))),'invalid_inputs_rejected':rejected,'after_error_cosine':float(cosine(singles[3],after)),'norm_max_error':float(np.max(np.abs(np.linalg.norm(batch,axis=1)-1)))}
 r['batch_reference_cosines']=cosine(batch,ref).tolist()
 r['single_norm_max_error']=float(np.max(np.abs(np.linalg.norm(singles,axis=1)-1)))
 r['unnormalized_batch_one_cosine']=float(cosine(raw_single,raw_batch))
 r['unnormalized_batch_one_norm_ratio']=float(np.linalg.norm(raw_single)/np.linalg.norm(raw_batch))
 r['pass']=bool(r['finite'] and r['repeat_exact'] and r['permutation_exact'] and min(c)>=.999 and min(bc)>=.999 and min(r['batch_reference_cosines'])>=.999 and r['single_norm_max_error']<1e-5 and all(rejected) and r['norm_max_error']<1e-5 and r['after_error_cosine']>.99999 and r['unnormalized_batch_one_cosine']>=.999 and abs(r['unnormalized_batch_one_norm_ratio']-1)<.01)
 reports.append(r); print(json.dumps(r),flush=True);Path(a.output).write_text(json.dumps(reports,indent=2)+'\n')
raise SystemExit(0 if all(r['pass'] for r in reports) else 3)
