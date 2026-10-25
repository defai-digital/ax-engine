import json
import numpy as np
import mlx.core as mx
import ax_engine
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from mlx_lm import load
spec=json.load(open('hub-pins.json'))[4];path=snapshot_download(spec['repo_id'],revision=spec['revision'],local_files_only=True)
tok=Tokenizer.from_file(path+'/tokenizer.json');base=tok.encode('Retrieval systems compare normalized vectors to find relevant documents. Stable local inference enables private search on Apple Silicon.').ids
ids=[(base*10)[:n] for n in [127,128,129]]
s=ax_engine.Session(mlx=True,mlx_model_artifacts_dir=path);singles=np.array([s.embed(row) for row in ids]);batch=s.embed_batch_array(ids);s.close();del s;mx.clear_cache()
model,_=load(path);ref=[]
for row in ids:
 h=model.model(mx.array([row]))[0,-1].astype(mx.float32);h=h/mx.linalg.norm(h);ref.append(np.array(h))
h=model.model(mx.array([row+[0]*(129-len(row)) for row in ids]));h=mx.stack([h[i,len(row)-1] for i,row in enumerate(ids)]).astype(mx.float32);h=h/mx.linalg.norm(h,axis=1,keepdims=True)
rbat=np.array(h);ref=np.array(ref)
print({'ax_single_ref':np.sum(ref*singles,axis=1).tolist(),'ax_batch_ref':np.sum(ref*batch,axis=1).tolist(),'oracle_batch_vs_single':np.sum(ref*rbat,axis=1).tolist(),'ax_batch_vs_oracle_batch':np.sum(rbat*batch,axis=1).tolist()})
