import json
import argparse
p=argparse.ArgumentParser();p.add_argument("--model-index",type=int,default=4);p.add_argument("--lengths",default="127,128,129");args=p.parse_args()
lengths=[int(n) for n in args.lengths.split(",")]
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from mlx_lm import load
spec=json.load(open('hub-pins.json'))[args.model_index];path=snapshot_download(spec['repo_id'],revision=spec['revision'],local_files_only=True)
tok=Tokenizer.from_file(path+'/tokenizer.json');base=tok.encode('Retrieval systems compare normalized vectors to find relevant documents. Stable local inference enables private search on Apple Silicon.').ids
ids=[(base*10)[:n] for n in lengths]
model,_=load(path);ref=[]
for row in ids:
 h=model.model(mx.array([row]))[0,-1].astype(mx.float32);h=h/mx.linalg.norm(h);ref.append(np.array(h))
ref=np.array(ref)
class RowLinear(nn.Module):
 def __init__(self,linear): super().__init__();self.linear=linear
 def __call__(self,x): return mx.concatenate([self.linear(x[i:i+1]) for i in range(x.shape[0])],axis=0) if x.ndim==3 and x.shape[0]>1 else self.linear(x)
for stage in ['original','qkv','all_linear']:
 if stage=='qkv':
  for layer in model.model.layers:
   for key in ['q_proj','k_proj','v_proj']:setattr(layer.self_attn,key,RowLinear(getattr(layer.self_attn,key)))
 if stage=='all_linear':
  for layer in model.model.layers:
   layer.self_attn.o_proj=RowLinear(layer.self_attn.o_proj)
   for key in ['gate_proj','up_proj','down_proj']:setattr(layer.mlp,key,RowLinear(getattr(layer.mlp,key)))
 h=model.model(mx.array([row+[0]*(max(lengths)-len(row)) for row in ids]));h=mx.stack([h[i,len(row)-1] for i,row in enumerate(ids)]).astype(mx.float32);h=h/mx.linalg.norm(h,axis=1,keepdims=True)
 print(stage,np.sum(ref*np.array(h),axis=1).tolist(),flush=True)
