import os,json,subprocess,sys
from pathlib import Path
if len(sys.argv)==1:
 for env in [{},{'AX_EMBED_NO_COMPILE':'1'},{'AX_MLX_EMBED_NATIVE_SDPA':'0'},{'AX_EMBED_NO_COMPILE':'1','AX_MLX_EMBED_NATIVE_SDPA':'0'}]:
  print(env,flush=True)
  subprocess.run([sys.executable,__file__,'worker'],env=dict(os.environ,**env),check=True)
else:
 import numpy as np, mlx.core as mx, ax_engine
 from huggingface_hub import snapshot_download
 from tokenizers import Tokenizer
 from mlx_lm import load
 spec=json.load(open('hub-pins.json'))[4];path=snapshot_download(spec['repo_id'],revision=spec['revision'],local_files_only=True)
 tok=Tokenizer.from_file(path+'/tokenizer.json');base=tok.encode('Retrieval systems compare normalized vectors to find relevant documents. Stable local inference enables private search on Apple Silicon.').ids
 rows=[(base*10)[:n] for n in [31,129]]
 s=ax_engine.Session(mlx=True,mlx_model_artifacts_dir=path);singles=np.array([s.embed(row) for row in rows]);batch=s.embed_batch_array(rows);s.close();del s;mx.clear_cache()
 model,_=load(path);ref=[]
 for row in rows:
  h=model.model(mx.array([row]))[0,-1].astype(mx.float32);h=h/mx.linalg.norm(h);ref.append(np.array(h))
 ref=np.array(ref);print({'single_ref':np.sum(ref*singles,axis=1).tolist(),'batch_ref':np.sum(ref*batch,axis=1).tolist()},flush=True)
