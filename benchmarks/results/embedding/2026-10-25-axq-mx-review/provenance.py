import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

root=Path.cwd()
files=[p for p in (root/'python'/'ax_engine').glob('_ax_engine*.so')]
r={'hardware':'MacBook Pro M5 Max 128 GB','os':platform.mac_ver()[0], 'architecture':platform.machine(),'python':platform.python_version(), 'base_commit':'fcf6e0fc8', 'source_state':'modified; see source-files.json', 'build_profile':'release-pyext', 'model_storage':'external volume; warm cache', 'performance_scope':'AX versus AX; CPU-materialized float32 matrices; no qualification or competitive claim'}
r['packages']={p:importlib.metadata.version(p) for p in ['mlx','mlx-metal','mlx-lm','mlx-vlm','mlx-embeddings','transformers','huggingface-hub','numpy','maturin']}
r['rustc']=subprocess.check_output(['/opt/homebrew/bin/rustc','--version'],text=True).strip()
r['cargo']=subprocess.check_output(['/opt/homebrew/bin/cargo','--version'],text=True).strip()
r['toolchain_source']='Homebrew; rustup unavailable'
r['binary_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
for p in [root/'target/release/ax-engine-server',root/'target/release/generate-manifest']:
 if p.exists():r['binary_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
r['model_manifests']={}
from huggingface_hub import snapshot_download
for spec in json.loads(Path('hub-pins.json').read_text()):
 model_dir=Path(snapshot_download(spec['repo_id'],revision=spec['revision'],local_files_only=True))
 r['model_manifests'][spec['repo_id']]={'revision':spec['revision'],'sha256':hashlib.sha256((model_dir/'model-manifest.json').read_bytes()).hexdigest()}
Path('evidence/environment-final.json').write_text(json.dumps(r,indent=2)+'\n')
paths=Path('source-files.txt').read_text().splitlines()
Path('evidence/source-files.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
