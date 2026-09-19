"""Installed-wheel lifecycle qualification with actual default/explicit MTP launch."""
import hashlib,json,os,subprocess,sys,threading,time,traceback,urllib.request
from pathlib import Path
root=Path(sys.argv[1]).resolve()
source=root/'product-final-source';os.chdir(source)
sys.path.insert(0,str(source/'scripts'))
import run_native_generation_fault_soak as soak
assert not [k for k in os.environ if k.startswith(('AX_','MLX_','DYLD_','MTL_','METAL_','PYTHONPATH'))]
import ax_engine
package=Path(ax_engine.__file__).parent
manifest=json.loads((root/'build-manifest-product-final-installed.json').read_text())
server=package/'_bin/ax-engine-server'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(server)==manifest['server_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
assert subprocess.check_output(['git','status','--porcelain'])==b''
out=root/'product-final-lifecycle';out.mkdir(exist_ok=False)
cli=root/'product-final-venv/bin/ax-engine'
assert sha(cli)==manifest['cli_sha256']
cache=root/'product-final-cache'
snapshot=cache/'models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP'/'snapshots'/manifest['model_revision']
snapshot.mkdir(parents=True,exist_ok=False)
for name in manifest['model_files']:
 destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
 os.link(root/'model'/name,destination)
doctor=json.loads(subprocess.check_output([str(cli),'doctor','--mlx-model-artifacts-dir',str(root/'model'),'--json'],cwd=out,text=True))
assert doctor['result']=='ready'
(out/'doctor-outside-source.json').write_text(json.dumps(doctor,indent=2)+'\n')
plan=json.loads(subprocess.check_output([str(cli),'serve','qwen3.8-27b:axq','--offline','--hf-cache-root',str(cache),'--dry-run','--json'],cwd=out,text=True))
assert plan['resolved']['revision']==manifest['model_revision'] and plan['resolved']['download']['required'] is False
(out/'cli-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
rows=[]
frozen=json.loads((root/'product-final-lifecycle-prompt.json').read_text())
assert frozen['vocab_size']==soak.bench.model_vocab_size(root/'model')
assert len(frozen['input_tokens'])==128
(out/'soak-prompt.json').write_text(json.dumps(frozen,indent=2)+'\n')
def frozen_prompt(vocab_size,count):
 assert vocab_size==frozen['vocab_size'] and count==len(frozen['input_tokens'])
 return list(frozen['input_tokens'])
soak.bench.mlx_lm_reference_prompt_tokens=frozen_prompt
original_kill=soak.bench.kill_proc
for mode in ['default','mtp']:
 row=dict(mode=mode,source_commit=manifest['source_commit'],wheel_sha256=manifest['wheel_sha256'],server_sha256=sha(server),runtime_overrides={},bundled_runtime_environment={k:v for k,v in os.environ.items() if k=='AX_ENGINE_METAL_BUILD_DIR'},passed=False)
 log=(out/(mode+'-server.log')).open('w');processes=[];timer=None;port=31503
 payload=dict(model='qwen3.8-27b',input_tokens=list(range(1,129)),max_output_tokens=64,sampling=dict(temperature=0,top_k=0,top_p=1,repetition_penalty=1,seed=0,ignore_eos=True))
 (out/'fixed-request.json').write_text(json.dumps(payload,indent=2)+'\n')
 def probe(label):
  row['phase']=label
  request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=180) as response: result=json.load(response)
  (out/(mode+'-'+label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  assert result['status']=='finished' and len(result['output_tokens'])==64
  mtp=result['performance']['mtp'];expected=mode=='mtp'
  assert mtp['requested'] is expected and mtp['active'] is expected and mtp['direct_fallback_steps']==0,mtp
  quiescent,metrics=soak.wait_for_quiescence(f'http://127.0.0.1:{port}',timeout=30)
  assert quiescent,metrics
  row[label]=dict(output_tokens=result['output_tokens'],mtp=mtp,metrics=metrics)
  return result['output_tokens']
 def launch(binary,model_dir,port_number,**kwargs):
  global timer
  row['phase']='launch'
  soak.bench.ensure_port_available(port_number)
  command=[str(cli),'serve','qwen3.8-27b:axq','--offline','--hf-cache-root',str(cache),'--port',str(port_number)]
  if mode=='mtp':command+=['--','--mlx-mtp-policy','required','--mlx-mtp-disable-ngram-stacking']
  row['command']=command;row['launch_cwd']=str(out)
  process=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT);processes.append(process)
  timer=threading.Timer(1800,lambda:original_kill(process));timer.daemon=True;timer.start()
  assert soak.bench.wait_for_server(f'http://127.0.0.1:{port_number}/health',proc=process)
  cold=probe('cold');warm=probe('warm');assert cold==warm,'cold/warm fixed-prompt mismatch'
  row['phase']='full_qa'
  qa_command=[sys.executable,'-u',str(source/'qa/run_qa.py'),'--base-url',f'http://127.0.0.1:{port_number}','--model','qwen3.8-27b','--mode','ngram' if mode=='default' else 'mtp','--streams','both','--max-tokens','1024','--temperature','0','--timeout','180','--all','--seed','20260917','--output',str(out/(mode+'-full-qa.html')),'--json-output',str(out/(mode+'-full-qa.json'))]
  row['full_qa_command']=qa_command
  with (out/(mode+'-full-qa.log')).open('w') as qa_log:
   qa_result=subprocess.run(qa_command,cwd=source,stdout=qa_log,stderr=subprocess.STDOUT,timeout=1500)
  row['full_qa_exit_code']=qa_result.returncode
  print(mode,'full QA exit',qa_result.returncode,flush=True)
  return process
 def finish(process):
  try:
   if process.poll() is None:
    assert probe('recovery')==row['cold']['output_tokens'],'post-cancellation recovery output changed'
    row['recovery_passed']=True
  finally:
   original_kill(process);row['server_exit_code']=process.returncode
 soak.bench.AX_ENGINE_SERVER=server;soak.bench.start_axengine=launch;soak.bench.kill_proc=finish
 try:
  code=soak.main_with_args(['--model-dir',str(root/'model'),'--model-id','qwen3.8-27b','--output',str(out/(mode+'-soak.json')),'--port',str(port),'--rounds','3','--concurrency','4','--slow-delay-ms','250','--stalled-hold-ms','180000','--stalled-output-tokens','8192'])
  row['soak_exit_code']=code
  row['passed']=code==0 and row.get('recovery_passed') is True and row.get('full_qa_exit_code')==0
 except Exception as error:row.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  if timer:timer.cancel()
  for process in processes:original_kill(process)
  log.close()
  row['owned_servers_reaped']=all(p.poll() is not None for p in processes)
  row['passed']=row['passed'] and row['owned_servers_reaped']
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(mode,row['passed'],row.get('error'),flush=True)
print('lifecycle complete',all(x['passed'] for x in rows),flush=True)
raise SystemExit(0 if all(x['passed'] for x in rows) else 1)
