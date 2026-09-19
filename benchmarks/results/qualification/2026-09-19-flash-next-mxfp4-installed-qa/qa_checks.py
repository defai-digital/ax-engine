"""Independently recompute full-cohort QA from raw responses with the frozen checkers."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys
ROOT=Path(__file__).resolve().parent
PREP=ROOT.parent/'qa-preparation'
SOURCE='5f583018032c406cd3016c6c1c016612dd8b2fbd'
WHEEL='06c65e8069431258e77490be0eefd9067adcb262cdea156e0c126c0525f0cd14'

def require(value,message):
 if not value:raise ValueError(message)

def load_inputs():
 pins=json.loads((ROOT/'reader-inputs.json').read_text())['files']
 expected={'closed_checks.py','checkers.py','prompt_def.py','full-qa-items.json','full-qa-inputs.json'}
 require(set(pins)==expected,'Wrong offline checker dependencies')
 for name,value in pins.items():
  require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,'Changed frozen checker/input '+name)
 import closed_checks as qa
 return qa,json.loads((ROOT/'full-qa-items.json').read_text()),json.loads((ROOT/'full-qa-inputs.json').read_text())['prepared']

def cases(record):
 qa,items,prompts=load_inputs();expected=[i['id'] for i in items]
 require(len(expected)==len(set(expected))==105 and [p['id'] for p in prompts]==expected,'Wrong frozen cohort')
 require(set(record['modes'])=={'disabled','required'},'Incomplete modes')
 failures={};stops=True;comparisons={};long_context={}
 for mode,row in record['modes'].items():
  require([c['id'] for c in row['cases']]==expected,'Missing or reordered cases')
  failures[mode]=[];comparisons[mode]=[]
  for case,item,prompt in zip(row['cases'],items,prompts,strict=True):
   response=case['response'];raw=response['response']
   require(isinstance(raw['choices'],list) and len(raw['choices'])==1,'Wrong raw choice count')
   choice=raw['choices'][0];usage=raw['usage'];cap=64 if item['id']=='long_context_record_mass' else 256
   require(type(choice['index']) is int and choice['index']==0 and isinstance(choice['text'],str)
           and response['text']==choice['text'] and response['finish_reason']==choice['finish_reason'],'Raw response alias changed')
   require(all(type(usage[k]) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
           and usage['prompt_tokens']==response['prompt_tokens']==prompt['token_count']
           and usage['completion_tokens']<=cap
           and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],'Usage or original budget changed')
   require(type(response['seconds']) in (int,float) and math.isfinite(response['seconds']) and response['seconds']>=0,'Invalid elapsed time')
   closed=qa.evaluate_checker(choice['text'],item).as_dict()
   report=qa.run_all_checks(choice['text'],qa.item_to_prompt(item)).as_dict()
   require(closed==case['closed_checker'] and report==case['report'],'Recomputed checker differs')
   if not (closed['passed'] and report['hard_pass']):failures[mode].append(item['id'])
   stops &= choice['finish_reason']=='stop'
   comparisons[mode].append((choice['text'],closed))
  long_context[mode]=row['cases'][-1]['response']['text']
  require(row['cases'][-1]['response']['prompt_tokens']==29774,'Long-context token count changed')
 pairs=list(zip(comparisons['disabled'],comparisons['required'],strict=True))
 text_identity=all(a[0]==b[0] for a,b in pairs);checker_identity=all(a[1]==b[1] for a,b in pairs)
 require(record['quality_failures']==failures and record['text_identity'] is text_identity
         and record['checker_identity'] is checker_identity and record['normal_stops'] is stops,'Aggregate summary differs from raw cases')
 passed=text_identity and checker_identity and stops and not any(failures.values())
 require(record['passed'] is passed,'Acceptance claim differs')
 return dict(requests=210,per_mode=105,quality_failures=failures,matching_text_pairs=sum(a[0]==b[0] for a,b in pairs),
             text_identity=text_identity,checker_identity=checker_identity,normal_stops=stops,passed=passed,long_context=long_context)

def terminal(record,contract,launch,transfer):
 require(record['completed'] is True and record['collection_complete'] is True
         and record['phase'] in ('complete','acceptance_failed'),'Incomplete QA collection')
 require(record['source_commit']==contract['final_source_commit']==SOURCE and record['wheel_sha256']==contract['wheel_sha256']==WHEEL
         and record['execution_contract_sha256']==launch['contract_sha256']
         and record['installation_sha256']==contract['installation']['sha256']
         and record['lifecycle_terminal_sha256']==contract['completed_lifecycle']['terminal']['sha256'],'Wrong QA source/prerequisites')
 require(record['qualification'] is False and record['release_ready'] is False
         and not any(k in record for k in ('active_pid','active_case','error','outstanding_pid','outstanding_group')),'Invalid terminal ownership')
 require(record['pack_layout_before']==record['pack_layout_after'] and len(record['pack_layout_before'])==48,'Incomplete pre/post layout')
 owned={record['pid']};require(record['pid']==launch['pid']==launch['process_group'],'Worker ownership changed')
 require(record['environment_overrides']=={'AX_ENGINE_FLASH_NEXT_EXPERIMENTAL':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},'Changed QA environment')
 for mode,row in record['modes'].items():
  expected=['<private-path:536bbd6fa29de21bd2f6>',
            'serve','--host','<private-address>','--port',str(31767+['disabled','required'].index(mode)),
            contract['pack'],'--','--api-key','','--model-id','flash-next-candidate',
            '--total-blocks','4096','--mlx-mtp-policy',mode]
  require(row['command']==expected,'Changed QA execution command')
  c=row['cleanup'];owned.add(c['pid'])
  require(c['pid']==c['pgid'] and all(c[k] is True for k in ('complete','reaped','group_gone'))
          and c['forced_kill'] is False and c['errors']==[] and type(c['exit_code']) is int and c['exit_code']==0
          and type(row['exit_code']) is int and row['exit_code']==0 and row['forced_kill'] is False,'Unclean QA server exit')
  model=row['health']['runtime']['mlx_model']
  require(model['model_family']=='qwen4_exp' and model['runtime_status']['ready'] is True and not model['runtime_status'].get('blockers'),'Wrong or blocked model')
  drafted=row['final_metrics']['ax_engine_mtp_draft_tokens_total']
  require(type(drafted) in (int,float) and math.isfinite(drafted) and (drafted>0 if mode=='required' else drafted==0),'Wrong MTP execution mode')
 require(set(transfer['absent_pids'])==owned and set(transfer['absent_groups'])==owned,'Owned processes not independently absent')
 report=cases(record)
 return dict(schema='ax.flash_next.current_source_qa_root_review.v1',collection_valid=True,source_commit=SOURCE,
             **report,qualification=False,release_ready=False)
