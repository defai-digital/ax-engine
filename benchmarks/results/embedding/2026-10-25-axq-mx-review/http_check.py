"""Check the real native embedding HTTP route against the direct API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

import ax_engine
from huggingface_hub import snapshot_download
import numpy as np
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pins', default='hub-pins.json')
p.add_argument('--output', required=True)
a = p.parse_args()
reports = []
for spec in json.loads(Path(a.pins).read_text()):
    path = Path(snapshot_download(spec['repo_id'], revision=spec['revision'], local_files_only=True))
    tokenizer = Tokenizer.from_file(str(path/'tokenizer.json'))
    texts = ['The capital of France is Paris.', 'A short query', 'Embedding models enable semantic retrieval of relevant documents.']
    ids = [tokenizer.encode(t).ids for t in texts]
    if spec['model_type'] == 'qwen3':
        ids = [row if row[-1] == 151643 else row+[151643] for row in ids]
    session = ax_engine.Session(mlx=True, mlx_model_artifacts_dir=str(path))
    expected = session.embed_batch_array(ids)
    session.close()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    model_id = spec['repo_id'].split('/')[-1]
    logfile = Path(a.output).with_suffix('.server.log')
    with logfile.open('w') as log:
        proc = subprocess.Popen(['target/release/ax-engine-server','--mlx','--model-id',model_id,'--mlx-model-artifacts-dir',str(path),'--host','127.0.0.1','--port',str(port)], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(240):
                if proc.poll() is not None: raise RuntimeError(f'server exited: {proc.returncode}; inspect local server log')
                try:
                    with socket.create_connection(('127.0.0.1',port), timeout=.2): break
                except OSError: time.sleep(.25)
            else: raise RuntimeError('server start timeout')
            def post(body, endpoint='/v1/embeddings'):
                req = urllib.request.Request(f'http://127.0.0.1:{port}{endpoint}', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(req,timeout=60) as response:
                        return response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code, json.loads(error.read())
            status, body = post({'model':model_id,'input':ids})
            assert status == 200, (status,body)
            matrix = np.array([r['embedding'] for r in body['data']])
            assert matrix.shape == expected.shape
            assert [r['index'] for r in body['data']] == list(range(len(ids)))
            cos = np.sum(matrix*expected,axis=1)/(np.linalg.norm(matrix,axis=1)*np.linalg.norm(expected,axis=1))
            assert min(cos) >= .999
            status, single = post({'input':ids[0]})
            assert status == 200
            assert np.dot(single['data'][0]['embedding'],expected[0]) >= .999
            status, text_body = post({'input':texts})
            assert status == 200, (status,text_body)
            text_matrix = np.array([r['embedding'] for r in text_body['data']])
            assert text_matrix.shape == expected.shape and np.isfinite(text_matrix).all()
            text_cos = np.sum(text_matrix*expected,axis=1)/(np.linalg.norm(text_matrix,axis=1)*np.linalg.norm(expected,axis=1))
            assert min(text_cos) >= .999, text_cos
            status, records = post({'records':[{'id':str(i),'text':text} for i,text in enumerate(texts)]}, '/v1/embedding_records')
            assert status == 200, (status,records)
            record_matrix = np.array([row['embedding'] for row in records['data']])
            record_cos = np.sum(record_matrix*expected,axis=1)/(np.linalg.norm(record_matrix,axis=1)*np.linalg.norm(expected,axis=1))
            assert min(record_cos) >= .999, record_cos
            raw = tokenizer.encode(texts[2], add_special_tokens=False).ids
            chunks = [raw[i:i+7] for i in range(0,len(raw),7)]
            chunk_ids = [chunk+[151643] if spec['model_type']=='qwen3' else [2]+chunk+[1] for chunk in chunks]
            status, chunk_reference = post({'input':chunk_ids})
            assert status == 200
            status, chunk_records = post({'records':[{'id':'chunks','text':texts[2]}], 'chunking':{'max_tokens':7,'overlap_tokens':0}}, '/v1/embedding_records')
            assert status == 200, (status,chunk_records)
            chunk_matrix = np.array([row['embedding'] for row in chunk_records['data']])
            chunk_expected = np.array([row['embedding'] for row in chunk_reference['data']])
            chunk_cos = np.sum(chunk_matrix*chunk_expected,axis=1)/(np.linalg.norm(chunk_matrix,axis=1)*np.linalg.norm(chunk_expected,axis=1))
            assert min(chunk_cos) >= .999, chunk_cos
            status, no_eos = post({'records':[{'id':'no-eos','text':texts[0]}], 'add_eos':False}, '/v1/embedding_records')
            assert status == 200
            status, no_eos_reference = post({'input':ids[0][:-1]})
            assert status == 200
            no_eos_cos = float(np.dot(no_eos['data'][0]['embedding'], no_eos_reference['data'][0]['embedding']))
            assert no_eos_cos >= .999
            status, explicit_eos = post({'records':[{'id':'eos','text':texts[0]}], 'add_eos':True}, '/v1/embedding_records')
            assert status == 200
            assert np.dot(explicit_eos['data'][0]['embedding'], expected[0]) >= .999
            errors = []
            invalid = [{'input':[2**32-1]}, {'input':[[1,2],[2**32-1]]}, {'input':[]}, {'input':ids[0],'dimensions':128}, {'input':ids[0],'encoding_format':'base64'}]
            if spec['model_type']=='embedding_gemma2': invalid += [{'input':[2,t,1]} for t in [258880,258881,258884]]
            for request in invalid:
                status, body = post(request)
                errors.append(status)
                assert status == 400, (request,status,body)
            status, _ = post({'input':ids[0]})
            assert status == 200
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent = list(pool.map(lambda _: post({'input':ids[0]}), range(8)))
            assert all(code == 200 and np.dot(reply['data'][0]['embedding'], expected[0]) >= .999 for code,reply in concurrent)
            chat_status = None
            if spec['model_type'] != 'qwen3':
                chat_status, _ = post({'model':model_id,'messages':[{'role':'user','content':'Hello'}],'max_tokens':1}, '/v1/chat/completions')
                assert 400 <= chat_status < 500, chat_status
            report = {**spec, 'no_eos_override_cosine':no_eos_cos, 'record_cosines':record_cos.tolist(), 'chunk_cosines':chunk_cos.tolist(), 'concurrent_requests':len(concurrent), 'chat_status':chat_status, 'shape':list(matrix.shape), 'direct_cosines':cos.tolist(), 'text_token_cosines':text_cos.tolist(), 'invalid_statuses':errors, 'recovery_status':status, 'pass':True}
            reports.append(report)
            Path(a.output).write_text(json.dumps(reports,indent=2)+'\n')
            print(json.dumps(report),flush=True)
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
